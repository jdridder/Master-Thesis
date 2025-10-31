import json
import multiprocessing
import os
import sys
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Optional, Tuple

import l4casadi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from bofire.data_models.acquisition_functions.api import qLogEI
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.strategies.api import SoboStrategy
from do_mpc.data import Data, MPCData, load_results, save_results
from do_mpc.model import Model
from do_mpc.simulator import Simulator

CURR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
CONFIG_NAME = "etox_control_task.yaml"
CONFIG_PATH = os.path.abspath(os.path.join(ROOT_DIR, "configs", CONFIG_NAME))
MODEL_CONFIG_PATH = os.path.abspath(os.path.join(ROOT_DIR, "models", "EtOxModel", "EtOxModel.yaml"))
sys.path.append(ROOT_DIR)
from models import EtOxModel
from postprocessing.performance_metrics import calculate_control_effort, calculate_mean_constraint_vio, calculate_mean_selectivity
from postprocessing.Visualizer import Visualizer
from run_closed_loop import configure_mpc, configure_simulator, run_closed_loop

from PYTHON.MYCODE.simulation.data_generation import DataStructurizer, load_data

objective = lambda selectivity, X_violation, T_violation, control_effort: selectivity - X_violation - T_violation - control_effort


def run_bayse_opt():
    n_measurements = 4
    n_back = 8

    with open(CONFIG_PATH, "r") as f:
        simulation_cfg = yaml.safe_load(f)
    with open(MODEL_CONFIG_PATH, "r") as f:
        meta_model_cfg = yaml.safe_load(f)

    objective = MaximizeObjective()
    output_feature = ContinuousOutput(key="f", objective=objective)
    r_penalty = ContinuousInput(key="lg r_penalty", bounds=(-3, 6))  # this is in logarithmic scale it means 10**x
    # lam_temperature = ContinuousInput(key="T_penalty", bounds=(0, 10))
    lam_conversion = ContinuousInput(key="lg X_penalty", bounds=(0, 10))
    domain = Domain(
        inputs=Inputs(features=[r_penalty, lam_conversion]),
        outputs=Outputs(features=[output_feature]),
    )
    sobo_strategy = SoboStrategy.make(domain=domain, acquisition_function=qLogEI(), seed=19)

    # Setup Control Loop
    structurizer = DataStructurizer(
        n_measurements=n_measurements, time_horizon=n_back, state_keys=simulation_cfg["states"]["keys"], input_keys=simulation_cfg["inputs"]["all_keys"], tvp_keys=simulation_cfg["tvps"]["keys"]
    )
    meta_model = EtOxModel(
        model_cfg=meta_model_cfg, state_keys=simulation_cfg["states"]["keys"], input_keys=simulation_cfg["inputs"]["all_keys"], N_finite_diff=simulation_cfg["simulation"]["N_finite_diff"]
    )
    simulation_kwargs = {
        "time_steps": 128,
        "structurizer": structurizer,
        "simulation_cfg": simulation_cfg,
        "meta_model": meta_model,
        # "results_path": f"{ROOT_DIR}/results/mpc_tuning/",
    }

    candidates = domain.inputs.sample(8)
    results = run_parallel_experiments(candidates, simulation_kwargs=simulation_kwargs, n_workers=8)
    all_results = results.copy()

    for _ in range(5):
        sobo_strategy.tell(results)
        candidates = sobo_strategy.ask(8)
        results = run_parallel_experiments(candidates, simulation_kwargs=simulation_kwargs, n_workers=8)
        all_results = pd.concat([all_results, results], axis=0, ignore_index=True)
        json_file = all_results.to_json()
        with open("results/mpc_tuning/bo.json", "w") as f:
            f.write(json_file)

    print(all_results)

    # Run the optimization loop


def run_parallel_experiments(candidates: pd.DataFrame, simulation_kwargs: Dict, n_workers: int = 10) -> pd.DataFrame:
    manager = multiprocessing.Manager()
    results_queue = manager.Queue()
    procss = []
    sub_candidates = np.array_split(candidates, n_workers, axis=0)

    for i in range(n_workers):
        proc = ClosedLoopProcess(name=f"core_{i}", result_queue=results_queue, candidates=sub_candidates[i], simulation_kwargs=simulation_kwargs)
        procss.append(proc)
        proc.start()
    [proc.join() for proc in procss]
    print("Processes joined.")
    # empty the queue
    results = []
    while not results_queue.empty():
        res = results_queue.get(timeout=5)
        results.append(res)
    results = pd.concat(results, axis=0, ignore_index=True)
    return results


class ClosedLoopProcess(Process):
    def __init__(self, name: str, result_queue: Queue, candidates: pd.DataFrame, simulation_kwargs: Dict):
        super().__init__(name=name)
        self.result_queue = result_queue
        self.candidates = candidates
        self.simulation_kwargs = simulation_kwargs
        self.is_initial_run = True
        self.simulator = None
        self.mpc_surrogate = None
        self.initial_state_data = None

    def run(self):
        if self.is_initial_run:  # create in child process to avoid pickling error as MX vars are not picklable
            self.simulator, self.mpc_surrogate, self.initial_state_data = prepare_mpc_configuration(
                data_structurizer=self.simulation_kwargs["structurizer"], simulation_cfg=self.simulation_kwargs["simulation_cfg"], meta_model=self.simulation_kwargs["meta_model"]
            )
            self.is_initial_run = False
        try:
            results = run_experiments(
                candidates=self.candidates, simulator=self.simulator, mpc_surrogate=self.mpc_surrogate, initial_state_data=self.initial_state_data, simulation_kwargs=self.simulation_kwargs
            )
            self.result_queue.put(results)
        except:
            print("An error occurred while trying to run the experiments.")


def run_experiments(candidates: pd.DataFrame, simulator: Simulator, mpc_surrogate: Model, initial_state_data: Dict, simulation_kwargs: Dict) -> pd.DataFrame:
    objective_results = []
    for _, row in candidates.iterrows():
        objective_result = run_mpc_configuration(
            simulator=simulator,
            mpc_surrogate=mpc_surrogate,
            initial_state_data=initial_state_data,
            r_penalty=10 ** row["lg r_penalty"],
            X_penalty=10 ** row["lg X_penalty"],
            **simulation_kwargs,
        )
        objective_results.append(objective_result)
    objective_results = pd.DataFrame(objective_results)
    candidates.reset_index(inplace=True, drop=True)
    return pd.concat([candidates, objective_results], axis=1)


def prepare_mpc_configuration(simulation_cfg: Dict, data_structurizer: DataStructurizer, meta_model: EtOxModel) -> Tuple[Simulator, Model, Dict]:
    r = 24  # rank of the reduced input feature space
    time_start = 10
    mpc_surrogate = configure_mpc_surrogate(
        data_structurizer=data_structurizer,
        super_model=meta_model,
        simulation_cfg=simulation_cfg,
        reduced_rank=r,
    )
    simulator = configure_simulator(meta_model=meta_model, simulation_cfg=simulation_cfg)

    N_trajectories = 32
    trajectory_idx = 6
    data_set_name = "test_100T"
    full_data = load_data(simulation_cfg["model_name"], data_set_name, num_trajectories=N_trajectories, num_time_steps=time_start + 1)[trajectory_idx]
    data = data_structurizer.reduce_measurements(full_data, simulation_cfg["simulation"]["N_finite_diff"])
    initial_state_data = {"simulator": full_data[time_start], "mpc": data[: time_start + 1]}  # mpc data has to include the data at the time instant time_start
    return simulator, mpc_surrogate, initial_state_data


def run_mpc_configuration(
    simulator: Simulator,
    mpc_surrogate: Model,
    initial_state_data: Dict,
    structurizer: DataStructurizer,
    simulation_cfg: Dict,
    meta_model: EtOxModel,
    time_steps: int,
    r_penalty: float,
    X_penalty: float,
    results_path: str = None,
) -> Dict:
    # This must be looped for the Bayesian Optimization loop
    solver_opts = {"ipopt": {"print_level": 0, "linear_solver": "mumps"}}
    mpc = configure_mpc(
        mpc_surrogate=mpc_surrogate,
        data_structurizer=structurizer,
        simulation_cfg=simulation_cfg,
        r_penalty=r_penalty,
        X_penalty=X_penalty,
        mpc_solver_opts=solver_opts,
        surpress_ipopt_output=True,
    )
    results = run_closed_loop(
        simulation_cfg=simulation_cfg,
        simulator=simulator,
        mpc=mpc,
        meta_model=meta_model,
        time_steps=time_steps,
        data_structurizer=structurizer,
        initial_state_data=initial_state_data,
    )
    if results_path is not None:
        save_results([mpc, simulator], result_path=results_path, overwrite=False)
    violation = calculate_mean_constraint_vio(results=results, X_min=0.6, T_max=630)
    average_selectivity = calculate_mean_selectivity(results=results)
    control_effort = calculate_control_effort(results=results, T_max=630)
    objective_result = objective(average_selectivity, violation["X"], violation["T"], control_effort)
    return {"S_avg": average_selectivity[0], "X_vio": violation["X"][0], "T_vio": violation["T"][0], "u_effort": control_effort[0], "f": objective_result[0]}


if __name__ == "__main__":
    run_bayse_opt()
