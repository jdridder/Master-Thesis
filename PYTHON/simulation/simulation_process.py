import multiprocessing
from multiprocessing import Process, Queue
from typing import Dict, List, Optional, Type, Union

import numpy as np
from do_mpc.model import Model
from models import EtOxModel
from routines.data_structurizer import DataStructurizer
from routines.setup_routines import SurrogateTypes, configure_dompc_model

from .simulation import simulate


class SimulationProcess(Process):
    def __init__(
        self,
        cfg: Dict,
        result_queue: Queue,
        name: str,
        t_steps: int,
        meta_model: EtOxModel,
        dompc_model_type: Type[SurrogateTypes],
        tvp_signals: np.ndarray,
        input_signals: Union[np.ndarray, List[np.ndarray]],
        initial_states: Union[np.ndarray, List[np.ndarray]],
        run_cfg: Optional[Dict] = None,
        scenario: Optional[str] = None,
        model_parameter_dir: Optional[str] = None,
        data_structurizer: Optional[DataStructurizer] = None,
        model_parameters: Optional[np.ndarray] = None,
        index: Optional[np.ndarray] = None,
    ) -> None:
        Process.__init__(self, name=name)
        self.cfg = cfg
        self.meta_model = meta_model
        self.result_queue = result_queue
        self.t_steps = t_steps
        self.scenario = scenario
        self.input_signals = input_signals
        self.data_structurizer = data_structurizer
        self.model_parameter_dir = model_parameter_dir
        self.dompc_model_type = dompc_model_type
        self.initial_states = initial_states
        self.tvp_signals = tvp_signals
        self.model_params = model_parameters
        self.run_cfg = run_cfg
        self.index = index

    def run(self):
        """Simulate a set of experiments defined by multiple input_trajectories and multiple parameter combination with a given set of parameters."""
        do_mpc_model = configure_dompc_model(  # create in the process because MX based models cannot be pickled.
            model_type=self.dompc_model_type,
            sim_cfg=self.cfg,
            scenario=self.scenario,
            data_structurizer=self.data_structurizer,
            meta_model=self.meta_model,
            model_parameter_dir=self.model_parameter_dir,
        )
        result_arr = simulate(
            simulation_cfg=self.cfg,
            n_time_steps=self.t_steps,
            do_mpc_model=do_mpc_model,
            tvp_signals=self.tvp_signals,
            initial_states=self.initial_states,
            physical_params=self.model_params,
            input_signals=self.input_signals,
            process_name=self.name,
            index=self.index,
            run_cfg=self.run_cfg,
        )
        if result_arr is not None:
            self.result_queue.put(result_arr)


def run_parallel_simulations(
    simulation_cfg: Dict,
    meta_model: EtOxModel,
    model_type: Type[SurrogateTypes],
    t_steps: int,
    tvp_signals: np.ndarray,
    input_signals: np.ndarray,
    initial_states: np.ndarray,
    run_cfg: Optional[Dict] = None,
    model_parameter_dir: Optional[str] = None,
    scenario: Optional[str] = None,
    data_structurizer: Optional[DataStructurizer] = None,
    index: Optional[np.ndarray] = None,
    n_workers: Optional[int] = 10,
    model_params: Optional[np.ndarray] = None,
):
    """Either the input signal is constant and given across all simulations or it is randomly generated every signle time."""
    assert initial_states.shape[0] == input_signals.shape[0], f"The batch size of input signals {input_signals.shape[0]} and initial states {initial_states.shape[0]} must match."
    if model_params is not None and model_params.ndim < 2:
        model_params = np.expand_dims(model_params, axis=0)
        model_params = model_params.repeat(axis=0, repeats=initial_states.shape[0])  # duplicate the parameters for all input trajectories.
    if index is None:
        index = np.arange(input_signals.shape[0])

    manager = multiprocessing.Manager()
    results_queue = manager.Queue()
    procss = []
    initial_states_batched = np.array_split(initial_states, n_workers, axis=0)
    input_signals_batched = np.array_split(input_signals, n_workers, axis=0)
    tvp_signals_batched = np.array_split(tvp_signals, n_workers, axis=0)
    model_params_batched = np.array_split(model_params, n_workers, axis=0)
    index_batched = np.array_split(index, n_workers, axis=0)

    for core in range(n_workers):
        # start the processes
        proc = SimulationProcess(
            name=f"Proc {core}",
            cfg=simulation_cfg,
            model_parameter_dir=model_parameter_dir,
            dompc_model_type=model_type,
            meta_model=meta_model,
            scenario=scenario,
            data_structurizer=data_structurizer,
            result_queue=results_queue,
            t_steps=t_steps,
            tvp_signals=tvp_signals_batched[core],
            initial_states=initial_states_batched[core],
            input_signals=input_signals_batched[core],
            model_parameters=model_params_batched[core],
            index=index_batched[core],
            run_cfg=run_cfg,
        )
        procss.append(proc)
        proc.start()
    # wait for all processes to join
    [proc.join() for proc in procss]
    results_concat = []
    while not results_queue.empty():
        results_concat.append(results_queue.get())
    if len(results_concat) > 0:
        results_concat = np.concatenate(results_concat, axis=0)
    print("Processes joined.")
    return results_concat
