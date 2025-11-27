import multiprocessing
from multiprocessing import Process, Queue
from typing import Dict, List, Optional, Type, Union

import numpy as np
from models import EtOxModel
from routines.data_structurizer import DataStructurizer
from routines.setup_routines import SurrogateTypes, configure_dompc_model

from .closed_loop import run_narx_mpc_loop


class MPCLoopProcess(Process):
    def __init__(
        self,
        t_steps: int,
        sim_cfg: Dict,
        meta_model: EtOxModel,
        mpc_cfg: Dict,
        state_dict_dir: str,
        data_structurizer: DataStructurizer,
        physical_params: np.ndarray,
        tvp_signals: np.ndarray,
        mpc_initial_states: np.ndarray,
        simulator_initial_states: np.ndarray,
        narx_type: Type[SurrogateTypes],
        scenarios: List[str],
        result_queue: Optional[Queue] = None,
        run_cfg: Optional[Dict] = None,
        name: Optional[str] = "",
    ) -> None:
        Process.__init__(self, name=name)
        self.cfg = sim_cfg
        self.meta_model = meta_model
        self.result_queue = result_queue
        self.t_steps = t_steps
        self.scenarios = scenarios
        self.data_structurizer = data_structurizer
        self.model_parameter_dir = state_dict_dir
        self.narx_type = narx_type
        self.mpc_initial_states = mpc_initial_states
        self.simulator_initial_states = simulator_initial_states
        self.tvp_signals = tvp_signals
        self.physical_params = physical_params
        self.mpc_cfg = mpc_cfg
        self.run_cfg = run_cfg

    def run(self):
        result = run_narx_mpc_loop(
            t_steps=self.t_steps,
            narx_type=self.narx_type,
            tvp_signals=self.tvp_signals,
            simulator_initial_states=self.simulator_initial_states,
            mpc_initial_states=self.mpc_initial_states,
            physical_params=self.physical_params,
            scenarios=self.scenarios,
            data_structurizer=self.data_structurizer,
            meta_model=self.meta_model,
            proc_name=self.name,
            state_dict_dir=self.model_parameter_dir,
            sim_cfg=self.cfg,
            mpc_cfg=self.mpc_cfg,
            run_cfg=self.run_cfg,
        )
        if result is not None:
            assert self.result_queue is not None, "Provide a result queue to the mpc process."
            self.result_queue.put(result)


def run_parallel_mpc_loop(
    t_steps: int,
    sim_cfg: Dict,
    meta_model: EtOxModel,
    mpc_cfg: Dict,
    state_dict_dir: str,
    data_structurizer: DataStructurizer,
    physical_params: np.ndarray,
    tvp_signals: np.ndarray,
    mpc_initial_states: np.ndarray,
    simulator_initial_states: np.ndarray,
    narx_type: Type[SurrogateTypes],
    scenarios: List[str],
    run_cfg: Optional[Dict] = None,
    n_workers: Optional[int] = 1,
):
    if physical_params is not None and physical_params.ndim < 2:
        physical_params = np.expand_dims(physical_params, axis=0)
        physical_params = physical_params.repeat(axis=0, repeats=mpc_initial_states.shape[0])  # duplicate the parameters for all input trajectories.

    manager = multiprocessing.Manager()
    results_queue = manager.Queue()
    procss = []
    sim_initial_states_batched = np.array_split(simulator_initial_states, n_workers, axis=0)
    mpc_input_signals_batched = np.array_split(mpc_initial_states, n_workers, axis=0)
    tvp_signals_batched = np.array_split(tvp_signals, n_workers, axis=0)
    physical_params_batched = np.array_split(physical_params, n_workers, axis=0)

    for core in range(n_workers):
        # start the processes
        proc = MPCLoopProcess(
            physical_params=physical_params_batched[core],
            tvp_signals=tvp_signals_batched[core],
            mpc_initial_states=mpc_input_signals_batched[core],
            simulator_initial_states=sim_initial_states_batched[core],
            data_structurizer=data_structurizer,
            meta_model=meta_model,
            narx_type=narx_type,
            scenarios=scenarios,
            state_dict_dir=state_dict_dir,
            t_steps=t_steps,
            name=f"Proc {core}",
            result_queue=results_queue,
            mpc_cfg=mpc_cfg,
            sim_cfg=sim_cfg,
            run_cfg=run_cfg,
        )
        procss.append(proc)
        proc.start()
    # wait for all processes to join
    [proc.join() for proc in procss]
    print("Processes joined.")
    results_concat = []
    while not results_queue.empty():
        results_concat.append(results_queue.get())
    if len(results_concat) > 0:
        return results_concat
