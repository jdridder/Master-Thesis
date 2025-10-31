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
        result_queue: Queue,
        name: str,
        cfg: Dict,
        meta_model: EtOxModel,
        dompc_model_type: Type[SurrogateTypes],
        model_parameter_dir: str,
        t_steps: int,
        scenario: str,
        data_structurizer: DataStructurizer,
        save_kwargs: Dict,
        tvp_signals: np.ndarray,
        input_signals: Union[np.ndarray, List[np.ndarray]],
        initial_states: Union[np.ndarray, List[np.ndarray]],
        model_parameters: np.ndarray = None,
        index: Optional[np.ndarray] = None,
        integration_opts: Optional[Dict] = None,
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
        self.save_kwargs = save_kwargs
        self.index = index
        self.integration_opts = integration_opts

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
        simulate(
            simulation_cfg=self.cfg,
            n_time_steps=self.t_steps,
            do_mpc_model=do_mpc_model,
            tvp_signals=self.tvp_signals,
            initial_states=self.initial_states,
            model_parameters=self.model_params,
            input_signals=self.input_signals,
            process_name=self.name,
            index=self.index,
            integration_opts=self.integration_opts,
            **self.save_kwargs,
        )


def run_parallel_simulations(
    simulation_cfg: Dict,
    model_parameter_dir: str,
    meta_model: Dict,
    model_type: Type[SurrogateTypes],
    scenario: str,
    data_structurizer: DataStructurizer,
    t_steps: int,
    tvp_signals: np.ndarray,
    input_signals: np.ndarray,
    initial_states: np.ndarray,
    save_kwargs: Dict,
    index: Optional[np.ndarray] = None,
    n_workers: int = 10,
    model_params: np.ndarray = None,
    integration_opts: Optional[Dict] = None,
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
            save_kwargs=save_kwargs,
            integration_opts=integration_opts,
        )
        procss.append(proc)
        proc.start()
    # wait for all processes to join
    [proc.join() for proc in procss]
    print("Processes joined.")
