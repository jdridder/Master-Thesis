import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

os.environ["OPENBLAS_NOFORK"] = "1"
os.environ["OPENBLAS_DISABLE_MAIN_THREAD_AFFINITY"] = "1"

import numpy as np
import yaml

CURR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))

sys.path.append(ROOT_DIR)

from models import EtOxModel
from postprocessing.plot import plot_loop
from routines.data_structurizer import DataStructurizer
from routines.setup_routines import *
from simulation.closed_loop_process import run_parallel_mpc_loop
from simulation.data_generation import generate_random_ramp_signal

mpc_perf_cfg = {
    "n_experiments": 1,
    "n_workers": 1,
    "covariance_gain": 3,
    "lam_bed_std": 0.08,
    "tvp_tau": 20,
    # "surrogate_types": ["naive", "pc"],
    "surrogate_types": ["vanilla"],
    "state_dict_folder": {"vanilla": "vanilla", "naive": "vanilla", "pc": "pc"},
    "t_steps": 64,
    "mpc_cfg": {
        "input_bounds": {"lower": 580, "upper": 625},
        "n_horizon": 13,
        "n_robust": 1,
        "mpc_t_step": 2,  # multiple of the fp systems time step
        "control_t_step": 2,  # multiple of the fp systems time step when a new input is applied
        # "uncertainty_values": {"alpha": [1]},
        "uncertainty_values": {"alpha": [[1, 0], [0, 1]]},
        "scenario_weights": [0.9, 0.1],
        "scenarios": ["nominal", "upper"],
        # "scenarios": ["nominal"],
        "lam_Tmax": 1e3,
        "ub_T": 630 / 625,
        "lam_dudt": {"T_c0": 250, "T_c1": 20, "T_c2": 15, "T_c3": 10},
        "lam_X": 2e1,
        "lam_T_Tcool": 1e3,
        "lb_X": 0.5,
        "input_scale": 625,
        "tvp_scale": 0.4,
        "store_full_solution": False,
        "surpress_ipopt_output": False,
        "BLAS_NUM_THREADS": "1",
        "solver_opts": {
            "ipopt": {
                "max_iter": 1000,
                "tol": 1e-4,
                "acceptable_tol": 5e-4,
                "print_level": 5,
                "warm_start_init_point": "yes",
                # "linear_solver": "mumps",
                "linear_solver": "ma57",
                "hessian_approximation": "limited-memory",
            }
        },
    },
}


def main():

    state_dict_path = os.path.abspath("/Users/jandavidridder/Desktop/Masterarbeit/Master-Thesis/experiments/001_poc/2025-12-07/trained_models/vanilla")
    path_to_init_data = os.path.abspath("/Users/jandavidridder/Desktop/Masterarbeit/Master-Thesis/PYTHON/models/EtOxModel/initialization_data.npy")

    sim_cfg_name = "etox_control_task.yaml"
    config_directory = os.path.abspath(os.path.join(ROOT_DIR, "configs"))
    with open(os.path.join(config_directory, sim_cfg_name), "r") as f:
        sim_cfg = yaml.safe_load(f)
    model_cfg_directory = os.path.abspath(os.path.join(ROOT_DIR, "models", sim_cfg["model_name"]))
    with open(os.path.join(model_cfg_directory, "EtOxModel.yaml"), "r") as f:
        model_cfg = yaml.safe_load(f)

    meta_model = EtOxModel(
        model_cfg=model_cfg,
        state_keys=sim_cfg["states"]["keys"],
        input_keys=sim_cfg["inputs"]["all_keys"],
        N_finite_diff=sim_cfg["simulation"]["N_finite_diff"],
    )
    structurizer = DataStructurizer(
        n_initial_measurements=sim_cfg["simulation"]["N_finite_diff"],
        n_measurements=sim_cfg["narx"]["n_measurements"],
        time_horizon=sim_cfg["narx"]["time_horizon"],
        state_keys=sim_cfg["states"]["keys"],
        input_keys=sim_cfg["inputs"]["all_keys"],
        tvp_keys=sim_cfg["tvps"]["keys"],
    )
    kinetic_parameters = meta_model.get_true_parameters(n_batches=mpc_perf_cfg.get("n_experiments"))
    # kinetic_parameters = meta_model.sample_parameters(
    #     n_batches=mpc_perf_cfg.get("n_experiments"),
    #     covariance_gain=mpc_perf_cfg.get("covariance_gain"),
    #     lam_bed_std=mpc_perf_cfg.get("lam_bed_std"),
    # )
    mpc_cfg = mpc_perf_cfg["mpc_cfg"]
    tvp_signals = generate_random_ramp_signal(
        # feature_bounds=[sim_cfg["tvps"]["level_bounds"]],
        feature_bounds=[[0.2, 0.4]],
        num_steps=(mpc_perf_cfg.get("t_steps") + mpc_cfg.get("n_horizon")) * mpc_cfg.get("control_t_step"),
        tau=mpc_perf_cfg.get("tvp_tau"),
        seed=4,
        batch_size=1,
        time_step=sim_cfg["simulation"]["t_step"],
    )

    # load initialization data
    init_data = structurizer.load_data(path_to_init_data, time_step=mpc_cfg["mpc_t_step"])

    init_data = np.expand_dims(init_data, axis=0)
    init_data = np.repeat(init_data, repeats=mpc_perf_cfg.get("n_experiments"), axis=0)
    sim_initial_states = structurizer.get_states_from_data(init_data[:, -1], n_measurements=sim_cfg["simulation"]["N_finite_diff"])
    narx_initial_states = structurizer.reduce_measurements(init_data)
    narx_initial_states = structurizer.to_dompc_vector(narx_initial_states)[..., -1]

    # loop over surrogate types
    data_list = run_parallel_mpc_loop(
        n_workers=mpc_perf_cfg.get("n_workers", 1),
        t_steps=mpc_perf_cfg.get("t_steps"),
        data_structurizer=structurizer,
        meta_model=meta_model,
        mpc_initial_states=narx_initial_states,
        simulator_initial_states=sim_initial_states,
        state_dict_dir=state_dict_path,
        narx_type=SurrogateTypes.Vanilla.value,
        scenarios=mpc_perf_cfg["mpc_cfg"].get("scenarios"),
        physical_params=kinetic_parameters,
        tvp_signals=tvp_signals,
        sim_cfg=sim_cfg,
        mpc_cfg=mpc_perf_cfg.get("mpc_cfg"),
        run_cfg={"save_variable_types": ["_y", "_u", "_tvp", "_aux", "t_wall_total"], "save_as": "return_simulator"},
    )

    tuning_dir = os.path.abspath("/Users/jandavidridder/Desktop/Masterarbeit/Master-Thesis/experiments/003_mpc_tuning")
    run_idx = 0
    for i, f in enumerate(os.listdir(tuning_dir)):
        path = os.path.join(tuning_dir, f)
        if os.path.isdir(path):
            run_idx += 1

    final_plot_dir = os.path.join(tuning_dir, f"{run_idx:03d}_tuning_run")
    os.makedirs(final_plot_dir)
    plot_loop(
        sim_cfg=sim_cfg,
        data=data_list[0],
        var_type="_x",
        animate=False,
        n_measurements=sim_cfg["simulation"]["N_finite_diff"],
        plot_cfg={
            "save_path": final_plot_dir,
            "show_fig": False,
        },
    )
    with open(os.path.join(final_plot_dir, "mpc_cfg.json"), "w") as f:
        f.write(json.dumps(mpc_perf_cfg, indent=4))


if __name__ == "__main__":
    main()
