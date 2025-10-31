import os
import sys

import numpy as np
import yaml
from tqdm import tqdm

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CURR_DIR, "..")
INITIAL_STATE_PATH = os.path.join(ROOT_DIR, "models", "EtOxModel", "initial_state.npy")
RESULTS_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "..", "results"))

sys.path.append(ROOT_DIR)
from do_mpc.data import save_results
from models.EtOxModel.EtOxModel import EtOxModel
from postprocessing.plot import plot_loop
from routines.setup_routines import configure_simulator, make_simulator_tvp_fun, set_p_fun
from simulation.data_generation import generate_initial_state, generate_random_ramp_signal

t_steps = 2**10


def main():
    with open(os.path.join(ROOT_DIR, "configs", "etox_control_task.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT_DIR, "models", "EtOxModel", "EtOxModel.yaml"), "r") as f:
        model_cfg = yaml.safe_load(f)

    etox_model = EtOxModel(model_cfg=model_cfg, state_keys=cfg["states"]["keys"], input_keys=cfg["inputs"]["all_keys"], N_finite_diff=cfg["simulation"]["N_finite_diff"])
    model = etox_model.create_physical_model()
    model.setup()

    u = generate_random_ramp_signal(
        feature_bounds=[cfg["inputs"]["level_bounds"]] * 4,
        num_steps=t_steps,
        tau=20,
        time_step=cfg["simulation"]["t_step"],
        hold_time_range=(0.8, 2),
        ramp_time_range=(0.1, 0.4),
    )

    # u = np.array([590, 590, 650, 610, 620])
    x0 = etox_model.get_initial_state(u0=0.3, Tc_0=620, path="/Users/jandavidridder/Desktop/Masterarbeit/src/PYTHON/MYCODE/models/EtOxModel/initial_state.npy")

    tvp_traj = generate_random_ramp_signal(
        feature_bounds=[cfg["tvps"]["level_bounds"]],
        num_steps=t_steps,
        time_step=cfg["simulation"]["t_step"],
        tau=20,
        # seed=42,
        hold_time_range=(4, 4),
        ramp_time_range=(0.5, 0.5),
    )

    simulator = configure_simulator(simulation_cfg=cfg, simulator_model=model)
    tvp_template = simulator.get_tvp_template()
    tvp_fun = make_simulator_tvp_fun(
        simulation_time_step=cfg["simulation"]["t_step"],
        tvp_template=tvp_template,
        tvp_traj=tvp_traj.flatten(),
        tvp_key="u",
    )
    simulator.set_tvp_fun(tvp_fun)
    kinetic_parameters = etox_model.sample_parameters(covariance_gain=2)
    set_p_fun(simulator, params=kinetic_parameters.flatten())
    simulator.setup()
    simulator.reset_history()
    simulator.x0 = x0[0]
    u = u[0]
    for i in tqdm(range(t_steps), desc="Simulating High Fidelity Model"):
        x_next = simulator.make_step(u0=u[i].reshape((-1, 1)))

    plot_loop(sim_cfg=cfg, data=simulator.data, surrogate_type="full", n_measurements=128)

    # save_results([simulator], result_path=f"{RESULTS_DIR}/full/", result_name="full_model", overwrite=False)


if __name__ == "__main__":
    main()
