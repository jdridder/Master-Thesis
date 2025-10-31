import numpy as np
from casadi import DM, exp, norm_2, vertcat
from do_mpc.model import Model
from do_mpc.model._pod_model import AbsMaxScaler, MinMaxScaler, ProperOrthogonalDecomposition, StandardScaler
from do_mpc.simulator import Simulator


def test_rom_projection():
    full_model = Model(model_type="continuous", symvar_type="SX")
    t_steps = 128
    n_finite_diff = 32  # number of finite difference points
    n_states = 2 * n_finite_diff
    n_components = 16
    n_batches = 3

    t_steps = 128  # number of time steps
    p = {"u": 0.018, "L": 4, "k": 0.008, "h": -0.5}
    del_z = p["L"] / n_finite_diff
    c = full_model.set_variable(var_type="_x", var_name=f"c", shape=(n_finite_diff, 1))
    T = full_model.set_variable(var_type="_x", var_name=f"T", shape=(n_finite_diff, 1))

    rhs_c = [DM(0)]
    rhs_T = [DM(0)]
    for i in range(1, n_finite_diff):
        rhs_c.append(-p["u"] * (c[i] - c[i - 1]) / del_z - p["k"])  # * c[i])  #  * exp(T[i]))
        rhs_T.append(-p["u"] * (T[i] - T[i - 1]) / del_z - p["k"] * p["h"])  # * c[i])  #  * exp(T[i]))
    rhs_c = vertcat(*rhs_c)
    rhs_T = vertcat(*rhs_T)

    full_model.set_rhs(var_name="c", expr=rhs_c)
    full_model.set_rhs(var_name="T", expr=rhs_T)
    full_model.setup()

    simulator = Simulator(full_model)
    simulator.settings.t_step = 0.5
    simulator.setup()

    x0_full = np.zeros((n_batches, 2 * n_finite_diff))
    x0_full[0, 0] = 1
    x0_full[0, n_finite_diff] = 2
    x0_full[1, :n_finite_diff] = 0.2
    x0_full[1, n_finite_diff:] = 0.1
    x0_full[2] = 1
    print(x0_full)
    snapshots = []

    for x0 in x0_full:
        simulator.reset_history()
        simulator.x0 = x0.reshape((-1, 1))
        for _ in range(t_steps):
            simulator.make_step()
        snapshots.append(simulator.data["_x"])
    snapshots = np.array(snapshots)

    scaler = StandardScaler()
    scaler.fit(X=snapshots)
    pod = ProperOrthogonalDecomposition()
    pod.set_scaler(scaler=scaler)

    pod.perform_svd(n_components=n_components, snapshots=snapshots)
    assert pod.phi.shape == (n_states, n_components), f"The shape of the projection matrix is incorrect it should be {(n_states, n_components)} it is {pod.phi.shape}."

    reduced_states = pod.full_to_rom(snapshots[1])
    assert reduced_states.shape == (t_steps, n_components), f"The shape of the reduced states {reduced_states.shape} is incorrect it should be {(t_steps, n_components)}."

    reconstructed_states = pod.rom_to_full(reduced_states)
    assert reconstructed_states.shape == (t_steps, n_states), f"The shape of the reconstructed states {reconstructed_states.shape} is incorrect it should be {(t_steps, n_states)}."

    assert np.allclose(snapshots[1], reconstructed_states, atol=1e-3), f"The reconstructed states do not match the original states. Difference {reconstructed_states - snapshots}."

    full_states_casadi = DM(snapshots[1][-1])
    reduced_states_casadi = pod.full_to_rom(full_states_casadi)
    assert reduced_states_casadi.shape == (n_components, 1), f"The shape of the reduced states in casadi format {reduced_states_casadi.shape} is not equal to the desired shape ({n_components}, 1)"

    reconstructed_states_casadi = pod.rom_to_full(reduced_states_casadi)
    assert (
        reconstructed_states_casadi.shape == full_states_casadi.shape
    ), f"The reconstructed casadi states have a different shape {reconstructed_states_casadi.shape} than the original casadi states {full_states_casadi.shape}."

    casadi_truncation_error = norm_2(full_states_casadi - reconstructed_states_casadi)
    assert casadi_truncation_error <= 1e-3, f"The casadi trucation error {casadi_truncation_error} is greater than the threshold 1e-4."

    rom_parameters = pod.export_parameters()

    del pod
    del scaler
    pod = ProperOrthogonalDecomposition()
    scaler = StandardScaler()
    scaler.import_params(rom_parameters["scaler"])
    pod.import_parameters(rom_parameters)
    pod.set_scaler(scaler=scaler)

    x0_reduced = pod.full_to_rom(x0_full[1])

    reduced_model = pod.reduce(full_model)
    reduced_model.setup()

    assert reduced_model.n_x == n_components, f"The number of states in the reduced model is incorrect it should be {n_components} it is {reduced_model.n_x}."

    simulator = Simulator(reduced_model)
    simulator.settings.t_step = 0.5

    simulator.setup()

    simulator.x0 = x0_reduced.reshape((-1, 1))

    for _ in range(t_steps):
        simulator.make_step()
    reduced_results = simulator.data["_x"]
    reconstructed_states_sim = pod.rom_to_full(reduced_results)

    if __name__ == "__main__":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
        ax[0].plot(snapshots[1, :, 0:n_finite_diff], label="full model", color="blue", alpha=0.3)
        ax[0].plot(reconstructed_states_sim[:, 0:n_finite_diff], label="reconstructed from rom simulation", color="orange", alpha=0.3)
        ax[0].plot(reconstructed_states[:, 0:n_finite_diff], ls="--", label="reconstructed from rom", color="green", alpha=0.3)
        ax[0].set_ylabel("$x_1$")

        ax[1].plot(snapshots[1, :, n_finite_diff:], label="full model", color="blue", alpha=0.3)
        ax[1].plot(reconstructed_states_sim[:, n_finite_diff:], label="rom simulated, then reconstructed", color="orange", alpha=0.3)
        ax[1].plot(reconstructed_states[:, n_finite_diff:], ls="--", label="reconstructed from rom", color="green", alpha=0.3)
        ax[1].set_ylabel("$x_2$")
        ax[1].set_xlabel("$t$ / s")
        # ax[0].set_ylim(0, 2)
        # ax[1].set_ylim(0, 2)

        handles, labels = ax[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax[0].legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.25), ncols=3)

        plt.show()


if __name__ == "__main__":
    test_rom_projection()
