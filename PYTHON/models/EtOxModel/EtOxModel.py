import inspect
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy
from casadi import *
from casadi.tools import *
from do_mpc.model import Model
from models.MyModel import MyModel


class EtOxModel(MyModel):
    def __init__(self, model_cfg: Dict, state_keys: List[str], input_keys: List[str], N_finite_diff: int):
        super().__init__(model_cfg=model_cfg, state_keys=state_keys, input_keys=input_keys, N_finite_diff=N_finite_diff)
        # Constraints are hardcoded here.
        # self.constraints = [lambda x_A_out, x_A_in: 1 - x_A_out / x_A_in - 0.98]
        self.map_to_scenario_idx = {"nominal": 0, "upper": 1, "lower": 2}
        self.bc = self.calculate_bc()
        self.kinetic_param_keys = ["k1", "EA1", "k2", "EA2", "lam_bed"]
        self.mean_kinetic_parameters = np.array([1.56214535e01, 7.48563390e04, 1.62725568e01, 8.98538394e04, 0.4037066551257339])
        self.parameter_bounds = np.array([[1.8e01, 85000, 2.0e1, 98000, 0.5], [1.0e01, 65000, 1.2e1, 80000, 0.3]])  # order is 0 = upper, 1 = lower

        self.A = self.p["x_E_in"] / self.p["T_in"] * self.p["p0"] / self.p["R_g"]
        self.eps = self.p["eps_0"] + (1 - self.p["eps_0"]) * 0.526 * self.p["dp"] / self.p["d"]
        self.cp_g = self.p["cp_g"] / self.p["M_tot"]  # mass heat capacity of the gas phase
        self.cp_tot = self.eps * self.cp_g + (1 - self.eps) * self.p["cp_p"]
        self.nu_E, self.nu_O2, self.nu_CO2 = (-1, -1), (-0.5, -3), (0, 2)
        self.nu_EO, self.nu_H2O = (1, 0), (0, 2)

    def calculate_bc(self) -> Dict:
        """Calculates the boundary conditions for the fictional dimensionless state chi = x / T."""
        bc = {}
        chi = lambda x: x / self.p["x_E_in"]
        for key in self.bc.keys():
            if key.startswith("x"):
                bc[key.replace("x", "chi")] = chi(x=self.bc[key])
            else:
                bc[key] = self.bc[key] / self.p["T_in"]
        return bc

    def get_bc_for_all_measurements(self, n_measurements: int, n_batches: int = 1) -> np.ndarray:
        """Returns the boundary conditions for all measurements as a numpy array."""
        boundary_conditions = np.array(list(self.bc.values()))
        boundary_conditions = boundary_conditions.repeat(n_measurements)
        boundary_conditions = np.expand_dims(boundary_conditions, axis=0)
        boundary_conditions = np.repeat(boundary_conditions, repeats=n_batches, axis=0)
        return boundary_conditions

    def get_scaling_factors(self, n_measurements: int) -> np.ndarray:
        concentration_factor = self.bc["c_E"]
        temperature_factor = self.bc["T"]
        concentration_scales = np.ones(n_measurements * (len(self.state_keys) - 1)) * concentration_factor
        temperature_scales = np.ones(n_measurements) * temperature_factor
        return np.concatenate([concentration_scales, temperature_scales])

    # Hardcoded functions depending on the Model, objective, constraints, parameter sampling, etc.
    # -----------------------

    # def get_time_constant_values(self) -> Dict:
    #     """Returns the time constants of the system as a dictionary."""
    #     t_constants = {
    #         "t_r1": self.bc["c_E"] ** 0.6 * self.bc["chi_O2"] ** 0.5 / self.p["k1"],
    #         "t_r2": self.bc["c_E"] ** 0.5 * self.bc["chi_O2"] ** 0.5 / self.p["k2"],
    #         "t_u": self.p["L"],
    #         "t_h": 20,
    #     }
    #     return t_constants

    def get_true_parameters(self, n_batches: int = None) -> np.ndarray:
        params = np.zeros(len(self.kinetic_param_keys))
        for i, key in enumerate(self.kinetic_param_keys):
            params[i] = float(self.p[key])
        if n_batches is not None:
            params = np.expand_dims(params, axis=0)
            params = np.repeat(params, repeats=n_batches, axis=0)
        return params

    def get_parameter_scenario(self, scenario: str = None) -> np.ndarray:
        # order is nominal, upper, lower bound
        parameters = np.concatenate([self.mean_kinetic_parameters.reshape((1, -1)), self.parameter_bounds], axis=0)
        parameters = self._apply_exp_to_lnk(parameters=parameters)
        idx = self.map_to_scenario_idx[scenario]
        # s = (
        #     slice(0, parameters.shape[0])
        #     if scenario is None
        #     else slice(self.map_to_scenario_idx[scenario], self.map_to_scenario_idx[scenario] + 1)
        # )
        return parameters[idx]

    def _apply_exp_to_lnk(self, parameters: np.ndarray):
        parameters[..., 0] = np.exp(parameters[..., 0])
        parameters[..., 2] = np.exp(parameters[..., 2])
        return parameters

    def sample_parameters(self, n_batches: int = 1, covariance_gain: float = 1, lam_bed_std: float = 0.05) -> np.ndarray:
        """Samples a set of parameters."""
        # covariance matrix of the parameter fitting # [W m1- K-1]
        covariance = np.array(
            [
                [1.75771835e-01, 8.84054955e02, 1.90226410e-01, 9.56652289e02],
                [8.84054955e02, 4.51587249e06, 9.56722528e02, 4.88651035e06],
                [1.90226410e-01, 9.56722528e02, 2.05897698e-01, 1.03542590e03],
                [9.56652289e02, 4.88651035e06, 1.03542590e03, 5.28826120e06],
            ]
        )
        # order is ln k1, EA1, ln k2, EA2
        rng = np.random.default_rng()
        # create new generator in case of parallel execution
        sampled = rng.multivariate_normal(self.mean_kinetic_parameters[:4], covariance * covariance_gain, size=n_batches)
        parameters = self._apply_exp_to_lnk(parameters=sampled)
        lam_sampled = rng.normal(self.mean_kinetic_parameters[-1], lam_bed_std, size=n_batches)
        parameters = np.insert(parameters, 4, lam_sampled, axis=-1)
        return parameters

    def get_balance_constraint_matrix(self, num_stacks: int = 1, include_temp_as_zero: Optional[bool] = True) -> Tuple[np.ndarray]:
        """Returns the matrix A and the corresponding vector b that define the linear equality constraints of the model to ensure physical consistency.
        These are intended to be incoorporated into an casadi optimization layer function which serves as the last layer of a neural network as surrogate model.
        The matrices are hard-coded depending on the model equations. They are formulated with respect to the rescaled (real) physical values of the quantities.
        The integer num_stacks indicates the length of the blocks of the unique states with increasing z coordinate. If so, the matrix A must be duplicated in such a way, that
        the constraints are valid for each point in z.

        Returns:
            Tuple[np.ndarray]: A tuple including the matrix A of shape (n_states, n_states) and a vector b of shape (n_states, 1)
        """
        # vector of states x_current at the current time step, that serves as the reference for thermodynamic trajectories
        A = scipy.linalg.null_space(self.model_cfg["stoiciometric_matrix"]).T
        # Append zeros for the temperature state, which is not part of the constraint
        if include_temp_as_zero:
            A = np.concatenate([A, np.zeros((A.shape[0], 1))], axis=1)  # E, O2, EO, H2O, CO2, T, include temperature as zero
        I = np.eye(num_stacks)
        # create a diagonal block matrix to apply the constraints for each point in space
        A = np.kron(A, I)
        # b = np.zeros(A.shape[0])
        # b = np.tile(b, reps=num_stacks)
        return A

    def get_element_species_matrix(self, num_stacks: int = 1, include_temp_as_zero: Optional[bool] = True) -> np.ndarray:
        element_species_matrix = np.array(self.model_cfg.get("element_species_matrix"))
        if include_temp_as_zero:
            element_species_matrix = np.concatenate([element_species_matrix, np.zeros((element_species_matrix.shape[0], 1))], axis=1)  # E, O2, EO, H2O, CO2, T, include temperature as zero
        I = np.eye(num_stacks)
        element_species_matrix = np.kron(element_species_matrix, I)
        return element_species_matrix

    def lambda_bed(self, eps: float):
        """Calculates the effective heat conductivity of the bed and the fluid"""
        lam_p = (1 - self.p["eps_p"]) * self.p["lam_pm"]
        lam_g = self.p["lam_g"]
        k_p = lam_p / lam_g
        B = 1.25 * ((1 - eps) / eps) ** (10 / 9)
        N = 1 - B / k_p
        k_c = 2 / N * (B / N**2 * (k_p - 1) / k_p * np.log(k_p / B) - 0.5 * (B + 1) - (B - 1) / N)
        k_bed = 1 - np.sqrt(1 - eps) + np.sqrt(1 - eps) * k_c
        lam_bed = k_bed * lam_g
        return lam_bed

    def _p_i(self, chi_i, T):
        return T * chi_i * self.p["x_E_in"] * self.p["p0"]

    def _rho_g(self, T):
        return self.p["M_tot"] * self.p["p0"] / (self.p["R_g"] * T * self.p["T_in"])

    def _rho_tot(self, T, eps):
        return eps * self._rho_g(T) + (1 - eps) * self.p["rho_p"]

    def _mu_g(self, T):
        return self.p["A_mu"] * (T * self.p["T_in"]) ** 2 + self.p["B_mu"] * T * self.p["T_in"] + self.p["F_mu"]

    # dimensionless numbers
    def _Re(self, T, u_flow):
        return self.p["dp"] * u_flow * self._rho_g(T) / self._mu_g(T)

    def _Pr(self, T, cp_g):
        return self._mu_g(T) * cp_g / self.p["lam_g"]

    def _Nu_w(self, T, lam_bed, cp_g, u_flow):
        return 0.19 * self._Re(T, u_flow) ** (3 / 4) * self._Pr(T, cp_g) ** (1 / 3) + lam_bed / self.p["lam_g"] * (1.3 + 5 * self.p["dp"] / self.p["d"])

    def _alpha_w(self, T, lam_bed, cp_g, u_flow):
        return self._Nu_w(T, lam_bed, cp_g, u_flow) * self.p["lam_g"] / self.p["dp"]

    def _r1(self, chi_E, chi_O2, chi_CO2, T, k1, EA1):
        return (
            self.p["act"]
            * self.p["eta"]
            * self.p["rho_p"]
            * k1
            * exp(-EA1 / (self.p["R_g"] * T * self.p["T_in"]))
            * self._p_i(chi_E, T) ** 0.6
            * self._p_i(chi_O2, T) ** 0.5
            * (1 + self.p["K1"] * exp(self.p["Tads1"] / (T * self.p["T_in"])) * self._p_i(chi_CO2, T)) ** (-1)
        )

    def _r2(self, chi_E, chi_O2, chi_CO2, T, k2, EA2):
        return (
            self.p["act"]
            * self.p["eta"]
            * self.p["rho_p"]
            * k2
            * exp(-EA2 / (self.p["R_g"] * T * self.p["T_in"]))
            * self._p_i(chi_E, T) ** 0.5
            * self._p_i(chi_O2, T) ** 0.5
            * (1 + self.p["K2"] * exp(self.p["Tads2"] / (T * self.p["T_in"])) * self._p_i(chi_CO2, T)) ** (-1)
        )

    def get_initial_state(self, u0: float = 0.3, Tc_0: float = 615, n_batches: int = 1, path: str = None) -> np.ndarray:
        path = "/Users/jandavidridder/Desktop/Masterarbeit/src/PYTHON/MYCODE/models/EtOxModel/initial_state.npy"
        if os.path.exists(path):
            x = np.load(file=path)
        else:
            c = (1 - self.eps) / (u0 * self.A)
            k1, EA1, k2, EA2, lam_bed = self.sample_parameters().flatten()
            x0 = np.array(list(self.bc.values()))
            x = np.ones_like(self.state_keys, dtype=np.float32)
            z_grid = np.linspace(0, self.p["L"], self.N_finite_diff)

            def dx(z, x):
                dx = np.empty_like(x)
                dx[0] = c * (self.nu_E[0] * self._r1(x[0], x[1], x[4], x[5], k1=k1, EA1=EA1) + self.nu_E[1] * self._r2(x[0], x[1], x[4], x[5], k2=k2, EA2=EA2))
                dx[1] = c * (self.nu_O2[0] * self._r1(x[0], x[1], x[4], x[5], k1=k1, EA1=EA1) + self.nu_O2[1] * self._r2(x[0], x[1], x[4], x[5], k2=k2, EA2=EA2))
                dx[2] = c * self.nu_EO[0] * self._r1(x[0], x[1], x[4], x[5], k1=k1, EA1=EA1)
                dx[3] = c * self.nu_H2O[1] * self._r2(x[0], x[1], x[4], x[5], k2=k2, EA2=EA2)
                dx[4] = c * self.nu_CO2[1] * self._r2(x[0], x[1], x[4], x[5], k2=k2, EA2=EA2)
                dx[5] = (
                    1
                    / (u0 * self._rho_tot(x[5], self.eps) * self.cp_tot)
                    * (
                        -self.p["dHR1"] * self._r1(x[0], x[1], x[4], x[5], k1, EA1)
                        - self.p["dHR2"] * self._r2(x[0], x[1], x[4], x[5], k2, EA2)
                        + 4 / self.p["d"] * self._alpha_w(x[5], lam_bed, self.cp_g, u0) * (Tc_0 / self.p["T_in"] - x[5])
                    )
                )
                return dx

            x = scipy.integrate.solve_ivp(dx, (0, self.p["L"]), x0, t_eval=z_grid)["y"]
        x = x.reshape((-1, 1))
        x = np.expand_dims(x, axis=0)
        x = np.repeat(x, repeats=n_batches, axis=0)
        return x

    def setup_physics(self, model: Model) -> Model:
        """Writes all system variables and equations into the Model object."""
        states = {}
        for state_key in self.state_keys:
            # setup the main state variables
            # unique state are sorted separetely according to the stacking of the snapshots for POD: c_0, c_1, ... c_N, T_0, ..., T_N
            states[state_key] = model.set_variable(
                var_type="_x",
                var_name=state_key,
                shape=(self.N_finite_diff, 1),
            )
        # parameters
        p = self.p
        # inputs
        inputs = {input_key: model.set_variable("_u", input_key) for input_key in self.input_keys}

        u_flow = model.set_variable(var_type="_tvp", var_name="u")
        k1 = model.set_variable("_p", "k1")
        EA1 = model.set_variable("_p", "EA1")
        k2 = model.set_variable("_p", "k2")
        EA2 = model.set_variable("_p", "EA2")
        lam_bed = model.set_variable("_p", "lam_bed")

        A = self.A
        eps = self.eps
        cp_g = self.cp_g
        cp_tot = self.cp_tot

        # Reaction rate expressions

        # System Equations
        # -----------------------
        # Lambda functions for the time derivatives as a function of the finite difference
        lam_expressions = {
            # Gas phase equations
            "chi_E": (
                lambda chi_E, chi_E_b, chi_O2, chi_CO2, T: -u_flow * (chi_E - chi_E_b) / self.Del_z
                + (1 - eps) / A * (self.nu_E[0] * self._r1(chi_E, chi_O2, chi_CO2, T, k1, EA1) + self.nu_E[1] * self._r2(chi_E, chi_O2, chi_CO2, T, k2, EA2))
            ),
            "chi_O2": (
                lambda chi_O2, chi_O2_b, chi_E, chi_CO2, T: -u_flow * (chi_O2 - chi_O2_b) / self.Del_z
                + (1 - eps) / A * (self.nu_O2[0] * self._r1(chi_E, chi_O2, chi_CO2, T, k1, EA1) + self.nu_O2[1] * self._r2(chi_E, chi_O2, chi_CO2, T, k2, EA2))
            ),
            "chi_EO": (lambda chi_EO, chi_EO_b, chi_E, chi_O2, chi_CO2, T: -u_flow * (chi_EO - chi_EO_b) / self.Del_z + (1 - eps) / A * (self.nu_EO[0] * self._r1(chi_E, chi_O2, chi_CO2, T, k1, EA1))),
            "chi_H2O": (
                lambda chi_H2O, chi_H2O_b, chi_E, chi_O2, chi_CO2, T: -u_flow * (chi_H2O - chi_H2O_b) / self.Del_z + (1 - eps) / A * (self.nu_H2O[1] * self._r2(chi_E, chi_O2, chi_CO2, T, k2, EA2))
            ),
            "chi_CO2": (lambda chi_CO2, chi_CO2_b, chi_E, chi_O2, T: -u_flow * (chi_CO2 - chi_CO2_b) / self.Del_z + (1 - eps) / A * (self.nu_CO2[1] * self._r2(chi_E, chi_O2, chi_CO2, T, k2, EA2))),
            "T": lambda T, T_b, chi_E, chi_O2, chi_CO2, T_c: (
                -u_flow * (T - T_b) / self.Del_z
                + (self._rho_tot(T, eps) * cp_tot) ** (-1)
                * (
                    (1 - eps) / p["T_in"] * (-p["dHR1"] * self._r1(chi_E, chi_O2, chi_CO2, T, k1, EA1) - p["dHR2"] * self._r2(chi_E, chi_O2, chi_CO2, T, k2, EA2))
                    + 4 / p["d"] * self._alpha_w(T, lam_bed, cp_g, u_flow) * (T_c / p["T_in"] - T)
                )
            ),
        }
        arg_names = {}
        for key in lam_expressions.keys():
            # get the argument names of the lambda function that defines the time derivative
            sig = inspect.signature(lam_expressions[key])
            arg_names[key] = list(sig.parameters.keys())

        inlet_cond = self.bc

        for state_key in self.state_keys:
            j = 0
            rhs_args = self.get_rhs_args(i=0, j=j, states=states, inputs=inputs, expr_key=state_key, arg_names=arg_names, bc_expr=inlet_cond[state_key])
            rhs_expressions = lam_expressions[state_key](**rhs_args)
            for i in range(1, self.N_finite_diff):
                if i % (self.N_finite_diff // p["n_c"]) == 0 and i > 0:
                    j += 1
                rhs_args = self.get_rhs_args(i, j, states, inputs, state_key, arg_names)
                rhs_expressions = vertcat(
                    rhs_expressions,
                    lam_expressions[state_key](**rhs_args),
                )
            model.set_rhs(var_name=state_key, expr=rhs_expressions)

        # expressions for debugging and monitoring only comment for simulation
        # for i in np.linspace(0, 127, 4):
        #     i = int(i)
        #     model.set_expression(f"Re_{i}", Re(T=states["T"][i]))
        #     model.set_expression(f"Nu_w_{i}", Nu_w(T=states["T"][i]))
        #     model.set_expression(f"alph_w_{i}", alph_w(T=states["T"][i]))
        #     model.set_expression(f"rho_tot{i}", rho_tot(T=states["T"][i]))

        delta_c_E = self.bc["chi_E"] - states["chi_E"][-1]
        selectivity = (states["chi_EO"][-1] - self.bc["chi_EO"]) / delta_c_E
        conversion = delta_c_E / self.bc["chi_E"]

        model.set_expression(expr_name="S", expr=selectivity)
        model.set_expression(expr_name="X", expr=conversion)
        return model
