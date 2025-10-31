from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np
from casadi import *
from do_mpc.model import Model
from do_mpc.simulator import Simulator


class MyModel:
    def __init__(self, model_cfg: Dict, state_keys: List[str], input_keys: List[str], N_finite_diff: int):
        self.model_cfg = model_cfg
        self.state_keys = state_keys
        self.input_keys = input_keys
        self.N_finite_diff = N_finite_diff
        self.n_x = len(state_keys)
        self.n_u = len(input_keys)
        self.p = self.parse_parameters_to_float(model_cfg["equation_parameters"])
        self.Del_z = self.p["L"] / (self.N_finite_diff - 1)  # length of the finite difference Elements [m]
        self.bc = model_cfg["boundary_conditions"]
        self.bounds = model_cfg["bounds"]

    def parse_parameters_to_float(self, equation_parameters: Dict) -> Dict:
        for key in equation_parameters.keys():
            item = equation_parameters[key]
            if isinstance(item, str):
                equation_parameters[key] = float(item)
        return equation_parameters

    def generate_bounds(self) -> Dict:
        """Returns the bounds on all states in spatial direction."""
        bounds = dict(self.bounds)
        for var_key in bounds.keys():
            if var_key in self.state_keys:
                var_bounds = np.ones((len(self.state_keys), self.N_finite_diff))
                var_bounds[0, :] = var_bounds[0, :] * bounds[var_key][0]
                var_bounds[1, :] = var_bounds[1, :] * bounds[var_key][1]
                bounds[var_key] = var_bounds
            else:
                bounds[var_key] = np.array(bounds[var_key])
        return bounds

    @staticmethod
    def get_rhs_args(i: int, j: int, states: Dict, inputs: Dict, expr_key: str, arg_names: Dict, bc_expr: Union[float, SX] = None) -> Dict[str, SX]:
        """Helper function to extract the needed current and previous states for the finite difference equations."""
        if i == 0 and bc_expr is None:
            raise ValueError("Provide a value for the boundary condition at i = 0.")
        args = {}
        for arg_name in arg_names[expr_key]:
            if arg_name.endswith("_b"):
                # the backward state is indicated by the suffix _b
                if i == 0:
                    args[arg_name] = bc_expr
                else:
                    args[arg_name] = states[arg_name.replace("_b", "")][i - 1]
            elif arg_name.endswith("_c"):
                # edit the argnames of the wall temperatures to match the cooling section
                args[arg_name] = inputs[arg_name + f"{j}"]
            else:
                # the current state has no suffix
                if arg_name in states.keys():
                    args[arg_name] = states[arg_name][i]
                elif arg_name in inputs.keys():
                    args[arg_name] = inputs[arg_name][j]
                else:
                    raise ValueError(f"Argname {arg_name} neither state nor input.")
        return args

    @abstractmethod
    def setup_physics(self, model: Model) -> Model:
        """Writes all system variables and equations into the Model object."""
        pass

    def create_physical_model(self) -> Model:
        """Creates a continuous do-mpc Model instance with the underlying balance equations."""
        model = Model("continuous", "SX")
        model = self.setup_physics(model=model)
        return model
