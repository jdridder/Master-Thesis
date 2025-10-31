import os
import sys

import numpy as np
import scipy
import yaml

sys.path.append("/Users/jandavidridder/Desktop/Masterarbeit/src/PYTHON/MYCODE")
from models.EtOxModel.EtOxModel import EtOxModel


def test_balance_constraint():
    """
    Tests the satisfaction of elemental balance and stoichiometric null space constraints.

    This function verifies that a hardcoded array of physical model output (y)
    adheres to the fundamental conservation laws defined in the EtOxModel configuration.
    It calculates the residual of the element species balance and the stoichiometric
    null space constraint, asserting that both residuals are negligibly small.

    The function relies on:

    Hardcoded, pre-calculated output data ('y') from a verified physical model.

    Loading the necessary constraint matrices (element_species_matrix and
    stoiciometric_null_space) from the EtOxModel class.

    The test performs two main checks, where 'boundary_cond' is subtracted from
    the output 'y' to isolate the physical concentrations, and 'M' is the constraint matrix:
    residual = (y - boundary_cond) @ M.T

    It asserts that:

    The residual for the element species constraint is close to zero (atol=1e-6).

    The residual for the stoichiometric null space constraint is close to zero (atol=1e-6).

    The stoichiometric null space is noted as being equivalent to the element
    species matrix in theory, but often better conditioned for numerical stability.

    Raises:
    AssertionError: If the elemental or stoichiometric constraints are violated
    (residual is not close to zero), providing the Euclidean
    distance (L2-norm) of the residual for context.
    FileNotFoundError: If the configuration file (path_to_cfg) is incorrect.
    KeyError: If the model configuration lacks 'element_species_matrix' or
    'stoiciometric_matrix' keys.
    """
    # hard code physical model output data
    y = np.array(
        [
            [
                0.70328705,
                0.5157943,
                0.41612201,
                0.35142447,
                0.55147867,
                0.36998763,
                0.27308458,
                0.21081688,
                0.24331368,
                0.39570856,
                0.47655409,
                0.52928405,
                0.11513188,
                0.18532761,
                0.22298114,
                0.24691629,
                0.37346521,
                0.44366095,
                0.48131447,
                0.50524962,
                1.03292137,
                1.02413874,
                1.00714362,
                1.01256346,
            ],
            [
                0.70331119,
                0.51567642,
                0.41831416,
                0.3476589,
                0.55150264,
                0.36986308,
                0.27529354,
                0.2067681,
                0.24329429,
                0.3958002,
                0.47480709,
                0.53218323,
                0.11512236,
                0.1853801,
                0.22209083,
                0.24864908,
                0.3734557,
                0.44371344,
                0.48042416,
                0.50698242,
                1.03280383,
                1.02388955,
                1.00408013,
                1.01270865,
            ],
            [
                0.7034056,
                0.5156864,
                0.42062865,
                0.34485372,
                0.55159637,
                0.36986794,
                0.27762122,
                0.20371968,
                0.2432185,
                0.39579016,
                0.47296078,
                0.53433007,
                0.11508515,
                0.18538021,
                0.22115448,
                0.24996575,
                0.37341848,
                0.44371354,
                0.47948781,
                0.50829908,
                1.03257015,
                1.02357492,
                1.00117248,
                1.01271476,
            ],
            [
                0.70361391,
                0.51582616,
                0.4230165,
                0.34298764,
                0.5518032,
                0.37000431,
                0.28001812,
                0.20165565,
                0.24305126,
                0.395677,
                0.47105412,
                0.53574376,
                0.115003,
                0.18532702,
                0.2201921,
                0.25087054,
                0.37333633,
                0.44366035,
                0.47852543,
                0.50920387,
                1.03222184,
                1.02325911,
                0.99845011,
                1.01256726,
            ],
            [
                0.70397694,
                0.51608569,
                0.42539304,
                0.34203659,
                0.55216369,
                0.37026226,
                0.28239949,
                0.20055448,
                0.24275981,
                0.39546874,
                0.46915481,
                0.53644455,
                0.11485982,
                0.18522447,
                0.21923763,
                0.25137105,
                0.37319316,
                0.4435578,
                0.47757096,
                0.50970439,
                1.03176056,
                1.02294438,
                0.99596422,
                1.01227366,
            ],
            [
                0.70453262,
                0.5164489,
                0.42764418,
                0.34197481,
                0.55271547,
                0.37062563,
                0.28465138,
                0.20039006,
                0.24231372,
                0.39517824,
                0.4673542,
                0.53645292,
                0.11464067,
                0.18507905,
                0.21833657,
                0.25147787,
                0.372974,
                0.44341239,
                0.47666991,
                0.50981121,
                1.03118807,
                1.02263461,
                0.99377577,
                1.01178383,
            ],
            [
                0.70531588,
                0.51689212,
                0.4296527,
                0.34276585,
                0.55349321,
                0.37107058,
                0.28665699,
                0.20112286,
                0.2416849,
                0.39482436,
                0.46574623,
                0.53579679,
                0.11433178,
                0.18490039,
                0.21753549,
                0.25120806,
                0.37266512,
                0.44323372,
                0.47586882,
                0.50954139,
                1.0305063,
                1.02233601,
                0.99193636,
                1.01105374,
            ],
            [
                0.70633466,
                0.51738345,
                0.43132564,
                0.34435032,
                0.55450475,
                0.3715649,
                0.28832405,
                0.20268779,
                0.24086697,
                0.39443248,
                0.46440552,
                0.5345214,
                0.11393007,
                0.18470146,
                0.21687102,
                0.2505899,
                0.3722634,
                0.4430348,
                0.47520435,
                0.50892323,
                1.02983516,
                1.02205693,
                0.99047281,
                1.01006012,
            ],
        ]
    )

    # import balance constraint from meta_model
    path_to_cfg = "/Users/jandavidridder/Desktop/Masterarbeit/src/PYTHON/MYCODE/models/EtOxModel/EtOxModel.yaml"
    with open(path_to_cfg, "r") as f:
        cfg = yaml.safe_load(f)

    etox_model = EtOxModel(model_cfg=cfg, state_keys=cfg["states"]["keys"], input_keys=cfg["inputs"]["keys"], N_finite_diff=128)
    boundary_cond = etox_model.get_bc_for_all_measurements(n_measurements=4)

    assert "element_species_matrix" in cfg.keys(), "The model cfg does not contain the element species matrix."
    assert "stoiciometric_matrix" in cfg.keys(), "The model cfg does not contain the stoiciometric matrix."

    element_species_matrix = etox_model.get_element_species_matrix(num_stacks=4, include_temp_as_zero=True)
    stoiciometric_null_space = etox_model.get_balance_constraint_matrix(num_stacks=4, include_temp_as_zero=True)
    # these are in fact equivalent, but the stoichiometric null space is better conditioned

    residual_element_species = (y - boundary_cond) @ element_species_matrix.T
    residual_null_space = (y - boundary_cond) @ stoiciometric_null_space.T

    distance = np.linalg.norm(residual_element_species, ord=2)
    assert np.allclose(
        residual_element_species, np.zeros_like(residual_element_species), atol=1e-6
    ), f"The residual of the element species constraint is not close to zero. Euclidian distance: {distance}."

    distance = np.linalg.norm(residual_null_space, ord=2)
    assert np.allclose(
        residual_null_space, np.zeros_like(residual_null_space), atol=1e-6
    ), f"The residual of the stoichiometric null space constraint is not close to zero. Euclidian distance: {distance}."


if __name__ == "__main__":
    test_balance_constraint()
