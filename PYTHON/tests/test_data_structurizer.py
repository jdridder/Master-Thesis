import os
import sys

import numpy as np
import pytest

CURR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
sys.path.append(ROOT_DIR)
from routines.data_structurizer import DataStructurizer


def test_to_dompc_vector():
    """The true do mpc vector must be stacked (x_0, x_-1, ..., x_-n, u_-1, ..., u_n, tvp_-1, ..., tvp-n)"""
    n_horizon = 4
    n_measurements = 5
    structurizer = DataStructurizer(
        n_measurements=n_measurements,
        n_initial_measurements=128,
        time_horizon=n_horizon,
        state_keys=["x1", "x2"],
        input_keys=["u1", "u2", "u3"],
        tvp_keys=["tvp1", "tvp2"],
    )
    data = np.array(
        [
            [1, 1.1, 1.2, 1.3, 1.4, 2, 2.1, 2.2, 2.3, 2.4, 100, 101, 102, 0.1, 0.2],
            [3, 3.1, 3.2, 3.3, 3.4, 4, 4.1, 4.2, 4.3, 4.4, 200, 201, 202, 0.3, 0.4],
            [5, 5.1, 5.2, 5.3, 5.4, 6, 6.1, 6.2, 6.3, 6.4, 300, 301, 302, 0.5, 0.6],
            [7, 7.1, 7.2, 7.3, 7.4, 8, 8.1, 8.2, 8.3, 8.4, 400, 401, 402, 0.7, 0.8],
            # [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ]
    )
    true_domp_vector = np.array(
        [
            7,
            7.1,
            7.2,
            7.3,
            7.4,
            8,
            8.1,
            8.2,
            8.3,
            8.4,
            5,
            5.1,
            5.2,
            5.3,
            5.4,
            6,
            6.1,
            6.2,
            6.3,
            6.4,
            3,
            3.1,
            3.2,
            3.3,
            3.4,
            4,
            4.1,
            4.2,
            4.3,
            4.4,
            1,
            1.1,
            1.2,
            1.3,
            1.4,
            2,
            2.1,
            2.2,
            2.3,
            2.4,
            300,
            301,
            302,
            200,
            201,
            202,
            100,
            101,
            102,
            0.5,
            0.6,
            0.3,
            0.4,
            0.1,
            0.2,
        ]
    ).reshape((-1, 1))
    dompc_vector = structurizer.to_dompc_vector(reduced_data_matrix=data)
    assert dompc_vector.shape == true_domp_vector.shape, f"The dompc vector has the wrong shape of {dompc_vector.shape} it should have ({true_domp_vector.shape})."
    if not (dompc_vector == true_domp_vector).all():
        raise AssertionError(f"The do mpc vector gets not build up properly. Elementwise difference: {dompc_vector - true_domp_vector}")


def test_make_training_XY():
    """The true input training data must be stacked (x_0, x_-1, ..., x_-n, u_0, u_-1, ..., u_n, tvp_0, tvp_-1, ..., tvp-n)"""
    n_horizon = 3
    n_measurements = 5
    structurizer = DataStructurizer(
        n_measurements=n_measurements,
        time_horizon=n_horizon,
        state_keys=["x1", "x2"],
        input_keys=["u1", "u2", "u3"],
        tvp_keys=["tvp1", "tvp2"],
        n_initial_measurements=128,
    )
    data = np.array(
        [
            [-0, -0.1, -0.2, -0.3, -0.4, -2, -2.1, -2.2, -2.3, -2.4, -200, -201, -202, -0.3, -0.4],
            [0, 0.1, 0.2, 0.3, 0.4, -1, -1.1, -1.2, -1.3, -1.4, -100, -101, -102, -0.1, -0.2],
            [1, 1.1, 1.2, 1.3, 1.4, 2, 2.1, 2.2, 2.3, 2.4, 100, 101, 102, 0.1, 0.2],
            [3, 3.1, 3.2, 3.3, 3.4, 4, 4.1, 4.2, 4.3, 4.4, 200, 201, 202, 0.3, 0.4],
            [5, 5.1, 5.2, 5.3, 5.4, 6, 6.1, 6.2, 6.3, 6.4, 300, 301, 302, 0.5, 0.6],
            [7, 7.1, 7.2, 7.3, 7.4, 8, 8.1, 8.2, 8.3, 8.4, 400, 401, 402, 0.7, 0.8],
        ]
    )
    data_batching = np.expand_dims(data, axis=0)
    true_X = np.array(
        [
            [
                5,
                5.1,
                5.2,
                5.3,
                5.4,
                6,
                6.1,
                6.2,
                6.3,
                6.4,
                3,
                3.1,
                3.2,
                3.3,
                3.4,
                4,
                4.1,
                4.2,
                4.3,
                4.4,
                1,
                1.1,
                1.2,
                1.3,
                1.4,
                2,
                2.1,
                2.2,
                2.3,
                2.4,
                300,
                301,
                302,
                200,
                201,
                202,
                100,
                101,
                102,
                0.5,
                0.6,
                0.3,
                0.4,
                0.1,
                0.2,
            ],
            [
                3,
                3.1,
                3.2,
                3.3,
                3.4,
                4,
                4.1,
                4.2,
                4.3,
                4.4,
                1,
                1.1,
                1.2,
                1.3,
                1.4,
                2,
                2.1,
                2.2,
                2.3,
                2.4,
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                -1,
                -1.1,
                -1.2,
                -1.3,
                -1.4,
                200,
                201,
                202,
                100,
                101,
                102,
                -100,
                -101,
                -102,
                0.3,
                0.4,
                0.1,
                0.2,
                -0.1,
                -0.2,
            ],
            [
                1,
                1.1,
                1.2,
                1.3,
                1.4,
                2,
                2.1,
                2.2,
                2.3,
                2.4,
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                -1,
                -1.1,
                -1.2,
                -1.3,
                -1.4,
                -0,
                -0.1,
                -0.2,
                -0.3,
                -0.4,
                -2,
                -2.1,
                -2.2,
                -2.3,
                -2.4,
                100,
                101,
                102,
                -100,
                -101,
                -102,
                -200,
                -201,
                -202,
                0.1,
                0.2,
                -0.1,
                -0.2,
                -0.3,
                -0.4,
            ],
        ]
    )
    true_Y = np.array([[7, 7.1, 7.2, 7.3, 7.4, 8, 8.1, 8.2, 8.3, 8.4], [5, 5.1, 5.2, 5.3, 5.4, 6, 6.1, 6.2, 6.3, 6.4], [3, 3.1, 3.2, 3.3, 3.4, 4, 4.1, 4.2, 4.3, 4.4]])
    true_X_batching, true_Y_batching = np.expand_dims(true_X, axis=0), np.expand_dims(true_Y, axis=0)

    for d, x, y in zip([data, data_batching], [true_X, true_X_batching], [true_Y, true_Y_batching]):
        X, Y = structurizer.make_training_XY(data_matrix=d)
        assert X.shape == x.shape, f"The input data has the wrong shape of {X.shape} it should be {x.shape}"
        assert Y.shape == y.shape, f"The output data has the wrong shape of {Y.shape} it should be {y.shape}"
        if not (X == x).all():
            raise AssertionError(f"The training input data has the wrong structure. Difference: {X - x}.")
        if not (Y == y).all():
            raise AssertionError(f"The training output data has the wrong structure. Difference: {Y - y}.")


if __name__ == "__main__":
    test_to_dompc_vector()
    test_make_training_XY()
