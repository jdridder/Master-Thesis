import os
import sys

import numpy as np

CURR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
sys.path.append(ROOT_DIR)

from simulation.data_generation import generate_random_ramp_signal


def test_random_ramp_signal():
    t_step = 1
    num_steps = 10
    n_batch = 5
    ramp_signal = generate_random_ramp_signal(
        feature_bounds=[(0, 10), (20, 25)],
        num_steps=num_steps,
        tau=2,
        batch_size=n_batch,
        time_step=t_step,
        seed=42,
    )
    assert ramp_signal.shape == (n_batch, num_steps, 2), f"The ramp signal has the wrong shape {ramp_signal.shape}."

    if __name__ == "__main__":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2)
        ax[0].plot(np.arange(0, num_steps * t_step, t_step), ramp_signal.swapaxes(0, 1)[..., 0], label="signal 1", alpha=0.3)
        ax[1].plot(np.arange(0, num_steps * t_step, t_step), ramp_signal.swapaxes(0, 1)[..., 1], label="signal 2", alpha=0.3)
        plt.show()


if __name__ == "__main__":
    test_random_ramp_signal()
