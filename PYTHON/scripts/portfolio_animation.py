import os
import sys
from typing import List, Tuple

import l4casadi
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from do_mpc.data import Data, load_results
from do_mpc.graphics import Graphics
from matplotlib.animation import FuncAnimation

plt.style.use("science")

colors = ["#a32cff", "#FDFCBB"]
bg_color = "#121212"


def set_mpt_settings():
    matplotlib.rcParams.update(
        {
            "lines.linewidth": 4,
            "text.color": "white",
            "lines.markeredgecolor": "black",
            "lines.markeredgewidth": 0.5,
            "lines.linewidth": 3,
            "font.size": 24,
            "axes.labelsize": "large",
            "figure.figsize": (10, 6),
            "axes.grid": False,
            "lines.markersize": 10,
            "axes.unicode_minus": False,
            "ps.fonttype": 42,  # Avoid type 3 fonts
            "pdf.fonttype": 42,  # Avoid type 3 fonts
        }
    )


def set_fig_preferences(fig, ax):
    fig.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.grid(False)
    ax.tick_params(
        axis="both",  # Betrifft beide Achsen
        which="both",  # Betrifft sowohl Haupt- als auch Neben-Ticks
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.spines["top"].set_color(bg_color)
    ax.spines["right"].set_color(bg_color)
    ax.spines["left"].set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_linewidth(4)  # Linke Achse dicker machen
    ax.spines["bottom"].set_linewidth(4)  # Untere Achse dicker machen
    return fig, ax


def main():
    n_measurements = 4
    horizon_length = 20
    n_scenario = 3
    cmap = matplotlib.colormaps["magma"]
    file_name = "results/closed_loop_rom.pkl"
    imported = load_results(file_name)
    mpc_data = imported["mpc"]
    time = mpc_data["_time"].flatten()
    step = 2
    horizon_time = np.arange(0, step * (horizon_length), step)

    # Change the Data Object from the NARX Simulation to separate the dummy states "x" into the original states
    set_mpt_settings()
    fig, ax = plt.subplots(1, figsize=(6, 6))
    set_fig_preferences(fig, ax)

    state_key = "T"
    var_type = "_aux"
    pos = 0
    states = mpc_data[var_type, state_key, pos]  # has shape (time_steps, 1)

    predictions = np.zeros((time.shape[0], horizon_length, n_scenario))  # has shape (time_steps, horizon_length, n_scenario)
    for t_ind in range(time.shape[0]):
        predictions[t_ind] = mpc_data.prediction((var_type, state_key, pos), t_ind)
    # the inital prediction of the next step should equal the first prediction from the previos time step only for the bounds
    predictions[:, :, 1] = predictions[:, :, 0] + 0.03 * predictions[:, :, 0] + 2 * np.random.rand(*predictions.shape[:2])
    predictions[:, :, 2] = predictions[:, :, 0] - 0.03 * predictions[:, :, 0] + 2 * np.random.rand(*predictions.shape[:2])
    predictions[1:, 0, 1:] = predictions[:-1, 1, 1:]

    # append the first step of the uncertainty prediction to the time step
    states = np.concatenate([states[1:], predictions[:-1, 1, 1:]], axis=-1)  # hast shape (time_step, state, bounds). The bounds are ordered nominal, high, low

    t_now = 10
    t_last = t_now - 10
    nominal_line = ax.plot(time[:t_now], states[:t_now, 0], c="#a32cff", label="system")
    intervall_lines = ax.plot(time[:t_now], states[:t_now, 1:], c="#a32cff", ls="dotted", linewidth=3, label="confidence", alpha=0.5)
    nominal_pred = ax.plot(time[t_now - 1] + horizon_time, predictions[t_now, :, 0], ls="-", color="#FDFCBB", label="prediction")
    intervall_preds = ax.plot(time[t_now - 1] + horizon_time, predictions[t_now, :, 1:], ls="dotted", color="#FDFCBB", linewidth=3, alpha=0.75)

    t_ind_end = 52
    t_end = time[t_ind_end]
    ax.set_ylim(550, 750)
    ax.set_xlim(time[0], t_end + horizon_time[-1])
    ax.set_ylabel("state", color="white")
    ax.set_xlabel("$t$", color="white")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left")

    state_lines = nominal_line + intervall_lines
    prediction_lines = nominal_pred + intervall_preds

    def update(t_now):
        t_last = t_now - 10
        past_time = time[:t_now]
        future_time = time[t_now - 1] + horizon_time
        for i, state_line in enumerate(state_lines):
            state_line.set_xdata(past_time)
            state_line.set_ydata(states[:t_now, i])
        for i, pred_line in enumerate(prediction_lines):
            pred_line.set_xdata(future_time)
            pred_line.set_ydata(predictions[t_now, :, i])

        ax.set_xlim(time[0], t_end + horizon_time[-1])

    animation = FuncAnimation(fig, update, frames=range(10, t_ind_end), repeat=True)
    animation.save("uncertain_mpc.gif", writer="imagemagick", fps=8, dpi=100)
    plt.show()


if __name__ == "__main__":
    main()
