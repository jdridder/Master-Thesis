import inspect
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from do_mpc.controller import MPC
from do_mpc.data import Data, MPCData
from do_mpc.graphics import Graphics
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class Visualizer:
    def __init__(
        self,
        cfg: Dict,
        cmap: str = "viridis",
    ):
        self.cfg = cfg
        self.n_measurements = 4
        self.cmap = cm.get_cmap(cmap)
        self.y_label_pad = 40
        self.markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]
        self.y_labels = cfg["plotting"]["ylabels"]
        self.plot_keys = self.cfg["states"]["keys"] + ["T_c"] + ["u"] + ["S"]
        self.state_keys = cfg["states"]["keys"]
        self.input_keys = cfg["inputs"]["keys"]
        self.tvp_keys = cfg["tvps"]["keys"]
        self.aux_keys = cfg["aux"]["keys"]

    def create_z_profile_graph(self, axes: List, x: np.ndarray, block_size: int, ls: str) -> List[Tuple]:
        """Creates a list of graphs for each state in the reactor profile."""
        graphs = []
        for i, state_key in enumerate(self.cfg["states"]["keys"]):
            color = self.cmap(i / (len(self.state_keys) - 1))
            (graph,) = axes[i].plot([], [], c=color, ls=ls)  # leere Scatter-Punkte
            graphs.append(graph)
            y_max = x[:, i * block_size : (i + 1) * block_size].max()
            y_min = x[:, i * block_size : (i + 1) * block_size].min()
            padding = np.abs(y_max - y_min) * 0.05
            axes[i].set_ylim(
                top=y_max + padding,
                bottom=y_min - padding,
            )
            if self.cfg is None:
                axes[i].set_ylabel(state_key)
            else:
                axes[i].set_ylabel(
                    self.cfg["plotting"]["y_coordinate_labels"][state_key],
                    rotation=0,
                    labelpad=self.y_label_pad,
                )
        return graphs

    def animate_reactor_profile(self, simulator_data: Data, title: str = "", ls: str = "-"):
        """Creates an animation of all states of the reactor profile over time.

        Args:
            simulator_data (Data): do_mpc Data object containing the simulation results.
            title (str, optional): Title of the plot. Defaults to "".
            ls (str, optional): Linestyle. Defaults to "-".

        Raises:
            ValueError: When data is empty or not available for animation.

        Returns:
            plt.FuncAnimation: plt.FuncAnimation object for the animation.
        """
        t = simulator_data["_time"]
        x = simulator_data["_x"]
        if t.shape[0] == 0:
            raise ValueError("No datapoints available for animating.")
        assert x.shape[1] % len(self.state_keys) == 0, f"The number of total states ({x.shape[1]}) cannot correspond to the number of unique states ({len(self.state_keys)})."
        block_size = x.shape[1] // len(self.state_keys)
        fig, axes = plt.subplots(len(self.state_keys), sharex=True, figsize=(16, 9))
        axes[-1].set_xlabel(r"$\frac{z}{L}$ / m")
        axes[-1].set_xlim(right=1.1)
        z_i = np.linspace(0, 1, block_size)
        graphs = self.create_z_profile_graph(axes=axes, x=x, block_size=block_size, ls=ls)

        def update(frame: int):
            for i, graph in enumerate(graphs):
                y = x[frame, i * block_size : (i + 1) * block_size]
                graph.set_data(z_i, y)
            return graphs

        animation = FuncAnimation(fig, update, frames=len(t), interval=30, blit=True)
        return animation

    def make_axes_for_all_vars(self, n_plots: int, cbar: bool = True, figsize=(10, 6)) -> Tuple[Figure, List[Axes]]:
        """Creates a grid of axes for all unique states, inputs, tvps and auxillary vars."""
        n_grids = n_plots + 2 if cbar else n_plots
        height_ratios = [1.5, 1] + [10] * n_plots if cbar else [10] * n_plots
        plot_idx = 2 if cbar else 0
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(n_grids, 1, height_ratios=height_ratios)
        # Colorbar on the top
        if cbar:
            cbar_ax = fig.add_subplot(gs[0, 0])
            norm = matplotlib.colors.Normalize(0, 1)
            mappable = cm.ScalarMappable(norm=norm, cmap=self.cmap)
            cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
            cbar.ax.set_ylabel(r"$\frac{z}{L}$ / -", rotation=0, labelpad=self.y_label_pad)
        axes = []
        for i in range(n_plots):
            ax_i = fig.add_subplot(gs[i + plot_idx, 0], sharex=axes[0] if axes else None)
            if i < n_plots - 1:
                ax_i.label_outer()
            axes.append(ax_i)
        return fig, axes

    def sync_ylims(self, axes: List, plot_keys: List):
        grouped_axes = defaultdict(list)
        for axis, label in zip(axes, plot_keys):
            key = label[0]
            grouped_axes[key].append(axis)
        for _, axes_group in grouped_axes.items():
            ymins = [a.get_ylim()[0] for a in axes_group]
            ymaxs = [a.get_ylim()[1] for a in axes_group]
            shared_ylim = (min(ymins), max(ymaxs))
            for a in axes_group:
                a.set_ylim(shared_ylim)
        return axes

    def plot_horizon_high_fideltiy(self, N_z: int, data: Data, title: str = "") -> Tuple[plt.figure, List[Graphics]]:
        graphic = Graphics(data)
        time = data["_time"]
        fig, ax = self.make_axes_for_all_vars(len(self.plot_keys))

        # Plotting States
        for j, var_key in enumerate(self.state_keys):
            graphic.add_line(var_type="_x", var_name=var_key, axis=ax[j])
            ax[j].set_ylabel(self.y_labels[var_key], rotation=0, labelpad=self.y_label_pad)

        for k, var_key in enumerate(self.input_keys):
            # Plotting Inputs
            graphic.add_line(var_type="_u", var_name=var_key, axis=ax[j + 1], color="black", label=var_key)
        ax[j + 1].set_ylabel(self.y_labels["T_c"], rotation=0, labelpad=self.y_label_pad)

        graphic.add_line("_tvp", "u", ax[-2], color="black", label="u")
        ax[j + 2].set_ylabel("$u$ / m s$^{-1}$", rotation=0, labelpad=self.y_label_pad)

        colors = self.cmap(np.arange(N_z) / (N_z - 1))

        if "S" in data.model["_aux"].keys():
            graphic.add_line(var_type="_aux", var_name="S", axis=ax[-1], color=colors[0], label=r"$S_{E,EO}$ / -", ls="dotted")
        if "X" in data.model["_aux"].keys():
            graphic.add_line(var_type="_aux", var_name="X", axis=ax[-1], color=colors[-10], label=r"$X_E$ / -")
        ax[-1].hlines(0.6, time[0], time[-1], color="gray", ls="--", label=r"$X_\text{min}$")

        for state_key in self.state_keys:
            lines = graphic.result_lines["_x", state_key]
            [line.set_color(colors[i]) for i, line in enumerate(lines)]
        lines = graphic.result_lines["_u", "T_c"]
        colors = self.cmap(np.arange(len(lines)) / (len(lines) - 1))
        [line.set_color(colors[i]) for i, line in enumerate(lines)]

        ax = self.sync_ylims(ax, self.plot_keys)

        ax[-1].set_ylabel("$X, S$ / -", rotation=0, labelpad=self.y_label_pad)
        ax[-1].legend()
        ax[-1].set_ylim(top=1.5, bottom=-0.5)
        ax[-1].set_xlabel("t / s")

        fig.align_ylabels()
        fig.subplots_adjust(left=0.2)

        def update(t_ind: int) -> None:
            if isinstance(data, MPCData):
                graphic.plot_results(t_ind=t_ind)
                graphic.reset_axes()
            else:
                graphic.plot_results(t_ind=t_ind)
                graphic.plot_predictions(t_ind=t_ind)
                # graphics[0].reset_axes()

        def animate(frames: int = 20, repeat: bool = False):
            anim = FuncAnimation(fig, update, frames=frames, repeat=repeat)
            return anim

        return {
            "fig": fig,
            "axes": ax,
            "graphic": graphic,
            "update_f": update,
            "animate_f": animate,
        }

    def plot_error_time_scenario(self, mpc_data: Data, model: str = ""):
        """Plots the fulfillment of the component balance by displaying the sum of all mole fractions that have to equal one.
        This regards the predictions of the mpc model for the different scenarions in multi-stage mpc.
        The sum of mole fractions are averaged along position. The prediction horizon is plotted on the y-axis for each actual time step.
        """
        """arr = data.prediction(('_x', 'x_1'))
        arr.shape
        >> (n_size, n_horizon, n_scenario)"""
        aux_keys = []
        for aux_key in mpc_data.model["_aux"].keys():
            if aux_key.startswith("sum_x"):
                aux_keys.append(aux_key)

        predictions = []
        for i in range(mpc_data["_time"].shape[0]):
            # average over all positions
            prediction = np.zeros_like(mpc_data.prediction(("_aux", aux_keys[0]), t_ind=i))
            for k, aux_key in enumerate(aux_keys):
                prediction += mpc_data.prediction(("_aux", aux_key), t_ind=i)
            predictions.append(prediction / len(aux_keys))
        # calculate the error to the conservation law
        predictions = np.abs(np.array(predictions) - 1)

        n_horizon, n_sceario = predictions[0].shape[1:]

        fig, ax = plt.subplots(
            n_sceario,
            figsize=(16, 9),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        norm = plt.Normalize(vmin=1, vmax=n_horizon)
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
        sm.set_array(np.arange(n_horizon))

        fig.suptitle(f"average mole fraction at prediction step – {model}")
        for k in range(n_sceario):
            for i in range(mpc_data["_time"].shape[0]):
                for j in range(n_horizon):
                    ax[k].scatter(
                        i,
                        predictions[i][0].T[k][j],
                        color=self.cmap(norm(j + 1)),
                        marker=self.markers[j % len(self.markers)],
                    )
            ax[k].set_ylabel(rf"$e(s_{k}$) / -")

        ax[-1].set_yscale("log")
        ax[-1].set_xlabel(r"$t_{step}$ / -")
        ax[-1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        cbar = fig.colorbar(
            sm,
            ax=ax.ravel().tolist(),
            orientation="vertical",
            shrink=0.95,
            pad=0.02,
        )
        cbar.set_label(r"$\hat{t}_{\text{step}}$ / -")
        cbar.set_ticks(np.arange(1, n_horizon))
        cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        fig.set_constrained_layout_pads(w_pad=1, h_pad=0.15)

        return fig
