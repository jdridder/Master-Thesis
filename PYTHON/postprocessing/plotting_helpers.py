import json
import os
from typing import Dict, List

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from routines.data_structurizer import DataStructurizer
from routines.setup_routines import SurrogateTypes

latex_notation_map = {
    "t_step": r"$t_\mathrm{step}$",
    "time": r"$t$ / $\mathrm{s}$",
    "n_horizon": r"$n_\mathrm{horizon}$",
    "n_robust": r"$n_\mathrm{robust}$",
    "lam_dTdz": r"$\lambda_{\Delta_z T}$",
    "lam_dudt": r"$\lambda_{\Delta_t T}$",
    "lam_Tmax": r"$\lambda_{Tmax}$",
    "lam_conversion": r"$\lambda_{Xmin}$",
    SurrogateTypes.Vanilla.value: {
        "general": "vanilla",
        "nominal": r"$\bm{\tilde{T}}_\mathrm{nominal}$",
        "upper": r"$\bm{\tilde{T}}_\mathrm{ub}$",
        "lower": r"$\bm{\tilde{T}}_\mathrm{lb}$",
    },
    SurrogateTypes.Rigorous.value: {"general": "rigorous"},
    SurrogateTypes.Naive.value: {"general": "naive_pc"},
    SurrogateTypes.Pc.value: {"general": "pc"},
    "b_residual": r"log$_{10} \; ||\bm{b}||_2$" + "/ -",
    "mse": {
        "T": r"mse($\bm{\tilde{T}}$)",
        "chi_E": r"mse($\bm{\tilde{x}}$)",
    },
    "machine_epsilon": r"$\epsilon_\mathrm{mach} (64 \; \mathrm{bit})$",
}
upper_ylims = {"T": 1.25, "c": 1.25}
lower_ylims = {"T": 0.75, "c": 0.25}


def set_mpt_settings(font_size: float = 16, figsize=(10, 6)):
    plt.style.use("science")
    matplotlib.rcParams.update(
        {
            "lines.linewidth": 2,
            "lines.markeredgecolor": "black",
            "lines.markeredgewidth": 0.5,
            "font.size": font_size,
            "axes.labelsize": "large",
            "figure.figsize": figsize,
            "axes.grid": False,
            "lines.markersize": 10,
            "axes.unicode_minus": False,
            "ps.fonttype": 42,  # Avoid type 3 fonts
            "pdf.fonttype": 42,  # Avoid type 3 fonts
            "pgf.texsystem": "xelatex",  # oder xelatex/lualatex
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{bm}",
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        }
    )


def set_line_colors(lines: List, cmap: str = "viridis"):
    cmap = cm.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, len(lines)))
    for i, line in enumerate(lines):
        line.set_color(colors[i])


def add_constraint_line(axes: List, axes_index: int, value: float, kwargs={}) -> List:
    axes[axes_index].axhline(value, color="black", ls="-.", **kwargs)
    return axes


def calculate_aux_from_states(sim_cfg: Dict, state_matrix: np.ndarray) -> np.ndarray:
    "Calculates selectivity and conversion directly from the concentrations."
    structurizer = DataStructurizer(
        n_measurements=4,
        time_horizon=8,
        state_keys=sim_cfg["states"]["keys"],
        input_keys=sim_cfg["inputs"]["all_keys"],
        tvp_keys=sim_cfg["tvps"]["keys"],
    )
    states_outlet = structurizer.get_states_at_measurement_from_data(data=state_matrix, measurement=3)
    # 2 == EO, 0 == E
    c_E_in = 29.099769529825323
    delta_EO = states_outlet[:, 2]
    delta_E = -states_outlet[:, 0] + c_E_in
    conversion = delta_E / c_E_in
    selectivity = delta_EO / delta_E
    aux = np.array([selectivity, conversion]).T
    return aux


def set_labels(axes: List, latex_notation_map: Dict[str, str], sim_cfg: Dict) -> List:
    ylabels = list(sim_cfg["plotting"]["ylabels"].values())
    for i, ax in enumerate(axes):
        label = r"{}".format(ylabels[i])
        ax.set_ylabel(label, rotation=45)
    axes[-1].set_xlabel(latex_notation_map["time"])


def make_colors(n_colors: int, alpha: float = 1, cmap: str = "viridis") -> List:
    cmap = cm.get_cmap(cmap)
    return cmap(np.linspace(0, 1, n_colors), alpha=alpha)


def make_color_cycler(n_colors: int) -> cycler:
    colors = make_colors(n_colors=n_colors)
    color_cycler = cycler(color=colors)
    return color_cycler


def format_legend(ax, plot_cfg: Dict):
    handles, labels = ax.get_legend_handles_labels()
    # Create a unique legend from all ax
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, plot_cfg.get("legend_y_pos", 1.4)), ncol=plot_cfg.get("legend_cols", 3))
    return ax


def make_line_cycler(n_styles: int) -> cycler:
    linestyles = list(["--", "solid", "-."])
    return cycler(linestyle=linestyles[:n_styles])


def make_marker_cyler(n_markers: int) -> cycler:
    markers = ["P", "o", "X"]
    return cycler(marker=markers[:n_markers])


def add_color_cyclers(axes: List, state_cycler, input_cycler, input_index: int):
    for ax in axes[:input_index]:
        ax.set_prop_cycle(state_cycler)
    axes[input_index].set_prop_cycle(input_cycler)


def save_meta_data(directory: str, file_name: str, meta_data: Dict):
    meta_file_name = "_".join(file_name.split("_")[:3])
    meta_data_file_path = os.path.join(directory, f"{meta_file_name}_meta.json")
    if not os.path.exists(meta_data_file_path):
        with open(meta_data_file_path, "w") as f:
            f.write(json.dumps(meta_data, indent=4))


def make_meta_data_annotation(meta_data: Dict, include: List) -> str:
    annotation = ""
    for i, key in enumerate(meta_data.keys()):
        annotation += "\n"
        if key in include:
            lhs = latex_notation_map[key] if key in latex_notation_map.keys() else key
            annotation += f"{lhs} = {meta_data[key]}"
    return annotation
