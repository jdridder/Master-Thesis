import datetime
import json
import os
import sys
from typing import Dict, List

import numpy as np
import yaml

CURR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
sys.path.insert(0, ROOT_DIR)
from configs.experiment_certain_open_loop import data_cfg, hidden_unit_list, test_cfg_list, training_cfgs
from models import EtOxModel
from postprocessing.performance_metrics import calculate_state_mses, calculate_state_physics_vio
from postprocessing.plot import plot_exec_time_distribution, plot_metric_summary, plot_mse_distribution, plot_mses_vs_time, plot_pc_violation_vs_time, plot_random_trajectories, plot_val_loss
from postprocessing.plotting_helpers import make_colors
from routines.data_structurizer import DataStructurizer
from routines.utils import NumpyEncoder, filter_test_data_for_surrogates, get_directory_for_today, load_json_results_for_all, merge_dict
from simulation.data_generation import generate_data_for_specs
from simulation.open_loop import run_open_loop_for_specs
from training.run import run_training


def run_certain_open_loop_experiment(
    sim_cfg: Dict,
    experiments_directory: str,
    meta_model: EtOxModel,
    data_structurizer: DataStructurizer,
    n_test_trajectories: int = 32,
):
    # create folder for the date
    experiment_name = "001_certain_open_loop_kpis"
    experiment_dir = os.path.join(experiments_directory, experiment_name)
    current_experiment_working_dir = os.path.join(experiment_dir, get_directory_for_today(experiment_dir))
    os.makedirs(current_experiment_working_dir, exist_ok=True)

    # generate training and test data
    for data_purpose, data_specs in data_cfg.items():
        data_target_dir = os.path.join(current_experiment_working_dir, "data", data_purpose)
        if not os.path.exists(data_target_dir):
            print(f"----- Generating data for {data_purpose}. -----")
            os.makedirs(data_target_dir, exist_ok=True)
            generate_data_for_specs(meta_model=meta_model, sim_cfg=sim_cfg, specs=data_specs, data_dir=data_target_dir)

    training_data_dir = os.path.join(current_experiment_working_dir, "data", "train")
    test_data_dir = os.path.join(current_experiment_working_dir, "data", "test")
    run_dir = os.path.join(current_experiment_working_dir, "runs")
    all_run_dirs = [os.path.join(run_dir, f"run_{hidden_units[0]}_units") for hidden_units in hidden_unit_list]
    training_histories = {}
    summaries = []

    for i, (hidden_units, current_run_dir) in enumerate(zip(hidden_unit_list, all_run_dirs)):
        print(f"---- Running experiment 001 for training config {i}.")
        trained_model_dir = os.path.join(current_run_dir, "trained_models")
        for training_cfg in training_cfgs:
            training_cfg["hidden_units"] = hidden_units
            # if not os.path.exists(current_run_dir):
            final_model_dir = os.path.join(trained_model_dir, training_cfg.get("save_dir", ""))
            os.makedirs(final_model_dir, exist_ok=True)
            boundary_cond = meta_model.get_bc_for_all_measurements(n_measurements=data_structurizer.n_measurements)[:, :20]  # only chi states
            run_training(
                model_parameter_dir=final_model_dir,
                training_data_dir=training_data_dir,
                data_structurizer=data_structurizer,
                training_cfg=training_cfg,
                constraint_matrix=meta_model.get_balance_constraint_matrix(num_stacks=data_structurizer.n_measurements, include_temp_as_zero=False),
                boundary_cond=boundary_cond,
            )

            # retrieve the training history from the .json files
            for model_key, model_specs in training_cfg["training_jobs"].items():
                if "states" in model_key:
                    history_path = f"{model_specs.get("save_path")}_hist.json"
                    pc_type = model_specs.get("save_path").split("/")[-2]
                    with open(history_path, "r") as f:
                        loaded_hist = json.load(f)
                    loaded_hist["units"] = hidden_units
                    if not pc_type in training_histories.keys():
                        training_histories[pc_type] = [loaded_hist]
                    else:
                        training_histories[pc_type].append(loaded_hist)

        # # run open loop for NARX, PC NARX, ROM trained using true model parameters for 100 Trajectories
        # test_data = data_structurizer.load_data(data_dir=test_data_dir, num_trajectories=-1)
        # result_directory = os.path.join(current_run_dir, "results")
        # os.makedirs(result_directory, exist_ok=True)
        # for specs in test_cfg_list:
        #     # run simulation batches
        #     specs["n_trajectories"] = test_data.shape[0]
        #     final_model_dir = os.path.join(trained_model_dir, specs.get("torch_module_type", "vanilla"))
        #     run_open_loop_for_specs(
        #         specs=specs,
        #         meta_model=meta_model,
        #         data_structurizer=data_structurizer,
        #         sim_cfg=sim_cfg,
        #         model_parameter_dir=final_model_dir,
        #         save_dir=result_directory,
        #         initialization_data=test_data,
        #     )

        # # calculate mse of these models to test data with true parameters
        # # one mse for the temperature, one mse for the other states
        # # as a function of time averaged over the measurement positions
        # simulation_t_steps = test_cfg_list[0].get("t_steps")
        # warm_up_steps = test_cfg_list[0].get("warm_up_steps")
        # surrogate_results = load_json_results_for_all(result_dir=result_directory, n_trajectories=n_test_trajectories)
        # time = list(surrogate_results.values())[0]["_time"][0].flatten()
        # reduced_test_data = data_structurizer.get_states_from_data(data_structurizer.reduce_measurements(test_data))[..., warm_up_steps : simulation_t_steps + warm_up_steps, :]
        # surrogate_test_data = filter_test_data_for_surrogates(test_data=reduced_test_data, surrogate_results=surrogate_results)
        # state_results = {key: data_structurizer.get_states_from_data(surrogate_results[key].get("_y" if key == "rom" else "_x")) for key in surrogate_results.keys()}
        # # calculate mse as a function of time
        # state_mses_time = calculate_state_mses(sim_cfg=sim_cfg, surrogate_pred_data=state_results, surrogate_test_data=surrogate_test_data, keep="time")
        # state_mses_all = calculate_state_mses(sim_cfg=sim_cfg, surrogate_pred_data=state_results, surrogate_test_data=surrogate_test_data, keep=None)
        # # calculate physics violation as a function of time
        # physics_violations = calculate_state_physics_vio(meta_model=meta_model, sim_cfg=sim_cfg, state_result_list=state_results.values(), norm_measurements=True)
        # pc_violation_dict = {key: physics_violations[i] for i, key in enumerate(surrogate_results.keys())}

        # # save the summary of the evaluation
        # summary_file = os.path.join(current_run_dir, "summary.json")
        # summary = {}
        # for state_key, mse_dict in state_mses_all.items():
        #     for surrogate_key, mse_arr in mse_dict.items():
        #         mse_key = f"mse_{state_key}"
        #         if surrogate_key not in summary.keys():
        #             summary[surrogate_key] = {}
        #         summary[surrogate_key][mse_key] = (np.median(mse_arr), mse_arr.std())

        # for surrogate_key in summary.keys():
        #     summary[surrogate_key]["n_units"] = training_cfg["hidden_units"][0]

        # summaries.append(summary)
        # with open(summary_file, "w") as f:
        #     f.write(json.dumps(summary, indent=4, cls=NumpyEncoder))

        # # ------ Plotting Functions -------
        # plot_dir = os.path.join(current_run_dir, "plots")
        # if not os.path.exists(plot_dir):
        #     print("----- Creating plots. -----")
        #     os.makedirs(plot_dir, exist_ok=True)

        #     # ------ Plot validation loss during training -------
        #     plot_pc_violation_vs_time(
        #         time=time,
        #         pc_violation_dict=pc_violation_dict,
        #         save_dir=plot_dir,
        #         plot_cfg={
        #             "legend_y_pos": 1.15,
        #         },
        #         save_cfg={
        #             "show_fig": False,
        #             "export_name": "pc_violation_vs_time",
        #         },
        #     )

        #     plot_exec_time_distribution(
        #         surrogate_result_dict=surrogate_results,
        #         plot_dir=plot_dir,
        #         plot_cfg=None,
        #         save_cfg={
        #             "show_fig": False,
        #             "export_name": "exe_time_distribution",
        #         },
        #     )

        #     print("---- Plotting mse vs time. ----")
        #     state_keys = ["chi_E", "T"]
        #     for state in state_keys:
        #         plot_mses_vs_time(
        #             mse_data_dict=state_mses_time[state],
        #             time=time,
        #             plot_dir=plot_dir,
        #             plot_cfg={
        #                 "legend_y_pos": 1.25,
        #                 "ylims": (0, 1e-4),
        #             },
        #             save_cfg={
        #                 "show_fig": False,
        #                 "export_name": f"mse_vs_time_{state}",
        #             },
        #         )

        #     print("---- Plotting mse distribution. ----")
        #     plot_mse_distribution(
        #         mse_data_dict_list=list(state_mses_all.values()),
        #         state_keys=state_keys,
        #         plot_dir=plot_dir,
        #         plot_cfg={
        #             "legend_y_pos": 1.15,
        #             "y_upper": [0, 0],
        #             # "y_lower": [-15, -15],
        #             "annotations_x": [0.49, 0.99, 0.88],
        #             "annotations_y": [0.9, 0.9],
        #         },
        #         save_cfg={
        #             "show_fig": False,
        #             "export_name": f"mse_distribution",
        #         },
        #     )

        # trajectory_plot_dir = os.path.join(plot_dir, "surrogate_trajectories")
        # if not os.path.exists(trajectory_plot_dir):
        #     plot_random_trajectories(
        #         sim_cfg=sim_cfg,
        #         n_trajectories=3,
        #         result_dir=result_directory,
        #         save_to_dir=trajectory_plot_dir,
        #         test_data=test_data,
        #         filter_test_trajectories=True,
        #         states=["chi_E", "T"],
        #         plot_cfg={
        #             "t_steps": 480,
        #             "legend_y_pos": 1.35,
        #             "ylabel_size": 20,
        #             "test_data_color": make_colors(4, alpha=1)[1],
        #             "annotations": ["mse"],
        #         },
        #         save_cfg={
        #             "export_name": None,
        #             "save_meta": True,
        #             "show_fig": False,
        #         },
        #     )

    merged_summary = merge_dict(dict_list=summaries)
    with open(os.path.join(current_experiment_working_dir, "summary.json"), "w") as f:
        f.write(json.dumps(merged_summary, indent=4, cls=NumpyEncoder))
    summary_plot_dir = os.path.join(current_experiment_working_dir, "plots")
    os.makedirs(summary_plot_dir, exist_ok=True)

    plot_val_loss(
        training_history_dict=training_histories,
        hidden_units_list=[[64], [16], [4]],
        save_dir=summary_plot_dir,
    )

    # plot_metric_summary(
    #     metric_summary=merged_summary,
    #     metric_keys=["mse_chi_E", "mse_T"],
    #     xaxis_key="n_units",
    #     save_dir=summary_plot_dir,
    #     plot_cfg={
    #         "xlabel": r"$n_\mathrm{units}$ / -",
    #         "ylabels": [r"log$_{10}$ $\langle\mathrm{MSE}\rangle_t$ / -"] * 2,
    #         "titles": ["all other states", "temperature"],
    #     },
    # )

    print("---- experiment finished. -----")


if __name__ == "__main__":
    sim_cfg_name = "etox_control_task.yaml"
    config_directory = os.path.abspath(os.path.join(ROOT_DIR, "configs"))
    with open(os.path.join(config_directory, sim_cfg_name), "r") as f:
        sim_cfg = yaml.safe_load(f)
    model_cfg_directory = os.path.abspath(os.path.join(ROOT_DIR, "models", sim_cfg["model_name"]))
    with open(os.path.join(model_cfg_directory, "EtOxModel.yaml"), "r") as f:
        model_cfg = yaml.safe_load(f)
    experiments_directory = os.path.abspath(os.path.join(ROOT_DIR, "..", "experiments"))

    meta_model = EtOxModel(
        model_cfg=model_cfg,
        state_keys=sim_cfg["states"]["keys"],
        input_keys=sim_cfg["inputs"]["all_keys"],
        N_finite_diff=sim_cfg["simulation"]["N_finite_diff"],
    )
    structurizer = DataStructurizer(
        n_initial_measurements=sim_cfg["simulation"]["N_finite_diff"],
        n_measurements=sim_cfg["narx"]["n_measurements"],
        time_horizon=sim_cfg["narx"]["time_horizon"],
        state_keys=sim_cfg["states"]["keys"],
        input_keys=sim_cfg["inputs"]["all_keys"],
        tvp_keys=sim_cfg["tvps"]["keys"],
    )

    run_certain_open_loop_experiment(
        sim_cfg=sim_cfg,
        experiments_directory=experiments_directory,
        meta_model=meta_model,
        data_structurizer=structurizer,
    )
