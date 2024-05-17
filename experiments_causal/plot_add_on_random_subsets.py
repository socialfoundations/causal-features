"""Python script to plot model performance across each indidividual candidate model."""
#%%
from pathlib import Path
import pandas as pd
import json
import os
import numpy as np
import itertools
import torch
import torch.utils.data as data_utils
from tqdm import tqdm

from tableshift import get_dataset

import seaborn as sns
from paretoset import paretoset
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import FormatStrFormatter
import matplotlib.markers as mmark

from experiments_causal.plot_config_colors import *
from experiments_causal.plot_experiment import get_results

from statsmodels.stats.proportion import proportion_confint
from experiments_causal.plot_config_colors import *
from experiments_causal.plot_config_tasks import dic_title

# Set plot configurations
sns.set_context("paper")
sns.set_style("white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 1200
list_mak = [
    mmark.MarkerStyle("s"),
    mmark.MarkerStyle("D"),
    mmark.MarkerStyle("o"),
    mmark.MarkerStyle("X"),
    mmark.MarkerStyle("*"),
]
list_lab = ["All", "Arguably causal", "Causal", "Constant","Random"]
list_color = [color_all, color_arguablycausal, color_causal, color_constant, plt.cm.Paired(1)]

list_mak_results = list_mak.copy()
list_mak_results.append("_")
list_lab_results = list_lab.copy()
list_lab_results.append("Diagonal")
list_color_results = list_color.copy()
list_color_results.append("black")

class MarkerHandler(HandlerBase):
    def create_artists(
        self, legend, tup, xdescent, ydescent, width, height, fontsize, trans
    ):
        return [
            plt.Line2D(
                [width / 2],
                [height / 2.0],
                ls="",
                marker=tup[1],
                markersize=markersize,
                color=tup[0],
                transform=trans,
            )
        ]
    
def get_dic_experiments_value(name: str) -> list:
    """Return list of experiment names for a task.

    Parameters
    ----------
    name : str
        The name of the task.

    Returns
    -------
    list
        List of experiment names (all features, causal features, arguably causal features).

    """
    return [f"{name}_random_test_{index}" for index in range(500)]


dic_experiments = {
    "acsemployment": get_dic_experiments_value("acsemployment"),
    "acsfoodstamps": get_dic_experiments_value("acsfoodstamps"),
    "acsincome": get_dic_experiments_value("acsincome"),
    "acspubcov": get_dic_experiments_value("acspubcov"),
    "acsunemployment": get_dic_experiments_value("acsunemployment"),
    "anes": get_dic_experiments_value("anes"),
    "assistments": get_dic_experiments_value("assistments"),
    "brfss_blood_pressure": get_dic_experiments_value("brfss_blood_pressure"),
    "brfss_diabetes": get_dic_experiments_value("brfss_diabetes"),
    "college_scorecard": get_dic_experiments_value("college_scorecard"),
    "diabetes_readmission": get_dic_experiments_value("diabetes_readmission"),
    "meps": get_dic_experiments_value("meps"),
    "mimic_extract_los_3": get_dic_experiments_value("mimic_extract_los_3"),
    "mimic_extract_mort_hosp": get_dic_experiments_value("mimic_extract_mort_hosp"),
    "nhanes_lead": get_dic_experiments_value("nhanes_lead"),
    "physionet": get_dic_experiments_value("physionet"),
    "sipp": get_dic_experiments_value("sipp"),
}

def get_results_random_subsets(experiment_name: str) -> pd.DataFrame:

    """Load json files of experiments from results folder, concat them into a dataframe and save it.

    Parameters
    ----------
    experiment_name : str
        The name of the task.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the results of the experiment.

    """
    experiments = dic_experiments[experiment_name]

    # Load all json files of experiments
    eval_all = pd.DataFrame()
    feature_selection = []
    for experiment in experiments:
        file_info = []
        try:
            RESULTS_DIR = Path(__file__).parents[0] / "add_on_results" / "random_subset" / experiment
            for filename in os.listdir(RESULTS_DIR):
                if filename == ".DS_Store":
                    pass
                else:
                    file_info.append(filename)

            for run in file_info:
                with open(str(RESULTS_DIR / run), "rb") as file:
                    # print(str(RESULTS_DIR / run))
                    try:
                        eval_json = json.load(file)
                        eval_pd = pd.DataFrame(
                            [
                                {
                                    "id_test": eval_json["id_test"],
                                    "id_test_lb": eval_json["id_test" + "_conf"][0],
                                    "id_test_ub": eval_json["id_test" + "_conf"][1],
                                    "ood_test": eval_json["ood_test"],
                                    "ood_test_lb": eval_json["ood_test" + "_conf"][0],
                                    "ood_test_ub": eval_json["ood_test" + "_conf"][1],
                                    "validation": eval_json["validation"],
                                    "features": f"random_{experiment.split('_')[-1]}",
                                    "model": run.split("_", 2)[0] + "_" + run.split("_", 2)[1] if run.split("_")[0] in ["ib", "and", "causirl"] else run.split("_")[0],
                                    "number": len(eval_json["features"]),
                                }
                            ]
                        )
                        eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)
                    except:
                        print(str(RESULTS_DIR / run))
        except:
            print(experiment)

    # Select model with highest in-domain validation accuracy
    list_model_data = []
    for set in eval_all["features"].unique():
        eval_feature = eval_all[eval_all["features"] == set]
        for model in eval_feature["model"].unique():
            model_data = eval_feature[eval_feature["model"] == model]
            model_data = model_data[
                model_data["validation"] == model_data["validation"].max()
            ]
            model_data.drop_duplicates(inplace=True)
            list_model_data.append(model_data)
    eval_all = pd.concat(list_model_data)
    return eval_all

# def plot_results(experiment_name: str):
    
experiments = [
    "acsfoodstamps",
    "acsincome",
    "acspubcov",
    "acsunemployment",
    "anes",
    "assistments",
    "brfss_blood_pressure",
    "brfss_diabetes",
    "college_scorecard",
    "diabetes_readmission",
    # "mimic_extract_mort_hosp",
    # "mimic_extract_los_3",
    # "nhanes_lead",
    "physionet",
    "meps",
    "sipp",
]


for index, experiment_name in enumerate(experiments):
    sns.set_style("white")
    if index % 4 == 0:
        fig = plt.figure(figsize=[5.5, 7])
        (subfig1, subfig2, subfig3, subfig4) = fig.subfigures(4, 1, hspace=0.5)  # create 4x1 subfigures

        subfigs = (subfig1, subfig2, subfig3, subfig4)

        ax1 = subfig1.subplots(
            1, 1, gridspec_kw={"top": 0.85}
        )  # create 1x4 subplots on subfig1
        ax2 = subfig2.subplots(
            1, 1, gridspec_kw={"top": 0.85}
        )  # create 1x4 subplots on subfig2
        ax3 = subfig3.subplots(
            1, 1, gridspec_kw={"top": 0.85}
        )  # create 1x4 subplots on subfig2
        ax4 = subfig4.subplots(
            1, 1, gridspec_kw={"top": 0.85}
        )  # create 1x4 subplots on subfig2
        axes = (ax1, ax2, ax3, ax4)

        subfig = subfigs[index % 4]
        subfig.subplots_adjust(wspace=0.4, bottom=0.3)
        ax = axes[index % 4]
        subfig.suptitle(dic_title[experiment_name], fontsize=9)  # set suptitle for subfig1
        ax.set_title(dic_title[experiment_name], fontsize=9)  # set suptitle for subfig1
        #############################################################################
        # plot errorbars for random features
        #############################################################################
        eval_all_random = get_results_random_subsets(experiment_name)

        for set in eval_all_random["features"].unique():
            eval_plot = eval_all_random[eval_all_random["features"] == set]
            eval_plot.sort_values("id_test", inplace=True)
            # Calculate the pareto set
            points = eval_plot[["id_test", "ood_test"]]
            mask = paretoset(points, sense=["max", "max"])
            markers = eval_plot[mask]
            errors = ax.errorbar(
                x=markers["id_test"],
                y=markers["ood_test"],
                xerr=markers["id_test_ub"] - markers["id_test"],
                yerr=markers["ood_test_ub"] - markers["ood_test"],
                fmt="*",
                color=plt.cm.Paired(1),
                ecolor="#78add2",
                markersize=markersize,
                capsize=capsize,
            )

        eval_all = get_results(experiment_name)

        eval_constant = eval_all[eval_all["features"] == "constant"]
        dic_shift_acc = {}

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

        ax.set_xlabel(f"In-domain accuracy")
        ax.set_ylabel(f"Out-of-domain\naccuracy")

        #############################################################################
        # plot errorbars for constant
        #############################################################################
        errors = ax.errorbar(
            x=eval_constant["id_test"],
            y=eval_constant["ood_test"],
            xerr=eval_constant["id_test_ub"] - eval_constant["id_test"],
            yerr=eval_constant["ood_test_ub"] - eval_constant["ood_test"],
            fmt="X",
            color=color_constant,
            ecolor=color_error,
            markersize=markersize,
            capsize=capsize,
            label="constant",
        )
        # get pareto set for shift vs accuracy
        shift_acc = eval_constant
        shift_acc["type"] = "constant"
        shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
        dic_shift_acc["constant"] = shift_acc

        #############################################################################
        # plot errorbars for causal features
        #############################################################################
        eval_plot = eval_all[eval_all["features"] == "causal"]
        eval_plot.sort_values("id_test", inplace=True)
        # Calculate the pareto set
        points = eval_plot[["id_test", "ood_test"]]
        mask = paretoset(points, sense=["max", "max"])
        markers = eval_plot[mask]
        markers = markers[markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]
        errors = ax.errorbar(
            x=markers["id_test"],
            y=markers["ood_test"],
            xerr=markers["id_test_ub"] - markers["id_test"],
            yerr=markers["ood_test_ub"] - markers["ood_test"],
            fmt="o",
            color=color_causal,
            ecolor=color_error,
            markersize=markersize,
            capsize=capsize,
            label="causal",
        )
        # get pareto set for shift vs accuracy
        shift_acc = eval_plot[
            eval_plot["ood_test"] == eval_plot["ood_test"].max()
        ].drop_duplicates()
        shift_acc["type"] = "causal"
        shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
        dic_shift_acc["causal"] = shift_acc
        #############################################################################
        # plot errorbars for arguablycausal features
        #############################################################################
        if (eval_all["features"] == "arguablycausal").any():
            eval_plot = eval_all[eval_all["features"] == "arguablycausal"]
            eval_plot.sort_values("id_test", inplace=True)
            # Calculate the pareto set
            points = eval_plot[["id_test", "ood_test"]]
            mask = paretoset(points, sense=["max", "max"])
            markers = eval_plot[mask]
            markers = markers[
                markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
            ]
            errors = ax.errorbar(
                x=markers["id_test"],
                y=markers["ood_test"],
                xerr=markers["id_test_ub"] - markers["id_test"],
                yerr=markers["ood_test_ub"] - markers["ood_test"],
                fmt="D",
                color=color_arguablycausal,
                ecolor=color_error,
                markersize=markersize,
                capsize=capsize,
                label="arguably\ncausal",
            )
            # get pareto set for shift vs accuracy
            shift_acc = eval_plot[
                eval_plot["ood_test"] == eval_plot["ood_test"].max()
            ].drop_duplicates()
            shift_acc["type"] = "arguablycausal"
            shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
            dic_shift_acc["arguablycausal"] = shift_acc

        #############################################################################
        # plot errorbars for all features
        #############################################################################
        eval_plot = eval_all[eval_all["features"] == "all"]
        eval_plot.sort_values("id_test", inplace=True)
        # Calculate the pareto set
        points = eval_plot[["id_test", "ood_test"]]
        mask = paretoset(points, sense=["max", "max"])
        markers = eval_plot[mask]
        markers = markers[markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]
        errors = ax.errorbar(
            x=markers["id_test"],
            y=markers["ood_test"],
            xerr=markers["id_test_ub"] - markers["id_test"],
            yerr=markers["ood_test_ub"] - markers["ood_test"],
            fmt="s",
            color=color_all,
            ecolor=color_error,
            markersize=markersize,
            capsize=capsize,
            label="all",
        )
        # get pareto set for shift vs accuracy
        shift_acc = eval_plot[
            eval_plot["ood_test"] == eval_plot["ood_test"].max()
        ].drop_duplicates()
        shift_acc["type"] = "all"
        shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
        dic_shift_acc["all"] = shift_acc

        #############################################################################
        # plot pareto dominated area for constant
        #############################################################################
        xmin, xmax = ax.set_xlim()
        ymin, ymax = ax.set_ylim()
        ax.plot(
            [xmin, eval_constant["id_test"].values[0]],
            [eval_constant["ood_test"].values[0], eval_constant["ood_test"].values[0]],
            color=color_constant,
            linestyle=(0, (1, 1)),
            linewidth=linewidth_bound,
        )
        ax.plot(
            [eval_constant["id_test"].values[0], eval_constant["id_test"].values[0]],
            [ymin, eval_constant["ood_test"].values[0]],
            color=color_constant,
            linestyle=(0, (1, 1)),
            linewidth=linewidth_bound,
        )
        ax.fill_between(
            [xmin, eval_constant["id_test"].values[0]],
            [ymin, ymin],
            [eval_constant["ood_test"].values[0], eval_constant["ood_test"].values[0]],
            color=color_constant,
            alpha=0.05,
        )

        #############################################################################
        # plot pareto dominated area for causal features
        #############################################################################
        eval_plot = eval_all[eval_all["features"] == "causal"]
        eval_plot.sort_values("id_test", inplace=True)
        # Calculate the pareto set
        points = eval_plot[["id_test", "ood_test"]]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[points["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]
        markers = eval_plot[mask]
        markers = markers[markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]
        # get extra points for the plot
        new_row = pd.DataFrame(
            {
                "id_test": [xmin, max(points["id_test"])],
                "ood_test": [max(points["ood_test"]), ymin],
            },
        )
        points = pd.concat([points, new_row], ignore_index=True)
        points.sort_values("id_test", inplace=True)
        ax.plot(
            points["id_test"],
            points["ood_test"],
            color=color_causal,
            linestyle=(0, (1, 1)),
            linewidth=linewidth_bound,
        )
        new_row = pd.DataFrame(
            {"id_test": [xmin], "ood_test": [ymin]},
        )
        points = pd.concat([points, new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        ax.fill(
            points[hull.vertices, 0], points[hull.vertices, 1], color=color_causal, alpha=0.05
        )

        #############################################################################
        # plot pareto dominated area for arguablycausal features
        #############################################################################
        if (eval_all["features"] == "arguablycausal").any():
            eval_plot = eval_all[eval_all["features"] == "arguablycausal"]
            eval_plot.sort_values("id_test", inplace=True)
            # Calculate the pareto set
            points = eval_plot[["id_test", "ood_test"]]
            mask = paretoset(points, sense=["max", "max"])
            points = points[mask]
            points = points[points["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]
            # get extra points for the plot
            new_row = pd.DataFrame(
                {
                    "id_test": [xmin, max(points["id_test"])],
                    "ood_test": [max(points["ood_test"]), ymin],
                },
            )
            points = pd.concat([points, new_row], ignore_index=True)
            points.sort_values("id_test", inplace=True)
            ax.plot(
                points["id_test"],
                points["ood_test"],
                color=color_arguablycausal,
                linestyle=(0, (1, 1)),
                linewidth=linewidth_bound,
            )
            new_row = pd.DataFrame(
                {"id_test": [xmin], "ood_test": [ymin]},
            )
            points = pd.concat([points, new_row], ignore_index=True)
            points = points.to_numpy()
            hull = ConvexHull(points)
            ax.fill(
                points[hull.vertices, 0],
                points[hull.vertices, 1],
                color=color_arguablycausal,
                alpha=0.05,
            )

        #############################################################################
        # plot pareto dominated area for all features
        #############################################################################
        eval_plot = eval_all[eval_all["features"] == "all"]
        eval_plot.sort_values("id_test", inplace=True)
        # Calculate the pareto set
        points = eval_plot[["id_test", "ood_test"]]
        mask = paretoset(points, sense=["max", "max"])
        points = points[mask]
        points = points[points["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]
        # get extra points for the plot
        new_row = pd.DataFrame(
            {
                "id_test": [xmin, max(points["id_test"])],
                "ood_test": [max(points["ood_test"]), ymin],
            },
        )
        points = pd.concat([points, new_row], ignore_index=True)
        points.sort_values("id_test", inplace=True)
        ax.plot(
            points["id_test"],
            points["ood_test"],
            color=color_all,
            linestyle=(0, (1, 1)),
            linewidth=linewidth_bound,
        )
        new_row = pd.DataFrame(
            {"id_test": [xmin], "ood_test": [ymin]},
        )
        points = pd.concat([points, new_row], ignore_index=True)
        points = points.to_numpy()
        hull = ConvexHull(points)
        ax.fill(
            points[hull.vertices, 0], points[hull.vertices, 1], color=color_all, alpha=0.05
        )

        #############################################################################
        # Add legend & diagonal, save plot
        #############################################################################
        # Plot the diagonal line
        start_lim = max(xmin, ymin)
        end_lim = min(xmax, ymax)
        ax.plot([start_lim, end_lim], [start_lim, end_lim], color=color_error)


        fig.legend(
            list(zip(list_color_results, list_mak_results)),
            list_lab_results,
            handler_map={tuple: MarkerHandler()},
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            fancybox=True,
            ncol=6,
        )

        fig.savefig(
            str(Path(__file__).parents[0] / f"plots_add_on/plot_random_subsets_{experiment_name}.pdf"),
            bbox_inches="tight",
        )

        if index % 4 == 3:
            fig.legend(
            list(zip(list_color_results, list_mak_results)),
            list_lab_results,
            handler_map={tuple: MarkerHandler()},
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            fancybox=True,
            ncol=6,
            )

            fig.savefig(
                str(Path(__file__).parents[0] / f"plots_add_on/plot_random_subsets_{int(index/4)}.pdf"),
                bbox_inches="tight",
            )

            fig.show()