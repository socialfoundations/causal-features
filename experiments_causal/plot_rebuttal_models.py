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

from tableshift import get_dataset
from otdd.pytorch.distance import DatasetDistance

import seaborn as sns
from paretoset import paretoset
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import FormatStrFormatter
import matplotlib.markers as mmark

from experiments_causal.plot_config_colors import *
from experiments_causal.plot_experiment import get_results
from experiments_causal.plot_experiment_anticausal import get_results as get_results_anticausal
from experiments_causal.plot_experiment_anticausal import dic_experiments as dic_experiments_anticausal
from experiments_causal.plot_config_tasks import dic_title

import warnings
warnings.filterwarnings("ignore")

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
]
list_lab = ["All", "Arguably causal", "Causal", "Constant"]
list_color = [color_all, color_arguablycausal, color_causal, color_constant]
list_mak.append("_")
list_lab.append("Same performance")
list_color.append("black")

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
    


# Define list of experiments to plot
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
    "mimic_extract_mort_hosp",
    "mimic_extract_los_3",
    "nhanes_lead",
    "physionet",
    "meps",
    "sipp",
]

encode_tableshift = {'tableshift:adv. label dro': 'aldro',
 'tableshift:catboost':'catboost',
 'tableshift:dro':'dro',
 'tableshift:ft-transformer':'ft',
 'tableshift:label group dro':'label',
 'tableshift:lightgbm':'lightgbm',
 'tableshift:mlp':'mlp',
 'tableshift:node':'node',
 'tableshift:resnet':'resnet',
 'tableshift:saint':'saint',
 'tableshift:tabtransformer':'tabtransformer',
 'tableshift:xgboost':'xgb',
 'tableshift:coral':'deepcoral',
 'tableshift:dann':'dann',
 'tableshift:group dro':'group',
 'tableshift:irm':'irm',
 'tableshift:mmd':'mmd',
 'tableshift:mixup':'mixup',
 'tableshift:vrex':'vrex',}

encode_model = {
    'aldro': "Adv. DRO",
    'and_mask':"AND Mask",
    'causirl_coral':"CausIRL Coral",
    'causirl_mmd':"CausIRL MMD",
    'dann':"DANN",
    'deepcoral':"DeepCORAL",
    'dro':"DRO",
    'ft':"FT-Transformer",
    'group':"Group DRO",
    'histgbm':"HistGBM",
    'ib_irm':"IB-IRM",
    'irm':"IRM",
    'label':"Label Group DRO",
    'lightgbm':"LightGBM",
    'mixup':"MixUp",
    'mlp':"MLP",
    'mmd':"MMD",
    'node':"NODE",
    'resnet':"ResNet",
    'saint':"SAINT",
    'tabtransformer':"TabTransformer",
    'vrex':"ReX",
    'xgb':"XGB"
}
eval_experiments = pd.DataFrame()
for index, experiment_name in enumerate(experiments):
    eval_all = get_results(experiment_name)
    eval_all["task"] = dic_title[experiment_name]

    eval_plot = pd.DataFrame()
    for set in eval_all["features"].unique():
        eval_feature = eval_all[eval_all["features"] == set]
        for model in eval_feature["model"]:
            eval_model = eval_feature[eval_feature["model"] == model]
            eval_model = eval_model[
                eval_model["ood_test"] == eval_model["ood_test"].max()
            ]
            eval_model.drop_duplicates(inplace=True)
            if model.startswith("tableshift"):
                eval_model["model"] = encode_tableshift[model]
            eval_plot = pd.concat([eval_plot, eval_model])
    eval_experiments = pd.concat([eval_experiments, eval_plot])
    dic_shift = {}
    dic_shift_acc = {}

list_models = list(eval_experiments["model"].unique())
list_models.remove('catboost')
list_models.remove('ib_erm')
markermap = dict(zip(list_models.copy(),['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X',
                                  "4","1","2","3","8","."]))
list_models.remove("constant")
list_models.sort()

#%%
#############################################################################
# plot performance across experiments for each models
#############################################################################
for model in list_models:
    eval_experiments_model = eval_experiments[eval_experiments["model"] == model]

    fig = plt.figure(figsize=(6.75, 1.5))
    ax = fig.subplots(
        1, 2, gridspec_kw={"width_ratios": [0.5, 0.5], "wspace": 0.3}
    )  # create 1x4 subplots on subfig1

    ax[0].set_xlabel(f"Tasks")
    ax[0].set_ylabel(f"Out-of-domain accuracy")

    #############################################################################
    # plot ood accuracy
    #############################################################################
    markers = {"constant": "X", "all": "s", "causal": "o", "arguablycausal": "D"}

    sets = list(eval_experiments_model["features"].unique())
    sets.sort()

    for index, set in enumerate(sets):
        eval_plot_features = eval_experiments_model[eval_experiments_model["features"] == set]
        eval_plot_features = eval_plot_features.sort_values("ood_test")
        ax[0].errorbar(
            x=eval_plot_features["task"],
            y=eval_plot_features["ood_test"],
            yerr=eval_plot_features["ood_test_ub"] - eval_plot_features["ood_test"],
            color=eval(f"color_{set}"),
            ecolor=color_error,
            fmt=markers[set],
            markersize=markersize,
            capsize=capsize,
            label=set.capitalize() if set != "arguablycausal" else "Arguably causal",
            zorder=len(sets) - index,
        )
        # get pareto set for shift vs accuracy
        shift_acc = eval_plot_features
        shift_acc["type"] = set
        shift_acc["gap"] = shift_acc["ood_test"] - shift_acc["id_test"]
        shift_acc["id_test_var"] = ((shift_acc["id_test_ub"] - shift_acc["id_test"])) ** 2
        shift_acc["ood_test_var"] = ((shift_acc["ood_test_ub"] - shift_acc["ood_test"])) ** 2
        shift_acc["gap_var"] = shift_acc["id_test_var"] + shift_acc["ood_test_var"]
        dic_shift_acc[set] = shift_acc

    ax[0].tick_params(axis="x", labelrotation=90)
    ax[0].set_ylim(top=1.0)
    ax[0].grid(axis="y")


    ax[1].set_xlabel(f"Tasks")
    ax[1].set_ylabel(f"Shift gap (higher is better)")
    #############################################################################
    # plot shift gap
    #############################################################################
    shift_acc = pd.concat(dic_shift_acc.values(), ignore_index=True)
    sets = list(eval_experiments_model["features"].unique())
    sets.sort()

    for index, set in enumerate(sets):
        shift_acc_plot = shift_acc[shift_acc["features"] == set]
        shift_acc_plot = shift_acc_plot.sort_values("ood_test")
        ax[1].errorbar(
            x=shift_acc_plot["task"],
            y=shift_acc_plot["gap"],
            yerr=shift_acc_plot["gap_var"] ** 0.5,
            color=eval(f"color_{set}"),
            ecolor=color_error,
            fmt=markers[set],
            markersize=markersize,
            capsize=capsize,
            label=set.capitalize() if set != "arguablycausal" else "Arguably causal",
            zorder=len(sets) - index,
        )

    ax[1].axhline(
        y=0,
        color="black",
        linestyle="--",
    )
    ax[1].tick_params(axis="x", labelrotation=90)

    ax[1].grid(axis="y")
    # plt.tight_layout()
    fig.legend(
        list(zip(list_color, list_mak)),
        list_lab,
        handler_map={tuple: MarkerHandler()},
        loc="lower center",
        bbox_to_anchor=(0.5, -0.9),
        fancybox=True,
        ncol=5,
    )

    fig.savefig(
        str(Path(__file__).parents[0] / f"plots_rebuttal/performance_across_experiments/plot_performance_{model}.pdf"),
        bbox_inches="tight",
    )

#%%
#############################################################################
# plot performance across models for each experiment
#############################################################################

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
    "mimic_extract_mort_hosp",
    "mimic_extract_los_3",
    "nhanes_lead",
    "physionet",
    "meps",
    "sipp",
]


for index, experiment_name in enumerate(experiments):
    sns.set_style("white")
    if index % 4 == 0:
        fig = plt.figure(figsize=[6.75, 1.5 * 4])
        (subfig1, subfig2, subfig3, subfig4) = fig.subfigures(4, 1, hspace=0.5)  # create 4x1 subfigures

        subfigs = (subfig1, subfig2, subfig3, subfig4)

        ax1 = subfig1.subplots(
            1, 2, gridspec_kw={"width_ratios": [0.5, 0.5], "top": 0.85}
        )  # create 1x4 subplots on subfig1
        ax2 = subfig2.subplots(
            1, 2, gridspec_kw={"width_ratios": [0.5, 0.5], "top": 0.85}
        )  # create 1x4 subplots on subfig2
        ax3 = subfig3.subplots(
            1, 2, gridspec_kw={"width_ratios": [0.5, 0.5], "top": 0.85}
        )  # create 1x4 subplots on subfig2
        ax4 = subfig4.subplots(
            1, 2, gridspec_kw={"width_ratios": [0.5, 0.5], "top": 0.85}
        )  # create 1x4 subplots on subfig2
        axes = (ax1, ax2, ax3, ax4)
    subfig = subfigs[index % 4]
    subfig.subplots_adjust(wspace=0.4, bottom=0.3)
    ax = axes[index % 4]
    subfig.suptitle(dic_title[experiment_name], fontsize=9)  # set suptitle for subfig1
    eval_all = get_results(experiment_name)
    eval_plot = pd.DataFrame()
    for set in eval_all["features"].unique():
        eval_feature = eval_all[eval_all["features"] == set]
        for model in eval_feature["model"]:
            eval_model = eval_feature[eval_feature["model"] == model]
            if model.startswith("tableshift"):
                eval_model["model"] = encode_tableshift[model]
            if model != 'tableshift:catboost':
                eval_plot = pd.concat([eval_plot, eval_model])
    eval_all = eval_plot

    eval_constant = eval_all[eval_all["features"] == "constant"]
    dic_shift = {}
    dic_shift_acc = {}

    ax[0].xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    ax[0].set_xlabel(f"In-domain accuracy")
    ax[0].set_ylabel(f"Out-of-domain\naccuracy")

    #############################################################################
    # plot errorbars and shift gap for constant
    #############################################################################
    errors = ax[0].errorbar(
        x=eval_constant["id_test"],
        y=eval_constant["ood_test"],
        xerr=eval_constant["id_test_ub"] - eval_constant["id_test"],
        yerr=eval_constant["ood_test_ub"] - eval_constant["ood_test"],
        fmt=markermap["constant"],
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
    # plot errorbars and shift gap for causal features
    #############################################################################
    eval_plot = eval_all[eval_all["features"] == "causal"]
    eval_plot.sort_values("id_test", inplace=True)
    eval_plot = eval_plot[eval_plot["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]

    for model in list_models:
        eval_model = eval_plot[eval_plot["model"] == model]
        errors = ax[0].errorbar(
            x=eval_model["id_test"],
            y=eval_model["ood_test"],
            xerr=eval_model["id_test_ub"] - eval_model["id_test"],
            yerr=eval_model["ood_test_ub"] - eval_model["ood_test"],
            fmt=markermap[model],
            color=color_causal,
            ecolor=color_error,
            markersize=markersize,
            capsize=capsize,
            label="causal",
        )
        # get pareto set for shift vs accuracy
        shift_acc = eval_model.drop_duplicates()
        shift_acc["type"] = "causal"
        shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
        shift_acc["model"] = model
        dic_shift_acc["causal_"+model] = shift_acc
    #############################################################################
    # plot errorbars and shift gap for arguablycausal features
    #############################################################################
    if (eval_all["features"] == "arguablycausal").any():
        eval_plot = eval_all[eval_all["features"] == "arguablycausal"]
        eval_plot.sort_values("id_test", inplace=True)
        eval_plot = eval_plot[eval_plot["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]

        for model in list_models:
            eval_model = eval_plot[eval_plot["model"] == model]
            errors = ax[0].errorbar(
                x=eval_model["id_test"],
                y=eval_model["ood_test"],
                xerr=eval_model["id_test_ub"] - eval_model["id_test"],
                yerr=eval_model["ood_test_ub"] - eval_model["ood_test"],
                fmt=markermap[model],
                color=color_arguablycausal,
                ecolor=color_error,
                markersize=markersize,
                capsize=capsize,
                label="arguably\ncausal",
            )
            # get pareto set for shift vs accuracy
            shift_acc = eval_model.drop_duplicates()
            shift_acc["type"] = "arguablycausal"
            shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
            shift_acc["model"] = model
            dic_shift_acc["arguablycausal_"+model] = shift_acc

    #############################################################################
    # plot errorbars and shift gap for all features
    #############################################################################
    eval_plot = eval_all[eval_all["features"] == "all"]
    eval_plot.sort_values("id_test", inplace=True)
    eval_plot = eval_plot[eval_plot["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]

    for model in list_models:
        eval_model = eval_plot[eval_plot["model"] == model]
        
        errors = ax[0].errorbar(
            x=eval_model["id_test"],
            y=eval_model["ood_test"],
            xerr=eval_model["id_test_ub"] - eval_model["id_test"],
            yerr=eval_model["ood_test_ub"] - eval_model["ood_test"],
            fmt=markermap[model],
            color=color_all,
            ecolor=color_error,
            markersize=markersize,
            capsize=capsize,
            label="all",
        )
        # get pareto set for shift vs accuracy
        shift_acc = eval_model.drop_duplicates()
        shift_acc["type"] = "all"
        shift_acc["gap"] = shift_acc["id_test"] - shift_acc["ood_test"]
        shift_acc["model"] = model
        dic_shift_acc["all_"+model] = shift_acc


    #############################################################################
    # plot pareto dominated area for constant
    #############################################################################
    xmin, xmax = ax[0].set_xlim()
    ymin, ymax = ax[0].set_ylim()
    ax[0].plot(
        [xmin, eval_constant["id_test"].values[0]],
        [eval_constant["ood_test"].values[0], eval_constant["ood_test"].values[0]],
        color=color_constant,
        linestyle=(0, (1, 1)),
        linewidth=linewidth_bound,
    )
    ax[0].plot(
        [eval_constant["id_test"].values[0], eval_constant["id_test"].values[0]],
        [ymin, eval_constant["ood_test"].values[0]],
        color=color_constant,
        linestyle=(0, (1, 1)),
        linewidth=linewidth_bound,
    )
    ax[0].fill_between(
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
    ax[0].plot(
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
    ax[0].fill(
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
        ax[0].plot(
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
        ax[0].fill(
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
    ax[0].plot(
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
    ax[0].fill(
        points[hull.vertices, 0], points[hull.vertices, 1], color=color_all, alpha=0.05
    )


    #############################################################################
    # Add legend & diagonal, save plot
    #############################################################################
    # Plot the diagonal line
    
    start_lim = max(xmin, ymin)
    end_lim = min(xmax, ymax)
    ax[0].plot([start_lim, end_lim], [start_lim, end_lim], color=color_error)

    #############################################################################
    # Plot shift gap vs accuarcy
    #############################################################################
    if (eval_all["features"] == "arguablycausal").any():
        ax[1].set_xlabel("Shift gap")
        ax[1].set_ylabel("Out-of-domain\naccuracy")
        shift_acc = pd.concat(dic_shift_acc.values(), ignore_index=True)

        for type in list(shift_acc["type"].unique()):
            type_shift = shift_acc[shift_acc["type"] == type]
            type_shift["id_test_var"] = (
                (type_shift["id_test_ub"] - type_shift["id_test"])
            ) ** 2
            type_shift["ood_test_var"] = (
                (type_shift["ood_test_ub"] - type_shift["ood_test"])
            ) ** 2
            type_shift["gap_var"] = type_shift["id_test_var"] + type_shift["ood_test_var"]

            if type != "constant":
                for model in list_models:
                    # Get markers
                    type_shift_model = type_shift[type_shift["model"] == model]
                    ax[1].errorbar(
                        x=-type_shift_model["gap"],
                        y=type_shift_model["ood_test"],
                        xerr=type_shift_model["gap_var"] ** 0.5,
                        yerr=type_shift_model["ood_test_ub"] - type_shift_model["ood_test"],
                        color=eval(f"color_{type}"),
                        ecolor=color_error,
                        fmt=markermap[model],
                        markersize=markersize,
                        capsize=capsize,
                        label="arguably\ncausal" if type == "arguablycausal" else f"{type}",
                        zorder=3,
                    )
            else:
                ax[1].errorbar(
                        x=-type_shift["gap"],
                        y=type_shift["ood_test"],
                        xerr=type_shift["gap_var"] ** 0.5,
                        yerr=type_shift["ood_test_ub"] - type_shift["ood_test"],
                        color=eval(f"color_{type}"),
                        ecolor=color_error,
                        fmt=markermap["constant"],
                        markersize=markersize,
                        capsize=capsize,
                        label="arguably\ncausal" if type == "arguablycausal" else f"{type}",
                        zorder=3,
                    )

        xmin, xmax = ax[1].get_xlim()
        ymin, ymax = ax[1].get_ylim()

        for type in shift_acc["type"].unique():
            type_shift = shift_acc[shift_acc["type"] == type]
            # Get 1 - shift gap
            type_shift["-gap"] = -type_shift["gap"]
            # Calculate the pareto set
            points = type_shift[["-gap", "ood_test"]]
            mask = paretoset(points, sense=["max", "max"])
            points = points[mask]
            # get extra points for the plot
            new_row = pd.DataFrame(
                {
                    "-gap": [xmin, max(points["-gap"])],
                    "ood_test": [max(points["ood_test"]), ymin],
                },
            )
            points = pd.concat([points, new_row], ignore_index=True)
            points.sort_values("-gap", inplace=True)
            
            new_row = pd.DataFrame(
                {"-gap": [xmin], "ood_test": [ymin]},
            )
            points = pd.concat([points, new_row], ignore_index=True)
            points = points.to_numpy()
            hull = ConvexHull(points)
            ax[1].fill(
                points[hull.vertices, 0],
                points[hull.vertices, 1],
                color=eval(f"color_{type}"),
                alpha=0.05,
            )
            if len(hull.vertices) > 4:
                ax[1].plot(
                    points[hull.vertices[:-1], 0],
                    points[hull.vertices[:-1], 1],
                    color=eval(f"color_{type}"),
                    linestyle=(0, (1, 1)),
                    linewidth=linewidth_bound,
                )
            else:
                ax[1].plot(
                    points[[hull.vertices[-1]]+list(hull.vertices[:-2]), 0],
                    points[[hull.vertices[-1]]+list(hull.vertices[:-2]), 1],
                    color=eval(f"color_{type}"),
                    linestyle=(0, (1, 1)),
                    linewidth=linewidth_bound,
                )

    if index % 4 == 3:
        fig.legend(
            list(zip(["k" for model in list_models], markermap.values())),
            [encode_model[model] for model in list_models],
            handler_map={tuple: MarkerHandler()},
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            fancybox=True,
            ncol=5,
        )

        fig.savefig(
            str(Path(__file__).parents[0] / f"plots_rebuttal/performance_across_models/plot_performance_{int(index/4)}.pdf"),
            bbox_inches="tight",
        )

        fig.show()

# %%
