"""Python script to get distribution shift metric and plot correlation of distribution shift and causal feature performance."""

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


def get_distances(
        experiment: str,
        cache_dir: str,
        save_dir: str):
    """Get metric of covariate shift and label shift.

    Parameters
    ----------
    experiment : str
        The name of the experiment to run.
    cache_dir : str
        Directory to cache raw data files to.
    save_dir : str
        Directory to save result files to.

    Returns
    -------
    dictionary.

    """
    cache_dir = Path(cache_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    dset = get_dataset(experiment, cache_dir)
    features_id, target_id, _, _ = dset.get_pandas("validation")
    features_ood, target_ood, _, _ = dset.get_pandas("ood_validation")

    # Get values of Dataframe and Series
    target_id = target_id.values
    features_id = features_id.values
    target_ood = target_ood.values
    features_ood = features_ood.values

    # Convert the features and target to PyTorch tensors
    features_id_tensor = torch.tensor(features_id.astype(float), dtype=torch.float32)
    target_id_tensor = torch.tensor(target_id.astype(float), dtype=torch.int)
    features_ood_tensor = torch.tensor(features_ood.astype(float), dtype=torch.float32)
    target_ood_tensor = torch.tensor(target_ood.astype(float), dtype=torch.int)

    # Create a TensorDataset
    dataset_id = data_utils.TensorDataset(features_id_tensor, target_id_tensor)
    dataset_ood = data_utils.TensorDataset(features_ood_tensor, target_ood_tensor)

    # Save distances in dictionary
    d = dict()

    # Get Optimal transport datashift distance (covariate shift + label shift)
    dist = DatasetDistance(dataset_id, dataset_ood)
    d["ot_datashift_distance"] = dist.distance().item()

    # Get L2 difference in base rates (label shift)
    d["label_l2_distance"] = torch.abs(target_id_tensor.float().mean()-target_ood_tensor.float().mean()).item()

    with open(f"{str(save_dir)}/{experiment}_distances.json", "w") as f:
        # Use json.dump to write the dictionary into the file
        json.dump(d, f)

    return d


if __name__ == "__main__":
    RESULTS_DIR = Path(
        "/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/icml-causal-features/causal-features/experiments_causal/rebuttal_results")
    CACHE_DIR = "/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/icml-causal-features/causal-features/tmp"

    list_experiments = [
        "acsfoodstamps",
        "acsincome",
        "acspubcov",
        "acsunemployment",
        "anes",
        # "assistments",
        "brfss_blood_pressure",
        "brfss_diabetes",
        "college_scorecard",
        "diabetes_readmission",
        "meps",
        # "mimic_extract_mort_hosp",
        # "mimic_extract_los_3",
        # "nhanes_lead",
        "physionet",
        "sipp",
    ]

    sns.set_style("white")
    fig = plt.figure(figsize=[6.75, 1.75])
    ax = fig.subplots(
        1,
        2,
        gridspec_kw={"width_ratios": [0.5, 0.5], "wspace": 0.3, "top": 0.8},
    )  # create 1x2 subplots on fig
    ax[0].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    colormap = np.array([color_all, color_causal, color_arguablycausal, color_anticausal])
    markermap = np.array([
        'o',
        'v',
        '^',
        '<',
        '>',
        's',
        'p',
        '*',
        'h',
        'H',
        '+',
        'x',
        'D',
        'd',
        '|',
        '_',
        'P',
        'X',
    ])

    for index, experiment in enumerate(list_experiments):
        dic_plot = {}
        if experiment in dic_experiments_anticausal.keys():
            eval_all = get_results_anticausal(experiment)
        else:
            eval_all = get_results(experiment)

        # Get data on all features
        eval_type = eval_all[eval_all["features"] == "all"]
        eval_type.sort_values("id_test", inplace=True)
        points = eval_type[["id_test", "ood_test"]]
        mask = paretoset(points, sense=["max", "max"])
        tmp = eval_type[mask]
        tmp = tmp[tmp["ood_test"] == tmp["ood_test"].max()]
        filename = f"{experiment}_distances.json"
        if filename in os.listdir(RESULTS_DIR):
            with open(str(RESULTS_DIR / filename), "rb") as file:
                d = json.load(file)
        else:
            d = get_distances(experiment=f"{experiment}", cache_dir=CACHE_DIR, save_dir=RESULTS_DIR)
        tmp["ot_datashift_distance"] = np.log(d["ot_datashift_distance"])
        tmp["label_l2_distance"] = d["label_l2_distance"]
        tmp["type"] = "all"
        tmp["color"] = 0
        tmp["experiment"] = experiment
        tmp["marker"] = index
        dic_plot["all"] = tmp

        # Get data on arguablycausal features
        eval_type = eval_all[eval_all["features"] == "arguablycausal"]
        eval_type.sort_values("id_test", inplace=True)
        points = eval_type[["id_test", "ood_test"]]
        mask = paretoset(points, sense=["max", "max"])
        tmp = eval_type[mask]
        tmp = tmp[tmp["ood_test"] == tmp["ood_test"].max()]
        filename = f"{experiment}_arguablycausal_distances.json"
        if filename in os.listdir(RESULTS_DIR):
            with open(str(RESULTS_DIR / filename), "rb") as file:
                d = json.load(file)
        else:
            d = get_distances(experiment=f"{experiment}_arguablycausal", cache_dir=CACHE_DIR, save_dir=RESULTS_DIR)
        tmp["ot_datashift_distance"] = np.log(d["ot_datashift_distance"])
        tmp["label_l2_distance"] = d["label_l2_distance"]
        tmp["type"] = "arguablycausal"
        tmp["color"] = 2
        tmp["experiment"] = experiment
        tmp["marker"] = index
        dic_plot["arguablycausal"] = tmp

        # Get data on causal features
        eval_type = eval_all[eval_all["features"] == "causal"]
        eval_type.sort_values("id_test", inplace=True)
        points = eval_type[["id_test", "ood_test"]]
        mask = paretoset(points, sense=["max", "max"])
        tmp = eval_type[mask]
        tmp = tmp[tmp["ood_test"] == tmp["ood_test"].max()]
        filename = f"{experiment}_causal_distances.json"
        if filename in os.listdir(RESULTS_DIR):
            with open(str(RESULTS_DIR / filename), "rb") as file:
                d = json.load(file)
        else:
            d = get_distances(experiment=f"{experiment}_causal", cache_dir=CACHE_DIR, save_dir=RESULTS_DIR)
        tmp["ot_datashift_distance"] = np.log(d["ot_datashift_distance"])
        tmp["label_l2_distance"] = d["label_l2_distance"]
        tmp["type"] = "causal"
        tmp["color"] = 1
        tmp["experiment"] = experiment
        tmp["marker"] = index
        dic_plot["causal"] = tmp

        # Get data on anticausal features
        if experiment in dic_experiments_anticausal.keys():
            eval_type = eval_all[eval_all["features"] == "anticausal"]
            eval_type.sort_values("id_test", inplace=True)
            points = eval_type[["id_test", "ood_test"]]
            mask = paretoset(points, sense=["max", "max"])
            tmp = eval_type[mask]
            tmp = tmp[tmp["ood_test"] == tmp["ood_test"].max()]
            filename = f"{experiment}_anticausal_distances.json"
            if filename in os.listdir(RESULTS_DIR):
                with open(str(RESULTS_DIR / filename), "rb") as file:
                    d = json.load(file)
            else:
                d = get_distances(experiment=f"{experiment}_anticausal", cache_dir=CACHE_DIR, save_dir=RESULTS_DIR)
            tmp["ot_datashift_distance"] = np.log(d["ot_datashift_distance"])
            tmp["label_l2_distance"] = d["label_l2_distance"]
            tmp["type"] = "anticausal"
            tmp["color"] = 3
            tmp["experiment"] = experiment
            tmp["marker"] = index
            dic_plot["anticausal"] = tmp

        plot_distance = pd.concat(dic_plot.values(), ignore_index=True)
        ax[0].scatter(plot_distance["ot_datashift_distance"], plot_distance["ood_test"],
                      c=colormap[plot_distance["color"]], marker=markermap[index])
        ax[1].scatter(plot_distance["label_l2_distance"], plot_distance["ood_test"],
                      c=colormap[plot_distance["color"]], marker=markermap[index])

    ax[0].set_xlabel("Log Optimal Transport Data Distance")
    ax[0].set_ylabel("Ood accuracy")
    ax[1].set_xlabel("Label Shift")
    ax[1].set_ylabel("Ood accuracy")

    fig.legend(
        list(zip(["k" for experiment in list_experiments], markermap[:len(list_experiments)+1])),
        [dic_title[experiment] for experiment in list_experiments],
        handler_map={tuple: MarkerHandler()},
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fancybox=True,
        ncol=4,
    )
    fig.show()
    fig.savefig(
        str(
            Path(__file__).parents[0]
            / f"plots_rebuttal/plot_distance.pdf"
        ),
        bbox_inches="tight",
    )
