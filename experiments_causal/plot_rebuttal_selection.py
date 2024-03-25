"""Python script to display model selection."""

import json
import numpy as np
import pandas as pd
import os
from pathlib import Path
import ast
from tableshift import get_dataset
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm
from experiments_causal.plot_config_tasks import dic_domain_label, dic_tableshift
import seaborn as sns
from paretoset import paretoset
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import FormatStrFormatter
import matplotlib.markers as mmark

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
]
list_lab = ["All", "Arguably causal", "Causal", "Constant"]
list_color = [color_all, color_arguablycausal, color_causal, color_constant]

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
    return [name, f"{name}_causal", f"{name}_arguablycausal"]


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


def get_results_step_0(experiment_name: str) -> pd.DataFrame:
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
    cache_dir = "tmp"
    experiments = dic_experiments[experiment_name]
    domain_label = dic_domain_label[experiment_name]

    # Load all json files of experiments
    eval_all = pd.DataFrame()
    feature_selection = []
    for experiment in experiments:
        file_info = []
        RESULTS_DIR = Path(__file__).parents[0] / "results" / experiment
        for filename in tqdm(os.listdir(RESULTS_DIR)):
            if filename == ".DS_Store":
                pass
            else:
                file_info.append(filename)

        def get_feature_selection(experiment):
            if experiment.endswith("_causal"):
                if "causal" not in feature_selection:
                    feature_selection.append("causal")
                return "causal"
            elif experiment.endswith("_arguablycausal"):
                if "arguablycausal" not in feature_selection:
                    feature_selection.append("arguablycausal")
                return "arguablycausal"
            elif experiment.endswith("_anticausal"):
                if "anticausal" not in feature_selection:
                    feature_selection.append("anticausal")
                return "anticausal"
            else:
                if "all" not in feature_selection:
                    feature_selection.append("all")
                return "all"

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
                                "features": get_feature_selection(experiment),
                                "model": run.split("_", 2)[0] + "_" + run.split("_", 2)[1] if run.split("_")[0] in ["ib", "and", "causirl"] else run.split("_")[0],
                                "number": len(eval_json["features"]),
                            }
                        ]
                    )
                    if get_feature_selection(experiment) == "causal":
                        causal_features = eval_json["features"]
                        causal_features.remove(domain_label)
                    if (
                        get_feature_selection(experiment) == "arguablycausal"
                        or get_feature_selection(experiment) == "anticausal"
                    ):
                        extra_features = eval_json["features"]
                        extra_features.remove(domain_label)
                    else:
                        extra_features = []
                    eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)
                except:
                    print(str(RESULTS_DIR / run))

    # Load or add results for constant prediction
    RESULTS_DIR = Path(__file__).parents[0] / "results"
    filename = f"{experiment_name}_constant"
    if filename in os.listdir(RESULTS_DIR):
        with open(str(RESULTS_DIR / filename), "rb") as file:
            # print(str(RESULTS_DIR / filename))
            eval_constant = json.load(file)
    else:
        eval_constant = {}
        dset = get_dataset(experiment_name, cache_dir)
        for test_split in ["id_test", "ood_test"]:
            X_te, y_te, _, _ = dset.get_pandas(test_split)
            majority_class = y_te.mode()[0]
            count = y_te.value_counts()[majority_class]
            nobs = len(y_te)
            acc = count / nobs
            acc_conf = proportion_confint(count, nobs, alpha=0.05, method="beta")

            eval_constant[test_split] = acc
            eval_constant[test_split + "_conf"] = acc_conf
        with open(str(RESULTS_DIR / filename), "w") as file:
            json.dump(eval_constant, file)

    eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)

    return eval_all

def get_results_step_1(experiment_name: str) -> pd.DataFrame:
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
    cache_dir = "tmp"
    experiments = dic_experiments[experiment_name]
    domain_label = dic_domain_label[experiment_name]

    # Load all json files of experiments
    eval_all = pd.DataFrame()
    feature_selection = []
    for experiment in experiments:
        file_info = []
        RESULTS_DIR = Path(__file__).parents[0] / "results" / experiment
        for filename in tqdm(os.listdir(RESULTS_DIR)):
            if filename == ".DS_Store":
                pass
            else:
                file_info.append(filename)

        def get_feature_selection(experiment):
            if experiment.endswith("_causal"):
                if "causal" not in feature_selection:
                    feature_selection.append("causal")
                return "causal"
            elif experiment.endswith("_arguablycausal"):
                if "arguablycausal" not in feature_selection:
                    feature_selection.append("arguablycausal")
                return "arguablycausal"
            elif experiment.endswith("_anticausal"):
                if "anticausal" not in feature_selection:
                    feature_selection.append("anticausal")
                return "anticausal"
            else:
                if "all" not in feature_selection:
                    feature_selection.append("all")
                return "all"

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
                                "features": get_feature_selection(experiment),
                                "model": run.split("_", 2)[0] + "_" + run.split("_", 2)[1] if run.split("_")[0] in ["ib", "and", "causirl"] else run.split("_")[0],
                                "number": len(eval_json["features"]),
                            }
                        ]
                    )
                    if get_feature_selection(experiment) == "causal":
                        causal_features = eval_json["features"]
                        causal_features.remove(domain_label)
                    if (
                        get_feature_selection(experiment) == "arguablycausal"
                        or get_feature_selection(experiment) == "anticausal"
                    ):
                        extra_features = eval_json["features"]
                        extra_features.remove(domain_label)
                    else:
                        extra_features = []
                    eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)
                except:
                    print(str(RESULTS_DIR / run))

    # Load or add results for constant prediction
    RESULTS_DIR = Path(__file__).parents[0] / "results"
    filename = f"{experiment_name}_constant"
    if filename in os.listdir(RESULTS_DIR):
        with open(str(RESULTS_DIR / filename), "rb") as file:
            # print(str(RESULTS_DIR / filename))
            eval_constant = json.load(file)
    else:
        eval_constant = {}
        dset = get_dataset(experiment_name, cache_dir)
        for test_split in ["id_test", "ood_test"]:
            X_te, y_te, _, _ = dset.get_pandas(test_split)
            majority_class = y_te.mode()[0]
            count = y_te.value_counts()[majority_class]
            nobs = len(y_te)
            acc = count / nobs
            acc_conf = proportion_confint(count, nobs, alpha=0.05, method="beta")

            eval_constant[test_split] = acc
            eval_constant[test_split + "_conf"] = acc_conf
        with open(str(RESULTS_DIR / filename), "w") as file:
            json.dump(eval_constant, file)

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

    eval_pd = pd.DataFrame(
        [
            {
                "id_test": eval_constant["id_test"],
                "id_test_lb": eval_constant["id_test_conf"][0],
                "id_test_ub": eval_constant["id_test_conf"][1],
                "ood_test": eval_constant["ood_test"],
                "ood_test_lb": eval_constant["ood_test_conf"][0],
                "ood_test_ub": eval_constant["ood_test_conf"][1],
                "features": "constant",
                "model": "constant",
            }
        ]
    )

    eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)
    return eval_all

def get_results(experiment_name: str) -> pd.DataFrame:
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
    cache_dir = "tmp"
    experiments = dic_experiments[experiment_name]
    domain_label = dic_domain_label[experiment_name]

    # Load all json files of experiments
    eval_all = pd.DataFrame()
    feature_selection = []
    for experiment in experiments:
        file_info = []
        RESULTS_DIR = Path(__file__).parents[0] / "results" / experiment
        for filename in tqdm(os.listdir(RESULTS_DIR)):
            if filename == ".DS_Store":
                pass
            else:
                file_info.append(filename)

        def get_feature_selection(experiment):
            if experiment.endswith("_causal"):
                if "causal" not in feature_selection:
                    feature_selection.append("causal")
                return "causal"
            elif experiment.endswith("_arguablycausal"):
                if "arguablycausal" not in feature_selection:
                    feature_selection.append("arguablycausal")
                return "arguablycausal"
            elif experiment.endswith("_anticausal"):
                if "anticausal" not in feature_selection:
                    feature_selection.append("anticausal")
                return "anticausal"
            else:
                if "all" not in feature_selection:
                    feature_selection.append("all")
                return "all"

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
                                "features": get_feature_selection(experiment),
                                "model": run.split("_", 2)[0] + "_" + run.split("_", 2)[1] if run.split("_")[0] in ["ib", "and", "causirl"] else run.split("_")[0],
                                "number": len(eval_json["features"]),
                            }
                        ]
                    )
                    if get_feature_selection(experiment) == "causal":
                        causal_features = eval_json["features"]
                        causal_features.remove(domain_label)
                    if (
                        get_feature_selection(experiment) == "arguablycausal"
                        or get_feature_selection(experiment) == "anticausal"
                    ):
                        extra_features = eval_json["features"]
                        extra_features.remove(domain_label)
                    else:
                        extra_features = []
                    eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)
                except:
                    print(str(RESULTS_DIR / run))

    # Load or add results for constant prediction
    RESULTS_DIR = Path(__file__).parents[0] / "results"
    filename = f"{experiment_name}_constant"
    if filename in os.listdir(RESULTS_DIR):
        with open(str(RESULTS_DIR / filename), "rb") as file:
            # print(str(RESULTS_DIR / filename))
            eval_constant = json.load(file)
    else:
        eval_constant = {}
        dset = get_dataset(experiment_name, cache_dir)
        for test_split in ["id_test", "ood_test"]:
            X_te, y_te, _, _ = dset.get_pandas(test_split)
            majority_class = y_te.mode()[0]
            count = y_te.value_counts()[majority_class]
            nobs = len(y_te)
            acc = count / nobs
            acc_conf = proportion_confint(count, nobs, alpha=0.05, method="beta")

            eval_constant[test_split] = acc
            eval_constant[test_split + "_conf"] = acc_conf
        with open(str(RESULTS_DIR / filename), "w") as file:
            json.dump(eval_constant, file)

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

    eval_pd = pd.DataFrame(
        [
            {
                "id_test": eval_constant["id_test"],
                "id_test_lb": eval_constant["id_test_conf"][0],
                "id_test_ub": eval_constant["id_test_conf"][1],
                "ood_test": eval_constant["ood_test"],
                "ood_test_lb": eval_constant["ood_test_conf"][0],
                "ood_test_ub": eval_constant["ood_test_conf"][1],
                "features": "constant",
                "model": "constant",
            }
        ]
    )

    eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)

    # Add best results provided in TableShift
    if experiment_name in dic_tableshift.keys():
        tableshift_results = pd.read_csv(
            str(
                Path(__file__).parents[0].parents[0]
                / "results"
                / "best_id_accuracy_results_by_task_and_model.csv"
            )
        )
        tableshift_results = tableshift_results[
            tableshift_results["task"] == dic_tableshift[experiment_name]
        ]

        tableshift_results["test_accuracy_clopper_pearson_95%_interval"] = (
            tableshift_results["test_accuracy_clopper_pearson_95%_interval"].apply(
                lambda s: ast.literal_eval(s) if s is not np.nan else np.nan
            )
        )
        tableshift_results_id = tableshift_results[
            tableshift_results["in_distribution"] == True
        ]
        tableshift_results_id.reset_index(inplace=True)
        tableshift_results_ood = tableshift_results[
            tableshift_results["in_distribution"] == False
        ]
        tableshift_results_ood.reset_index(inplace=True)
        for model in tableshift_results["estimator"].unique():
            model_tableshift_results_id = tableshift_results_id[
                tableshift_results_id["estimator"] == model
            ]
            model_tableshift_results_id.reset_index(inplace=True)
            model_tableshift_results_ood = tableshift_results_ood[
                tableshift_results_ood["estimator"] == model
            ]
            model_tableshift_results_ood.reset_index(inplace=True)
            try:
                eval_pd = pd.DataFrame(
                    [
                        {
                            "id_test": model_tableshift_results_id["test_accuracy"][0],
                            "id_test_lb": model_tableshift_results_id[
                                "test_accuracy_clopper_pearson_95%_interval"
                            ][0][0],
                            "id_test_ub": model_tableshift_results_id[
                                "test_accuracy_clopper_pearson_95%_interval"
                            ][0][1],
                            "ood_test": model_tableshift_results_ood["test_accuracy"][0],
                            "ood_test_lb": model_tableshift_results_ood[
                                "test_accuracy_clopper_pearson_95%_interval"
                            ][0][0],
                            "ood_test_ub": model_tableshift_results_ood[
                                "test_accuracy_clopper_pearson_95%_interval"
                            ][0][1],
                            "validation": np.nan,
                            "features": "all",
                            "model": f"tableshift:{model_tableshift_results_id['estimator'][0].lower()}",
                        }
                    ]
                )
            except:
                print(experiment_name, model)
            eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)

    return eval_all


def plot_results_simple(experiments: list, get_function, step: int):
    sns.set_style("white")

    fig = plt.figure(figsize=(6.75, 1.75))
    axes = fig.subplots(
                1,
                3,
                gridspec_kw={"width_ratios": [0.3, 0.3, 0.3], "wspace": 0.6, "top": 0.8},
            )  # create 3x2 subplots on fig
    for index, ax in enumerate(axes):
        experiment_name = experiments[index]
        ax.set_title(dic_title[experiment_name], fontsize=9)  # set suptitle for subfig1
        eval_all = get_function(experiment_name)

        eval_constant = eval_all[eval_all["features"] == "constant"]
        dic_shift_acc = {}

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

        ax.set_xlabel(f"In-domain accuracy")
        ax.set_ylabel(f"Out-of-domain\naccuracy")

        #############################################################################
        # plot errorbars and shift gap for constant
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
        # plot errorbars and shift gap for causal features
        #############################################################################
        eval_plot = eval_all[eval_all["features"] == "causal"]
        eval_plot.sort_values("id_test", inplace=True)
        errors = ax.errorbar(
            x=eval_plot["id_test"],
            y=eval_plot["ood_test"],
            xerr=eval_plot["id_test_ub"] - eval_plot["id_test"],
            yerr=eval_plot["ood_test_ub"] - eval_plot["ood_test"],
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
        # plot errorbars and shift gap for arguablycausal features
        #############################################################################
        if (eval_all["features"] == "arguablycausal").any():
            eval_plot = eval_all[eval_all["features"] == "arguablycausal"]
            eval_plot.sort_values("id_test", inplace=True)
            errors = ax.errorbar(
                x=eval_plot["id_test"],
                y=eval_plot["ood_test"],
                xerr=eval_plot["id_test_ub"] - eval_plot["id_test"],
                yerr=eval_plot["ood_test_ub"] - eval_plot["ood_test"],
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
            # plot errorbars and shift gap for all features
            #############################################################################
            eval_plot = eval_all[eval_all["features"] == "all"]
            eval_plot.sort_values("id_test", inplace=True)
            errors = ax.errorbar(
                x=eval_plot["id_test"],
                y=eval_plot["ood_test"],
                xerr=eval_plot["id_test_ub"] - eval_plot["id_test"],
                yerr=eval_plot["ood_test_ub"] - eval_plot["ood_test"],
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

        xmin, xmax = ax.set_xlim()
        ymin, ymax = ax.set_ylim()
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
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        ncol=5,
    )

    fig.savefig(
        str(Path(__file__).parents[0] / f"plots_rebuttal/plot_selection_step_{step}.pdf"),
        bbox_inches="tight",
    )

def plot_results_pareto(experiment_name:list):
    sns.set_style("white")

    fig = plt.figure(figsize=(6.75, 1.75))
    axes = fig.subplots(
                1,
                3,
                gridspec_kw={"width_ratios": [0.3, 0.3, 0.3], "wspace": 0.6, "top": 0.8},
            )  # create 3x2 subplots on fig
    for index, ax in enumerate(axes):
        experiment_name = experiments[index]
        ax.set_title(dic_title[experiment_name], fontsize=9)  # set suptitle for subfig1
        eval_all = get_results(experiment_name)

        eval_constant = eval_all[eval_all["features"] == "constant"]
        dic_shift_acc = {}

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

        ax.set_xlabel(f"In-domain accuracy")
        ax.set_ylabel(f"Out-of-domain\naccuracy")

        #############################################################################
        # plot errorbars and shift gap for constant
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
        # plot errorbars and shift gap for causal features
        #############################################################################
        eval_plot = eval_all[eval_all["features"] == "causal"]
        eval_plot.sort_values("id_test", inplace=True)
        # Calculate the pareto set
        points = eval_plot[["id_test", "ood_test"]]
        mask = paretoset(points, sense=["max", "max"])
        markers = eval_plot[mask]
        # markers = markers[markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]
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
        # plot errorbars and shift gap for arguablycausal features
        #############################################################################
        if (eval_all["features"] == "arguablycausal").any():
            eval_plot = eval_all[eval_all["features"] == "arguablycausal"]
            eval_plot.sort_values("id_test", inplace=True)
            # Calculate the pareto set
            points = eval_plot[["id_test", "ood_test"]]
            mask = paretoset(points, sense=["max", "max"])
            markers = eval_plot[mask]
            # markers = markers[
            #     markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)
            # ]
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
        # plot errorbars and shift gap for all features
        #############################################################################
        eval_plot = eval_all[eval_all["features"] == "all"]
        eval_plot.sort_values("id_test", inplace=True)
        # Calculate the pareto set
        points = eval_plot[["id_test", "ood_test"]]
        mask = paretoset(points, sense=["max", "max"])
        markers = eval_plot[mask]
        # markers = markers[markers["id_test"] >= (eval_constant["id_test"].values[0] - 0.01)]
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

        
        xmin, xmax = ax.set_xlim()
        ymin, ymax = ax.set_ylim()
        
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
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True,
            ncol=5,
        )

        fig.savefig(
            str(Path(__file__).parents[0] / f"plots_rebuttal/plot_selection_step_3.pdf"),
            bbox_inches="tight",
        )

def plot_results(experiments:list):
    sns.set_style("white")

    fig = plt.figure(figsize=(6.75, 1.75))
    axes = fig.subplots(
                1,
                3,
                gridspec_kw={"width_ratios": [0.3, 0.3, 0.3], "wspace": 0.6, "top": 0.8},
            )  # create 3x2 subplots on fig
    for index, ax in enumerate(axes):
        experiment_name = experiments[index]
        ax.set_title(dic_title[experiment_name], fontsize=9)  # set suptitle for subfig1
        eval_all = get_results(experiment_name)

        eval_constant = eval_all[eval_all["features"] == "constant"]
        dic_shift_acc = {}

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

        ax.set_xlabel(f"In-domain accuracy")
        ax.set_ylabel(f"Out-of-domain\naccuracy")

        #############################################################################
        # plot errorbars and shift gap for constant
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
        # plot errorbars and shift gap for causal features
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
        # plot errorbars and shift gap for arguablycausal features
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
        # plot errorbars and shift gap for all features
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
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True,
            ncol=5,
        )

        fig.savefig(
            str(Path(__file__).parents[0] / f"plots_rebuttal/plot_selection_step_4.pdf"),
            bbox_inches="tight",
        )


experiments = [
    # "acsfoodstamps",
    "acsincome",
    "acspubcov",
    # "acsunemployment",
    # "anes",
    # "assistments",
    "brfss_blood_pressure",
    # "brfss_diabetes",
    # "college_scorecard",
    # "diabetes_readmission",
    # "mimic_extract_mort_hosp",
    # "mimic_extract_los_3",
    # "nhanes_lead",
    # "physionet",
    # "meps",
    # "sipp",
]

plot_results_simple(experiments, get_results_step_0, 0)
    
plot_results_simple(experiments, get_results_step_1, 1)

plot_results_simple(experiments, get_results, 2)

plot_results_pareto(experiments)

plot_results(experiments)