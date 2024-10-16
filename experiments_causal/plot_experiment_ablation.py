"""Python script to load json files of experiments with robustness test of arguably causal features."""
#%%
import json
import numpy as np
import pandas as pd
from pathlib import Path
import ast
import os
from tableshift import get_dataset
from statsmodels.stats.proportion import proportion_confint
from tableshift.datasets import *
from experiments_causal.plot_config_tasks import dic_domain_label, dic_tableshift

from experiments_causal.plot_config_colors import *
from experiments_causal.plot_config_tasks import dic_title
from scipy.spatial import ConvexHull
from paretoset import paretoset
import seaborn as sns
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import FormatStrFormatter
import matplotlib.markers as mmark
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def get_dic_experiments_value(name: str, superset: int) -> list:
    """Return list of experiment names for a task.

    Parameters
    ----------
    name : str
        The name of the task..
    superset : int
        Number of robustness tests.

    Returns
    -------
    list
        List of experiment names (all features, arguably causal features, robustness tests).

    """
    return [name] + [
        f"{name}_arguablycausal_test_{index}" for index in range(superset)
    ]


# Define dictionary to map experiments to number of robustness tests
dic_robust_number = {
    "acsincome": ACS_INCOME_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "acsfoodstamps": ACS_FOODSTAMPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "brfss_diabetes": BRFSS_DIABETES_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "brfss_blood_pressure": BRFSS_BLOOD_PRESSURE_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "diabetes_readmission": DIABETES_READMISSION_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "anes": ANES_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "acsunemployment": ACS_UNEMPLOYMENT_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "assistments": ASSISTMENTS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "college_scorecard": COLLEGE_SCORECARD_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "diabetes_readmission": DIABETES_READMISSION_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "sipp": SIPP_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "acspubcov": ACS_PUBCOV_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "meps": MEPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "physionet": PHYSIONET_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
    "nhanes_lead": NHANES_LEAD_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER,
}

# Define dictionary of all considered experiments
dic_experiments = {
    "acsincome": get_dic_experiments_value("acsincome", dic_robust_number["acsincome"]),
    "acsfoodstamps": get_dic_experiments_value(
        "acsfoodstamps", dic_robust_number["acsfoodstamps"]
    ),
    "brfss_diabetes": get_dic_experiments_value(
        "brfss_diabetes", dic_robust_number["brfss_diabetes"]
    ),
    "brfss_blood_pressure": get_dic_experiments_value(
        "brfss_blood_pressure", dic_robust_number["brfss_blood_pressure"]
    ),
    "anes": get_dic_experiments_value("anes", dic_robust_number["anes"]),
    "acsunemployment": get_dic_experiments_value(
        "acsunemployment", dic_robust_number["acsunemployment"]
    ),
    "assistments": get_dic_experiments_value(
        "assistments", dic_robust_number["assistments"]
    ),
    "college_scorecard": get_dic_experiments_value(
        "college_scorecard", dic_robust_number["college_scorecard"]
    ),
    "diabetes_readmission": get_dic_experiments_value(
        "diabetes_readmission", dic_robust_number["diabetes_readmission"]
    ),
    "sipp": get_dic_experiments_value("sipp", dic_robust_number["sipp"]),
    "acspubcov": get_dic_experiments_value("acspubcov", dic_robust_number["acspubcov"]),
    "meps": get_dic_experiments_value("meps", dic_robust_number["meps"]),
    "physionet": get_dic_experiments_value("physionet", dic_robust_number["physionet"]),
    "nhanes_lead": get_dic_experiments_value(
        "nhanes_lead", dic_robust_number["nhanes_lead"]
    ),
}

def select_non_causal(experiment_name) -> list:
    """Generate subset of all features minus one non-causal feature.

    Parameters
    ----------
    x : list
        List of features.
    allfeatures : list
        List of current and additional features.

    Returns
    -------
    list
        List of supersets of features adding one feature.

    """
    experiment_name = experiment_name.upper()
    if experiment_name.startswith("ACS"):
        experiment_name = "ACS_" + experiment_name[3:]
    x = eval(experiment_name + "_FEATURES_ARGUABLYCAUSAL.features")
    if experiment_name.startswith("ACS"):
        allfeatures = eval(experiment_name + "_FEATURES.features") + ACS_SHARED_FEATURES.features
    elif experiment_name.startswith("BRFSS"):
        allfeatures = eval(experiment_name + "_FEATURES.features") + BRFSS_SHARED_FEATURES.features
    elif experiment_name.startswith("NHANES"):
       allfeatures = eval(experiment_name + "_FEATURES.features") + NHANES_SHARED_FEATURES.features
    else:
        allfeatures = eval(experiment_name + "_FEATURES.features")
    supersets = []
    feature_names_x = [feature.name for feature in x]
    feature_names_all = [feature.name for feature in allfeatures]
    feature_names_additional = list(set(feature_names_all).difference(feature_names_x))
    additional = [feature for feature in allfeatures if feature.name in feature_names_additional]
    for item in feature_names_additional:
        supersets.append(item)
    return supersets

def get_anticausal_feature(experiment_name) -> list:
    if experiment_name in  ["acsincome","acsunemployment","brfss_diabetes","brfss_blood_pressure","sipp",]:
        experiment_name = experiment_name.upper()
        if experiment_name.startswith("ACS"):
            experiment_name = "ACS_" + experiment_name[3:]
        return [feature.name for feature in eval(experiment_name + "_FEATURES_ANTICAUSAL.features")]
    else:
        return []


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
        RESULTS_DIR = Path(__file__).parents[0] / "add_on_results" / "ablation" / experiment
        try:
            for filename in os.listdir(RESULTS_DIR):
                if filename == ".DS_Store":
                    pass
                else:
                    file_info.append(filename)
        except:
            pass

        def get_feature_selection(experiment):
            if experiment.endswith("_arguablycausal"):
                if "arguablycausal" not in feature_selection:
                    feature_selection.append("arguablycausal")
                return "arguablycausal"
            elif experiment.endswith("_los_3"):
                feature_selection.append("all")
                return "all"
            elif experiment[-2].isdigit():
                if f"test{experiment[-2]}" not in feature_selection:
                    feature_selection.append(f"test{experiment[-2:]}")
                return f"test{experiment[-2:]}"
            elif experiment[-1].isdigit():
                if f"test{experiment[-1]}" not in feature_selection:
                    feature_selection.append(f"test{experiment[-1]}")
                return f"test{experiment[-1]}"
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
                                "validation": (
                                    eval_json["validation"]
                                    if "validation" in eval_json
                                    else np.nan
                                ),
                                "features": get_feature_selection(experiment),
                                "model": run.split("_")[0],
                            }
                        ]
                    )
                    if get_feature_selection(experiment) == "arguablycausal":
                        causal_features = eval_json["features"]
                        causal_features.remove(domain_label)
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
            if not set[-1].isdigit():
                model_data = model_data[
                    model_data["validation"] == model_data["validation"].max()
                ]
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

#%%

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



import matplotlib.colors as mcolors
def lighten_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    c = mcolors.to_rgb(c)
    c = [min(1, max(0, channel + amount * (1 - channel))) for channel in c]
    return c

#%%

# Define list of experiments to plot
experiments = [
    # "acsfoodstamps",
    # "acsincome",
    # "acspubcov",
    # "acsunemployment",
    # "anes",
    # "assistments",
    # "brfss_blood_pressure",
    # "brfss_diabetes",
    # "college_scorecard",
    # "diabetes_readmission",
    "meps",
    # # "mimic_extract_mort_hosp",
    # # "mimic_extract_los_3",
    # "nhanes_lead",
    # "physionet",
    # "sipp",
]

# experiment_name = "brfss_blood_pressure"
for experiment_name in experiments:
    eval_all = get_results(experiment_name)
    eval_constant = eval_all[eval_all["features"] == "constant"]
    dic_shift = {}
    fig = plt.figure(figsize=[5.5, 1.75])
    ax = fig.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    #############################################################################
    # plot errorbars and shift gap for constant
    #############################################################################

    #############################################################################
    # plot errorbars and shift gap for all features
    #############################################################################
    eval_plot = eval_all[eval_all["features"] == "all"]
    eval_plot.sort_values("id_test", inplace=True)
    # Calculate the pareto set
    points = eval_plot[["id_test", "ood_test"]]
    mask = paretoset(points, sense=["max", "max"])
    shift = eval_plot[mask]
    shift = shift[shift["ood_test"] == shift["ood_test"].max()]
    shift["type"] = "All"
    dic_shift["all"] = shift

    #############################################################################
    # plot errorbars and shift gap for robustness tests
    #############################################################################
    existing_results = []
    for index in range(dic_robust_number[experiment_name]):
        if (eval_all["features"] == f"test{index}").any():
            existing_results.append(index)
            eval_plot = eval_all[eval_all["features"] == f"test{index}"]
            eval_plot.sort_values("id_test", inplace=True)
            # Calculate the pareto set
            points = eval_plot[["id_test", "ood_test"]]
            mask = paretoset(points, sense=["max", "max"])
            points = points[mask]
            shift = eval_plot[mask]
            shift = shift[shift["ood_test"] == shift["ood_test"].max()]
            shift["type"] = f"Test {index}"
            dic_shift[f"test{index}"] = shift

    #############################################################################
    # Plot Out-of-domain\naccuracy as bars
    #############################################################################
    # add constant shift gap
    shift = eval_constant
    shift["type"] = "Constant"
    dic_shift["constant"] = shift

    shift = pd.concat(dic_shift.values(), ignore_index=True)
    shift.drop_duplicates(inplace=True)
    # shift["gap"] = shift["id_test"] - shift["ood_test"]
    shift["type"] = ["All"] + [select_non_causal(experiment_name)[index] for index in existing_results] + ["Constant"]

    barlist = ax.bar(
        shift["type"],
        shift["ood_test"] - eval_constant["ood_test"].values[0] + 0.01,
        yerr=shift["ood_test_ub"] - shift["ood_test"],
        color=[color_all]
        + [
            color_anticausal if feature in get_anticausal_feature(experiment_name) else color_arguablycausal_robust for feature in select_non_causal(experiment_name)
        ]
        + [color_constant],
        ecolor=color_error,
        align="center",
        capsize=capsize,
        bottom=eval_constant["ood_test"].values[0] - 0.01,
    )
    plt.hlines(shift["ood_test_lb"].loc[0],xmin="All", xmax="Constant",ls="dotted",color=color_all)
    ax.tick_params(axis="x", labelrotation=90)
    if experiment_name == "meps":
        ax.tick_params(axis="x", labelrotation=90, labelsize=4)
    else:
        ax.tick_params(axis="x", labelrotation=90)
    plt.title(dic_title[experiment_name])
    ax.set_ylabel("Out-of-domain accuracy")
    fig.savefig(
        str(Path(__file__).parents[0] / f"plots_add_on/ablation/plot_{experiment_name}.pdf"),
        bbox_inches="tight",
    )
# %%
