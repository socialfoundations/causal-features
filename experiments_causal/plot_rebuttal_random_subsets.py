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
from experiments_causal.plot_config_tasks import dic_domain_label

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