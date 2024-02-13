"""Python script to load json files of experiments and return balanced accuracy."""
import json
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tableshift import get_dataset
from statsmodels.stats.proportion import proportion_confint
import ast
from experiments_causal.plot_config_tasks import dic_domain_label, dic_tableshift


def get_dic_experiments_value(name: str) -> list:
    """


    Parameters
    ----------
    name : str
        The name of the task.

    Returns
    -------
    list
        List of experiment names (all features, causal features, arguably causal features.

    """
    return [name, f"{name}_causal", f"{name}_arguablycausal"]


# Define dictionary of all considered experiments
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


def get_results(experiment_name: str) -> pd.DataFrame:
    """Load json files of experiments from results folder, concat them into a dataframe and save it.


    Parameters
    ----------
    experiment_name : str
        The name of the task.

    Returns
    -------
    TYPE
        Dataframe containing the results of the experiment, using balanced accuracy.

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
        for filename in os.listdir(RESULTS_DIR):
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
            else:
                if "all" not in feature_selection:
                    feature_selection.append("all")
                return "all"

        for run in file_info:
            with open(str(RESULTS_DIR / run), "rb") as file:
                try:
                    eval_json = json.load(file)
                    eval_pd = pd.DataFrame(
                        [
                            {
                                "id_test": eval_json["id_test" + "_balanced"],
                                "id_test_lb": eval_json[
                                    "id_test" + "_balanced" + "_conf"
                                ][0],
                                "id_test_ub": eval_json[
                                    "id_test" + "_balanced" + "_conf"
                                ][1],
                                "ood_test": eval_json["ood_test" + "_balanced"],
                                "ood_test_lb": eval_json[
                                    "ood_test" + "_balanced" + "_conf"
                                ][0],
                                "ood_test_ub": eval_json[
                                    "ood_test" + "_balanced" + "_conf"
                                ][1],
                                "validation": eval_json["validation"],
                                "features": get_feature_selection(experiment),
                                "model": run.split("_")[0],
                            }
                        ]
                    )
                    if get_feature_selection(experiment) == "causal":
                        causal_features = eval_json["features"]
                        causal_features.remove(domain_label)
                    if (
                        get_feature_selection(experiment) == "arguablycausal"
                        or get_feature_selection(experiment) == "causal without tuition"
                        or get_feature_selection(experiment) == "anticausal"
                    ):
                        extra_features = eval_json["features"]
                        extra_features.remove(domain_label)
                    else:
                        extra_features = []
                    eval_all = pd.concat([eval_all, eval_pd], ignore_index=True)
                except:
                    print(str(RESULTS_DIR / run))
    RESULTS_DIR = Path(__file__).parents[0] / "results"

    # Add results for constant prediction
    eval_constant = {}
    for test_split in ["id_test", "ood_test"]:
        eval_constant[test_split] = 0.5
        eval_constant[test_split + "_conf"] = (0.5, 0.5)

    # Select model with highest in-domain validation accuracy
    list_model_data = []
    for set in eval_all["features"].unique():
        eval_feature = eval_all[eval_all["features"] == set]
        for model in eval_feature["model"].unique():
            model_data = eval_feature[eval_feature["model"] == model]
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
    return eval_all, causal_features, extra_features
