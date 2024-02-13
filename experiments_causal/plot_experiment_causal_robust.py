"""Python script to load json files of experiments with robustness test of causal features."""

import json
import numpy as np
import pandas as pd
import os
from pathlib import Path
import ast
from tableshift import get_dataset
from statsmodels.stats.proportion import proportion_confint
from tableshift.datasets import (
    ACS_INCOME_FEATURES_CAUSAL_SUBSETS_NUMBER,
    ACS_FOODSTAMPS_FEATURES_CAUSAL_SUBSETS_NUMBER,
    ACS_PUBCOV_FEATURES_CAUSAL_SUBSETS_NUMBER,
    ACS_UNEMPLOYMENT_FEATURES_CAUSAL_SUBSETS_NUMBER,
    BRFSS_DIABETES_FEATURES_CAUSAL_SUBSETS_NUMBER,
    BRFSS_BLOOD_PRESSURE_FEATURES_CAUSAL_SUBSETS_NUMBER,
    DIABETES_READMISSION_FEATURES_CAUSAL_NUMBER,
    ANES_FEATURES_CAUSAL_SUBSETS_NUMBER,
    ASSISTMENTS_FEATURES_CAUSAL_SUBSETS_NUMBER,
    COLLEGE_SCORECARD_FEATURES_CAUSAL_SUBSETS_NUMBER,
    MIMIC_EXTRACT_LOS_3_FEATURES_CAUSAL_SUBSETS_NUMBER,
    MIMIC_EXTRACT_MORT_HOSP_FEATURES_CAUSAL_SUBSETS_NUMBER,
    SIPP_FEATURES_CAUSAL_SUBSETS_NUMBER,
    MEPS_FEATURES_CAUSAL_SUBSETS_NUMBER,
    PHYSIONET_FEATURES_CAUSAL_SUBSETS_NUMBER,
    NHANES_LEAD_FEATURES_CAUSAL_SUBSETS_NUMBER,
)
from experiments_causal.plot_config_tasks import dic_domain_label, dic_tableshift


def get_dic_experiments_value(name: str, subset: int) -> list:
    """Return list of experiment names for a task.

    Parameters
    ----------
    name : str
        The name of the task..
    subset : int
        Number of robustness tests.

    Returns
    -------
    list
        List of experiment names (all features, causal features, robustness tests).

    """
    return [name, f"{name}_causal"] + [
        f"{name}_causal_test_{index}" for index in range(subset)
    ]


# Define dictionary to map experiments to number of robustness tests
dic_robust_number = {
    "acsincome": ACS_INCOME_FEATURES_CAUSAL_SUBSETS_NUMBER,
    "acsfoodstamps": ACS_FOODSTAMPS_FEATURES_CAUSAL_SUBSETS_NUMBER,
    "brfss_diabetes": BRFSS_DIABETES_FEATURES_CAUSAL_SUBSETS_NUMBER,
    "brfss_blood_pressure": BRFSS_BLOOD_PRESSURE_FEATURES_CAUSAL_SUBSETS_NUMBER,
    "diabetes_readmission": DIABETES_READMISSION_FEATURES_CAUSAL_NUMBER,
    "anes": ANES_FEATURES_CAUSAL_SUBSETS_NUMBER,
    "acsunemployment": ACS_UNEMPLOYMENT_FEATURES_CAUSAL_SUBSETS_NUMBER,
    "assistments": ASSISTMENTS_FEATURES_CAUSAL_SUBSETS_NUMBER,
    "college_scorecard": COLLEGE_SCORECARD_FEATURES_CAUSAL_SUBSETS_NUMBER,
    "diabetes_readmission": DIABETES_READMISSION_FEATURES_CAUSAL_NUMBER,
    "mimic_extract_los_3": MIMIC_EXTRACT_LOS_3_FEATURES_CAUSAL_SUBSETS_NUMBER,
    "mimic_extract_mort_hosp": MIMIC_EXTRACT_MORT_HOSP_FEATURES_CAUSAL_SUBSETS_NUMBER,
    "sipp": SIPP_FEATURES_CAUSAL_SUBSETS_NUMBER,
    "acspubcov": ACS_PUBCOV_FEATURES_CAUSAL_SUBSETS_NUMBER,
    "meps": MEPS_FEATURES_CAUSAL_SUBSETS_NUMBER,
    "physionet": PHYSIONET_FEATURES_CAUSAL_SUBSETS_NUMBER,
    "nhanes_lead": NHANES_LEAD_FEATURES_CAUSAL_SUBSETS_NUMBER-1,
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
    "mimic_extract_los_3": get_dic_experiments_value(
        "mimic_extract_los_3", dic_robust_number["mimic_extract_los_3"]
    ),
    "mimic_extract_mort_hosp": get_dic_experiments_value(
        "mimic_extract_mort_hosp", dic_robust_number["mimic_extract_mort_hosp"]
    ),
    "sipp": get_dic_experiments_value("sipp", dic_robust_number["sipp"]),
    "acspubcov": get_dic_experiments_value("acspubcov", dic_robust_number["acspubcov"]),
    "meps": get_dic_experiments_value("meps", dic_robust_number["meps"]),
    "physionet": get_dic_experiments_value("physionet", dic_robust_number["physionet"]),
    "nhanes_lead": get_dic_experiments_value(
        "nhanes_lead", dic_robust_number["nhanes_lead"]
    ),
}


def get_results(experiment_name) -> pd.DataFrame:
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
                try:
                    # print(str(RESULTS_DIR / run))
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
                    if get_feature_selection(experiment) == "causal":
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
