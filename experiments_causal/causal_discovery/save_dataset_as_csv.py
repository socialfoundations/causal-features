"""Python script to run experiment and record the performance."""
import argparse
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score
import json
from tableshift.core.features import cat_dtype
from statsmodels.stats.proportion import proportion_confint

from tableshift import get_dataset
from tableshift.models.training import train
from tableshift.models.utils import get_estimator
from tableshift.models.default_hparams import get_default_config
from tableshift.datasets import *
from experiments_causal.plot_config_tasks import dic_domain_label

from sklearn.preprocessing import LabelEncoder

cache_dir = "/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/causal-features/tmp"
cache_dir = Path(cache_dir)

experiment = "brfss_blood_pressure"
experiment_name = "bloodpressure"
feature_list = BRFSS_BLOOD_PRESSURE_FEATURES

# experiment = "physionet"
# experiment_name = "sepsis"
# feature_list = PHYSIONET_FEATURES

# experiment = "meps"
# experiment_name = "utilization"
# feature_list = MEPS_FEATURES

# experiment = "sipp"
# experiment_name = "poverty"
# feature_list = SIPP_FEATURES

# experiment = "acspubcov"
# experiment_name = "pubcov"
# feature_list = ACS_PUBCOV_FEATURES

# experiment = "assistments"
# experiment_name = "assistments"
# feature_list = ASSISTMENTS_FEATURES

# experiment = "nhanes_lead"
# experiment_name = "lead"
# feature_list = NHANES_LEAD_FEATURES

# experiment = "physionet"
# experiment_name = "sepsis"
# feature_list = PHYSIONET_FEATURES

target = feature_list.target
domain = dic_domain_label[experiment]
execption = [] # voting ['VCF0104','VCF0105a'], readmission ["race","gender"]
dset = get_dataset(experiment, cache_dir)

# Case: non-pytorch estimator; perform test-split evaluation.

test_split = "validation"
# Fetch predictions and labels for a sklearn model.
X_te, y_te, _, domains_te = dset.get_pandas(test_split)

discovery_data = X_te
discovery_data["target"] = y_te
discovery_data["domain"] = domains_te
discovery_data.to_csv(f"/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/causal-features/tmp_preprocessed/{experiment_name}.csv", index = False)

feature_list.to_jsonl(f"/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/causal-features/tmp_preprocessed/{experiment_name}_variables.json")

# Get categorical variables
tmp = list()
data_tmp = list()
for feature in feature_list:
    if feature.kind == cat_dtype:
        if (feature.name != domain) & (feature.name != target) & (feature.name not in execption):
            tmp.append(feature.name)
            if experiment_name == "college":
                discovery_data_tmp = pd.DataFrame(discovery_data[feature.name])
            else:
                discovery_data_tmp = pd.from_dummies(discovery_data[[col for col in discovery_data.columns if col.startswith(feature.name)]], sep="_")
            if len(discovery_data.columns)>1:
                data_tmp.append(pd.DataFrame(discovery_data_tmp.iloc[:,0]).copy())
            else:
                data_tmp.append(discovery_data_tmp.copy())

# Get dataset with categorical variables
discovery_data_categories = pd.DataFrame()
for feature in feature_list:
    if (feature.kind != cat_dtype) | (feature.name in execption):
        if (feature.name != domain) & (feature.name != target):
            discovery_data_categories[feature.name] = discovery_data[feature.name]

for col in data_tmp:
    LE = LabelEncoder()
    discovery_data_categories[col.columns[0]] = pd.Series(LE.fit_transform(col),index = discovery_data.index)

discovery_data_categories["target"] = discovery_data["target"]
discovery_data_categories["domain"] = discovery_data["domain"]

discovery_data_categories.to_csv(f"/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/causal-features/tmp_preprocessed/{experiment_name}_categories.csv", index = False)

# Get dataset with categorical variables and discrete intervals for continous variables
maximal_bins = 5

discovery_data_discrete = pd.DataFrame()
for feature in feature_list:
    if (feature.kind != cat_dtype) | (feature.name in execption):
        if (feature.name != domain) & (feature.name != target):
            number_values = len(discovery_data[feature.name].unique())
            number_bins = int(min(number_values,maximal_bins))
            discovery_data_discrete[feature.name] = pd.cut(x=discovery_data[feature.name],bins=number_bins,labels=False)

for col in data_tmp:
    LE = LabelEncoder()
    discovery_data_discrete[col.columns[0]] = pd.Series(LE.fit_transform(col),index = discovery_data.index)

discovery_data_discrete["target"] = discovery_data["target"]
discovery_data_discrete["domain"] = discovery_data["domain"]

discovery_data_discrete.to_csv(f"/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/causal-features/tmp_preprocessed/{experiment_name}_discrete_5.csv", index = False)

# Get dataset with categorical variables and discrete intervals for continous variables
maximal_bins = 10

discovery_data_discrete = pd.DataFrame()
for feature in feature_list:
   if (feature.kind != cat_dtype) | (feature.name in execption):
        if (feature.name != domain) & (feature.name != target):
            number_values = len(discovery_data[feature.name].unique())
            number_bins = int(min(number_values,maximal_bins))
            discovery_data_discrete[feature.name] = pd.cut(x=discovery_data[feature.name],bins=number_bins,labels=False)

for col in data_tmp:
    LE = LabelEncoder()
    discovery_data_discrete[col.columns[0]] = pd.Series(LE.fit_transform(col),index = discovery_data.index)

discovery_data_discrete["target"] = discovery_data["target"]
discovery_data_discrete["domain"] = discovery_data["domain"]

discovery_data_discrete.to_csv(f"/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/causal-features/tmp_preprocessed/{experiment_name}_discrete_10.csv", index = False)
