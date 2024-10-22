"""
Utilities for the SIPP dataset.

For more information on the SIPP dataset, see:
    * https://www.census.gov/programs-surveys/sipp.html

For access of the SIPP dataset, see adapted code from Kim & Hardt (2023):
    /backward_predictor/sipp/data/README.md

Created for 'Predictors from Causal Features Do Not Generalize Better to New Domains'.
"""
import pandas as pd
import json
import os

from tableshift.core.features import Feature, FeatureList, cat_dtype
from tableshift.datasets.robustness import get_causal_robust, get_arguablycausal_robust, get_random_subset, get_feature_distribution
# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the directory of the current file
current_file_directory = os.path.dirname(current_file_path)

# Open the JSON file
with open(f'{current_file_directory}/sipp_feature_types.json', 'r') as file:
    # Use json.load to load the data from the file
    feature_types = json.load(file)

with open(f'{current_file_directory}/sipp_feature_demographic_types.json', 'r') as file:
    # Use json.load to load the data from the file
    feature_demographic_types = json.load(file)

type_dict = {'int8': int, 'int16': int, 'int64': int, 'category': cat_dtype, "float64": float}

SIPP_ALL_FEATURES = [Feature(x, type_dict[feature_types[x]]) for x in feature_types]
SIPP_DEMOGRAPHIC_FEATURES = [Feature(x, type_dict[feature_types[x]]) for x in feature_demographic_types]

SIPP_FEATURES = FeatureList(features=[
    Feature('OPM_RATIO', int, is_target=True,
            name_extended="Household income-to-poverty ratio in the 2019 calendar year, excluding Type 2 individuals"),
] + SIPP_ALL_FEATURES)


def preprocess_sipp(df: pd.DataFrame) -> pd.DataFrame:
    df[SIPP_FEATURES.target] = (1.0*(df[SIPP_FEATURES.target] >= 3)).astype(int)
    return df

################################################################################
# Feature list for causal, arguably causal and (if applicable) anticausal features
################################################################################


# List of causal features, that are not demographic features
list_causal_feature_names = [
    'ORIGIN',
    'HEALTHDISAB',
    'HEALTH_HEARING',
    'HEALTH_SEEING',
    'HEALTH_COGNITIVE',
    'HEALTH_AMBULATORY',
    'HEALTH_SELF_CARE',
    'HEALTH_ERRANDS_DIFFICULTY',
    'HEALTH_CORE_DISABILITY',
    'HEALTH_SUPPLEMENTAL_DISABILITY',
]
SIPP_ADDITIONAL_CAUSAL_FEATURES = [
    feature for feature in SIPP_ALL_FEATURES if feature.name in list_causal_feature_names]

SIPP_FEATURES_CAUSAL = FeatureList(features=[
    Feature('OPM_RATIO', int, is_target=True,
            name_extended="Household income-to-poverty ratio in the 2019 calendar year, excluding Type 2 individuals"),
] + SIPP_DEMOGRAPHIC_FEATURES + SIPP_ADDITIONAL_CAUSAL_FEATURES)

target = Feature('OPM_RATIO', int, is_target=True,
                 name_extended="Household income-to-poverty ratio in the 2019 calendar year,excluding Type 2 individuals")
domain = [feature for feature in SIPP_ALL_FEATURES if feature.name == 'CITIZENSHIP_STATUS'][0]
SIPP_FEATURES_CAUSAL_SUBSETS = get_causal_robust(SIPP_FEATURES_CAUSAL, target, domain)
SIPP_FEATURES_CAUSAL_SUBSETS_NUMBER = len(SIPP_FEATURES_CAUSAL_SUBSETS)

SIPP_FEATURES_DIST = get_feature_distribution(SIPP_FEATURES, target, domain)


# List of features that are neither causal nor arguably causal
list_other_feature_names = [
    'LIVING_QUARTERS_TYPE',
    'FOOD_ASSISTANCE',
    'TANF_ASSISTANCE',
    'SNAP_ASSISTANCE',
    'WIC_ASSISTANCE',
    'MEDICAID_ASSISTANCE',
]

SIPP_ONLY_ARGUABLYCAUSAL_FEATURE = [
    feature for feature in SIPP_ALL_FEATURES if feature.name not in list_other_feature_names]
SIPP_FEATURES_ARGUABLYCAUSAL = FeatureList(features=[
    Feature('OPM_RATIO', int, is_target=True,
            name_extended="Household income-to-poverty ratio in the 2019 calendar year, excluding Type 2 individuals"),
] + SIPP_ONLY_ARGUABLYCAUSAL_FEATURE)

SIPP_FEATURES_ARGUABLYCAUSAL_SUPERSETS = get_arguablycausal_robust(SIPP_FEATURES_ARGUABLYCAUSAL, SIPP_FEATURES.features)
SIPP_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER = len(SIPP_FEATURES_ARGUABLYCAUSAL_SUPERSETS)

# List of anticausal features and domain
list_anticausal_feature_names = list_other_feature_names + ['CITIZENSHIP_STATUS']

SIPP_ONLY_ANTICAUSAL_FEATURES = [
    feature for feature in SIPP_ALL_FEATURES if feature.name in list_anticausal_feature_names]
SIPP_FEATURES_ANTICAUSAL = FeatureList(features=[
    Feature('OPM_RATIO', int, is_target=True,
            name_extended="Household income-to-poverty ratio in the 2019 calendar year, excluding Type 2 individuals"),
] + SIPP_ONLY_ANTICAUSAL_FEATURES)

SIPP_FEATURES_RANDOM_SUBSETS = get_random_subset(SIPP_FEATURES.features, target, domain, k=500)
SIPP_FEATURES_RANDOM_SUBSETS_NUMBER = len(SIPP_FEATURES_RANDOM_SUBSETS)
