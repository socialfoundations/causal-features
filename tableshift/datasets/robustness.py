"""Python script to define functions for creating robustness tests.

Created for 'Predictors from Causal Features Do Not Generalize Better to New Domains'.
"""

import numpy as np
import pandas as pd
import torch
from itertools import combinations
import random
from tableshift.core.features import Feature, FeatureList


def select_powerset(x: pd.DataFrame) -> list:
    """Generate powerset of columns.

    Parameters
    ----------
    x : pd.DataFrame
        The dataframe containing the features.

    Returns
    -------
    powerset : list
        List of powerset of column names.

    """
    powerset = []
    for r in range(len(x.columns) + 1):
        combinations_r = list(combinations(x.columns, r))
        for item in combinations_r:
            powerset.append(list(item))
    for item in powerset:
        if len(item) == 0:
            powerset.remove(item)
    return powerset
    
def select_subset_minus_one(x: list) -> list:
    """Generate subsets missing one feature.

    Parameters
    ----------
    x : list
        List of features.

    Returns
    -------
    subsets : list
        List of subsets of feature missing one feature.

    """
    subsets = []
    r = len(x)-1
    combinations_r = list(combinations(x, r))
    for item in combinations_r:
        subsets.append(list(item))
    return subsets


def select_superset_plus_one(x: list, allfeatures: list) -> list:
    """Generate superset adding one feature.

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
    supersets = []
    feature_names_x = [feature.name for feature in x]
    feature_names_all = [feature.name for feature in allfeatures]
    feature_names_additional = list(set(feature_names_all).difference(feature_names_x))
    additional = [feature for feature in allfeatures if feature.name in feature_names_additional]
    for item in additional:
        supersets.append(x+[item])
    return supersets


def get_causal_robust(
        featurelist: FeatureList,
        target: Feature,
        domain: Feature) -> list:
    """Generate robustness test for causal features

    Parameters
    ----------
    featurelist : FeatureList
        Causal features.
    target : Feature
        Target.
    domain : Feature
        Domain variable.

    Returns
    -------
    list
        List of robustness tests.

    """
    causal_features = featurelist.features.copy()
    if target in causal_features:
        causal_features.remove(target)
    causal_features.remove(domain)
    causal_subsets = select_subset_minus_one(causal_features)
    subsets = []
    for subset in causal_subsets:
        subset.append(target)
        subset.append(domain)
        subsets.append(FeatureList(subset))
    return subsets


def get_arguablycausal_robust(
        featurelist: FeatureList,
        allfeatures: list) -> list:
    """Generate robustness test for arguably causal features

    Parameters
    ----------
    featurelist : FeatureList
        Arguably causal features.
    allfeatures: list
        List of all features names.

    Returns
    -------
    list
        List of robustness tests.

    """
    arguablycausal_supersets = select_superset_plus_one(
        featurelist.features, allfeatures)
    supersets = []
    for superset in arguablycausal_supersets:
        supersets.append(FeatureList(superset))
    return supersets

def get_random_subset(featurelist: FeatureList,
        target: Feature,
        domain: Feature,
        k: int) -> list:
    """Generate random subset of features

    Parameters
    ----------
    featurelist : list
        List of all features.
    target : Feature
        Target.
    domain : Feature
        Domain variable.
    k: int
        Number of random subsets

    Returns
    -------
    list
        List of random subsets.

    """
    random.seed(42)

    features = featurelist.copy()
    if target in features:
        features.remove(target)
    features.remove(domain)
    powerset = select_powerset(features)

    k = min(k,len(powerset))
    random_subsets = random.sample(powerset, k)
    subsets = []
    for subset in random_subsets:
        subset.append(target)
        subset.append(domain)
        subsets.append(FeatureList(subset))
    return subsets