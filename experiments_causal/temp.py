"""Python script to run experiment and record the performance."""
import argparse
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score
import json
from statsmodels.stats.proportion import proportion_confint

from tableshift import get_dataset
from tableshift.models.training import train
from tableshift.models.utils import get_estimator
from tableshift.models.default_hparams import get_default_config

cache_dir = "/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/icml-causal-features/causal-features/tmp"
cache_dir = Path(cache_dir)

experiment = "acsincome"

dset = get_dataset(experiment, cache_dir)

# Case: non-pytorch estimator; perform test-split evaluation.

test_split = "id_test"
# Fetch predictions and labels for a sklearn model.
X_te, y_te, _, _ = dset.get_pandas(test_split)