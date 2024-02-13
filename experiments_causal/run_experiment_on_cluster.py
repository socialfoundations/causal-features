"""Python script to run experiment and record the performance."""
import argparse
import logging
from pathlib import Path
import pickle
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
import json
from statsmodels.stats.proportion import proportion_confint

from tableshift.models.training import train
from tableshift.models.utils import get_estimator
from tableshift.models.default_hparams import get_default_config
from experiments_causal.metrics import balanced_accuracy_score


def main(
        experiment: str,
        model: str,
        cache_dir: str,
        save_dir: str,
        trial: str | int,
        debug: bool):
    """Run the experiment with the specified model.

    Parameters
    ----------
    experiment : str
        The name of the experiment to run.
    model : str
        The name of the model to train.
    cache_dir : str
        Directory to the preprocessed data.
    save_dir : str
        Directory to save result files to.
    trial : str | int
        Number of trial.
    debug : bool
        Debug mode.

    Returns
    -------
    None.

    """
    cache_dir = Path(cache_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=False)

    with open(f"{str(cache_dir)}/{experiment}.pickle", "rb") as f:
        dset = pickle.load(f)

    if debug:
        print("[INFO] running in debug mode.")
        experiment = "_debug"

    config = get_default_config(model, dset)
    estimator = get_estimator(model, **config)
    estimator = train(estimator, dset, config=config)

    if not isinstance(estimator, torch.nn.Module):
        evaluation = {}
        # Case: non-pytorch estimator; perform test-split evaluation.
        test_splits = (
            ["id_test", "ood_test", "validation"] if dset.is_domain_split else ["test"]
        )
        for test_split in test_splits:
            # Fetch predictions and labels for a sklearn model.
            X_te, y_te, _, _ = dset.get_pandas(test_split)
            X_te = X_te.astype(float)
            yhat_te = estimator.predict(X_te)

            # Calculate accuracy
            acc = accuracy_score(y_true=y_te, y_pred=yhat_te)
            evaluation[test_split] = acc
            nobs = len(y_te)
            count = nobs * acc
            # beta : Clopper-Pearson interval based on Beta distribution
            acc_conf = proportion_confint(count, nobs, alpha=0.05, method="beta")
            evaluation[test_split + "_conf"] = acc_conf
            print(f"training completed! {test_split} accuracy: {acc:.4f}")

            # Calculate balanced accuracy
            balanced_acc, balanced_acc_se = balanced_accuracy_score(
                target=y_te, prediction=yhat_te
            )
            evaluation[test_split + "_balanced"] = balanced_acc
            balanced_acc_conf = (
                balanced_acc - 1.96 * balanced_acc_se,
                balanced_acc + 1.96 * balanced_acc_se,
            )
            evaluation[test_split + "_balanced" + "_conf"] = balanced_acc_conf
            print(
                f"training completed! {test_split} balanced accuracy: {balanced_acc:.4f}"
            )

        # Open a file in write mode
        SAVE_DIR_EXP = save_dir / experiment
        SAVE_DIR_EXP.mkdir(exist_ok=True)
        with open(f"{str(SAVE_DIR_EXP)}/{model}_eval_{trial}.json", "w") as f:
            # Use json.dump to write the dictionary into the file
            evaluation["features"] = dset.predictors
            json.dump(evaluation, f)

    else:
        # Case: pytorch estimator; eval is already performed + printed by train().
        print("training completed!")
        evaluation = estimator.fit_metrics
        evaluation_balanced = estimator.fit_metrics_balanced
        for test_split in ["id_test", "ood_test"]:
            # Get accuracy
            # Fetch predictions and labels for a sklearn model.
            X_te, y_te, _, _ = dset.get_pandas(test_split)
            X_te = X_te.astype(float)
            nobs = len(y_te)
            acc = evaluation[test_split]
            count = nobs * acc
            acc_conf = proportion_confint(count, nobs, alpha=0.05, method="beta")
            evaluation[test_split + "_conf"] = acc_conf

            # Get balanced accuracy
            balanced_acc = evaluation_balanced["score"][test_split]
            balanced_acc_se = evaluation_balanced["se"][test_split]
            evaluation[test_split + "_balanced"] = balanced_acc
            balanced_acc_conf = (
                balanced_acc - 1.96 * balanced_acc_se,
                balanced_acc + 1.96 * balanced_acc_se,
            )
            evaluation[test_split + "_balanced" + "_conf"] = balanced_acc_conf
            print(
                f"training completed! {test_split} balanced accuracy: {balanced_acc:.4f}"
            )

        # Open a file in write mode
        SAVE_DIR_EXP = save_dir / experiment
        SAVE_DIR_EXP.mkdir(exist_ok=True)
        with open(f"{str(SAVE_DIR_EXP)}/{model}_eval_{trial}.json", "w") as f:
            evaluation["features"] = dset.predictors
            json.dump(evaluation, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir", default="tmp", help="Directory to cache raw data files to."
    )
    parser.add_argument(
        "--save_dir", default="tmp", help="Directory to save result files to."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Whether to run in debug mode. If True, various "
        "truncations/simplifications are performed to "
        "speed up experiment.",
    )
    parser.add_argument(
        "--experiment",
        default="diabetes_readmission",
        help="Experiment to run. Overridden when debug=True.",
    )
    parser.add_argument("--model", default="histgbm", help="model to use.")
    parser.add_argument("--trial", default=0, help="Number of trial.")
    args = parser.parse_args()
    main(**vars(args))
