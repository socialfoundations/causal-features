"""Python script to preprocess data and launch condor jobs to train models and record the performance."""
import sys
import random
import dataclasses
from pathlib import Path
import numpy as np
import pandas as pd
import json
import argparse
import pickle
from time import sleep

from tableshift import get_dataset

if __name__ == "__main__":
    import htcondor
    import classad


####################################################
#  START of: details on which experiments to run.  #
####################################################
# List of models
MODELS = (
    "ft_transformer",
    # "histgbm",
    # "mlp",
    "saint",
    # "tabtransformer",
    # "resnet",
    "xgb",
    "lightgbm",
    # "aldro",
    # "dro",
    # "node",
    # "group_dro",
    # "label_group_dro",
)

# List of domain generalization models
DG_MODELS = (
    # "dann",
    # "irm",
    # "vrex",
    # "mixup",
    # "mmd",
    # "deepcoral",
    # "and_mask",
    # "ib_erm",
    # "ib_irm",
    # "causirl_coral",
    # "causirl_mmd"
)

# List of task that do not allow domain generalization
NOT_DG_TASKS = (
    "acspubcov",
    "physionet",
    "nhanes_lead",
    "brfss_blood_pressure",
    "brfss_diabetes",
    "sipp",
    "assistments",
    "meps",
)


def IS_TASK_DG(task: str) -> bool:
    """Classify whether a tasks allows domain generalization.


    Parameters
    ----------
    task : str
        The name of task to classify whether it allows domain generalization.

    Returns
    -------
    bool
        The task allows domain generalization if True.

    """
    for dg_task in NOT_DG_TASKS:
        if task.startswith(dg_task):
            return False
    return True


@dataclasses.dataclass
class ExperimentConfigs:
    name: str
    model: str
    job_memory_gb: int  # = JOB_MEMORY_GB

    n_trials: int  # = dic_args["N_TRIALS"]
    job_cpus: int  # = dic_args["JOB_CPUS"]
    job_bid: int  # = dic_args["JOB_MIN_BID"]
    job_gpus: int = 0

    # def __post_init__(self):
    #     self.job_bid = max(self.job_bid, dic_args["JOB_MIN_BID"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", help="Directory to load data files from.")
    parser.add_argument("--RESULTS_DIR", help="Directory to save result files to.")
    parser.add_argument(
        "--CLUSTER_LOGS_SAVE_DIR",
        help="Directory to save cluster logs and job stdout/stderr.",
    )
    parser.add_argument(
        "--N_TRIALS",
        default="1",
        help="Number of experiments to run per model, per dataset.",
    )
    parser.add_argument(
        "--JOB_CPUS", default="1", help="Number of CPUs per experiment (per cluster job)."
    )
    parser.add_argument("--JOB_MEMORY_GB", default="64", help="GBs of memory.")
    parser.add_argument("--JOB_MIN_BID", default="15", help="Htcondor bid.")
    parser.add_argument(
        "--task",
        default="diabetes_readmission",
        help="Task to run. Overridden when debug=True.",
    )
    args = parser.parse_args()

    dic_args = vars(args)

    DATA_DIR_PREPROCESSED = Path(dic_args["DATA_DIR"]) / "preprocessed"
    DATA_DIR_PREPROCESSED.mkdir(exist_ok=True)

    CLUSTER_LOGS_SAVE_ERR_DIR = (
        Path(dic_args["CLUSTER_LOGS_SAVE_DIR"]) / "error" / "experiments"
    )
    CLUSTER_LOGS_SAVE_ERR_DIR.mkdir(exist_ok=True)

    CLUSTER_LOGS_SAVE_OUT_DIR = (
        Path(dic_args["CLUSTER_LOGS_SAVE_DIR"]) / "output" / "experiments"
    )
    CLUSTER_LOGS_SAVE_OUT_DIR.mkdir(exist_ok=True)

    CLUSTER_LOGS_SAVE_LOG_DIR = (
        Path(dic_args["CLUSTER_LOGS_SAVE_DIR"]) / "logs" / "experiments"
    )
    CLUSTER_LOGS_SAVE_LOG_DIR.mkdir(exist_ok=True)

    # enforce min bid

    # Set up experiments for the task
    all_task_experiments = []

    if IS_TASK_DG(dic_args["task"]):
        TASK_MODELS = MODELS + DG_MODELS
    else:
        TASK_MODELS = MODELS

    job_memory_gb = dic_args["JOB_MEMORY_GB"]  # BIG_JOB_MEMORY_GB

    for model in TASK_MODELS:
        all_task_experiments.append(
            ExperimentConfigs(
                name=dic_args["task"],
                model=model,
                job_memory_gb=job_memory_gb,
                n_trials=int(dic_args["N_TRIALS"]),
                job_cpus=int(dic_args["JOB_CPUS"]),
                job_bid=int(dic_args["JOB_MIN_BID"]),
            )
        )

    # Process data for the task and save processed data
    dset = get_dataset(dic_args["task"], dic_args["DATA_DIR"])
    with open(f"{str(DATA_DIR_PREPROCESSED)}/{dic_args['task']}.pickle", "wb") as f:
        pickle.dump(dset, f)

    ##################################################
    #  END of: details on which experiments to run.  #
    ##################################################

    def launch_experiments_jobs(
        task: str,
        exp_obj: ExperimentConfigs,
    ):
        """Launches the cluster jobs to execute all `n_trials` of a given experiment.

        Parameters
        ----------
        task : str
            The name of the task/data to use.
        exp_obj : ExperimentConfigs
            The detailed configs to run an experiment.
        """

        # Name/prefix for cluster logs related to this job
        cluster_job_err_name = str(
            CLUSTER_LOGS_SAVE_ERR_DIR
            / f"{exp_obj.name}_{exp_obj.model}_$(Cluster).$(Process)"
        )

        cluster_job_out_name = str(
            CLUSTER_LOGS_SAVE_OUT_DIR
            / f"{exp_obj.name}_{exp_obj.model}_$(Cluster).$(Process)"
        )

        cluster_job_log_name = str(
            CLUSTER_LOGS_SAVE_LOG_DIR
            / f"{exp_obj.name}_{exp_obj.model}_$(Cluster).$(Process)"
        )

        EXP_RESULTS_DIR = Path(dic_args["RESULTS_DIR"])
        EXP_RESULTS_DIR.mkdir(exist_ok=True, parents=False)

        # Construct job description
        job_description = htcondor.Submit(
            {
                "executable": "/home/vnastl/miniconda3/envs/causalfeatures/bin/python3",
                # "arguments": "foo.py",    # NOTE: used for testing
                "arguments": (
                    "/home/vnastl/causal-features/experiments_causal/run_experiment_on_cluster.py "
                    f"--experiment {exp_obj.name} "
                    f"--model {exp_obj.model} "
                    f"--cache_dir {str(DATA_DIR_PREPROCESSED)} "
                    f"--save_dir {str(EXP_RESULTS_DIR)} "
                    f"--trial $(Process) "
                    # f"{'--verbose' if VERBOSE else ''} "
                ),
                "output": f"{cluster_job_out_name}.out",
                "error": f"{cluster_job_err_name}.err",
                "log": f"{cluster_job_log_name}.log",
                "request_cpus": f"{exp_obj.job_cpus}",
                "request_gpus": f"{exp_obj.job_gpus}",
                "request_memory": f"{exp_obj.job_memory_gb}GB",
                # "request_disk": "2GB",
                "jobprio": f"{exp_obj.job_bid - 1000}",
                "notification": "error",
                # "job_seed_macro": f"$(Process) + {random.randrange(int(1e9))}",
                # "job_seed": "$INT(job_seed_macro)",
                # Concurrency limits:
                # > each job uses this amount of resources out of a pool of 10k
                # "concurrency_limits": "user.theoremfivepointsix:10000",     # 1 job
                # "concurrency_limits": "user.theoremfivepointsix:100",     # 100 jobs in parallel
                "concurrency_limits": "user.theoremfivepointsix:10",  # 5000 jobs in parallel
                "+MaxRunningPrice": 100,
                # "+RunningPriceExceededAction": classad.quote("restart"),
                "periodic_remove": f"(JobStatus == 2) && (time() - EnteredCurrentStatus) > (6 * 3600)",
            }
        )

        # Submit `n_trials` jobs to the scheduler
        schedd = htcondor.Schedd()
        submit_result = schedd.submit(job_description, count=exp_obj.n_trials)

    print(
        f"\n*** *** ***\n"
        f"Launching {len(all_task_experiments)} * {dic_args['N_TRIALS']} = "
        f"{dic_args['N_TRIALS'] * len(all_task_experiments)} "
        f"experiments for task={dic_args['task']}"
        f"\n*** *** ***\n"
    )

    for i, exp_obj in enumerate(all_task_experiments):
        print(f"{i}. Launching {exp_obj.n_trials} trials for the experiment '{exp_obj}'")
        success = False
        while not success:
            try:
                launch_experiments_jobs(task=dic_args["task"], exp_obj=exp_obj)
                success = True
            except:
                sleep(600)
