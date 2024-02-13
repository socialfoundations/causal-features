![tableshift logo](img/tableshift.png)

# Predictors from causal features do not generalize better to new domains

This is code to reproduce experiments in the paper:

> Predictors from causal features do not generalize better to new domains

![logo](experiments_causal/dag_diabetes.svg)

Our code is build on TableShift and code from [Kim & Hardt (2023)](https://doi.org/10.1145/3617694.3623225). You can read more about TableShift at [tableshift.org](https://tableshift.org/index.html) or read the full paper (published in NeurIPS 2023 Datasets & Benchmarks Track) on [arxiv](https://arxiv.org/abs/2312.07577).

## Quickstart
We adapt the setup for a conda environment provided by TableShift.
Simply clone the repo, enter the root directory and create a local execution environment for TableShift.

```bash
git clone https://github.com/socialfoundations/causal-features.git
# set up the environment
conda env create -f environment.yml
```

Run the following commands to test the local execution environment:
```bash
conda env create -f environment.yml
conda activate tableshift
# test the install by running the training script
python examples/run_expt.py
```
The final line above will print some detailed logging output as the script executes. When you see `training completed! test accuracy: 0.6221` your environment is ready to go! (Accuracy may vary slightly due to randomness.)

## Dataset Availability

The datasets we use in our paper are either publicly available, or provide open credentialized access.
The datasets with open credentialized access require signing a data use agreement. For the tasks `ICU Mortality` and `ICU Length of Stay`, it is required to complete  training CITI Data or Specimens Only Research, as they contain sensitive personal information.
Hence, these datasets must be manually fetched and stored locally.

A list of datasets, their names in our code, and the corresponding access levels are below. The string identifier is the value that should be passed as the `experiment` parameter to the `--experiment` flag of `experiments_causal/run_experiment.py`.
The causal, arguably causal, and anti-causal feature sets are obtained by appending `_causal`, `_arguablycausal` and `_anticausal` to the string identifier.


| Tasks                 | String Identifier         | Availability                                                                                                 | Source                                                                                                                 | Preprocessing |
|-------------------------|---------------------------|--------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|-------------|
| Voting                  | `anes`                    | Public Credentialized Access ([source](https://electionstudies.org))                                         | [American National Election Studies (ANES)](https://electionstudies.org)                                               | TableShift |
| ASSISTments             | `assistments`             | Public                                                                                                       | [Kaggle](https://www.kaggle.com/datasets/nicolaswattiez/skillbuilder-data-2009-2010)                                   | TableShift |
| Childhood Lead          | `nhanes_lead`             | Public                                                                                                       | [National Health and Nutrition Examination Survey (NHANES)](https://www.cdc.gov/nchs/nhanes/index.htm)                 | TableShift |
| College Scorecard       | `college_scorecard`       | Public                                                                                                       | [College Scorecard](http://collegescorecard.ed.gov)                                                                    | TableShift |
| Diabetes                | `brfss_diabetes`          | Public                                                                                                       | [Behavioral Risk Factor Surveillance System (BRFSS)](https://www.cdc.gov/brfss/index.html)                             | TableShift |
| Food Stamps             | `acsfoodstamps`           | Public                                                                                                       | [American Community Survey](https://www.census.gov/programs-surveys/acs) (via [folktables](http://folktables.org))      |                                        | TableShift |
| Hospital Readmission    | `diabetes_readmission`    | Public                                                                                                       | [UCI](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)                           | TableShift |
| Hypertension            | `brfss_blood_pressure`    | Public                                                                                                       | [Behavioral Risk Factor Surveillance System (BRFSS)](https://www.cdc.gov/brfss/index.html)                             | TableShift |
| ICU Length of Stay      | `mimic_extract_los_3`     | Public Credentialized Access ([source](https://mimic.mit.edu/docs/gettingstarted/))                          | [MIMIC-iii](https://physionet.org/content/mimiciii/) via [MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract) | TableShift |
| ICU Mortality           | `mimic_extract_mort_hosp` | Public Credentialized Access ([source](https://mimic.mit.edu/docs/gettingstarted/))                          | [MIMIC-iii](https://physionet.org/content/mimiciii/) via [MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract) | TableShift |
| Income                  | `acsincome`               | Public                                                                                                       | [American Community Survey](https://www.census.gov/programs-surveys/acs) (via [folktables](http://folktables.org))      | TableShift |
| Public Health Insurance | `acspubcov`               | Public                                                                                                       | [American Community Survey](https://www.census.gov/programs-surveys/acs) (via [folktables](http://folktables.org))      | TableShift |
| Sepsis                  | `physionet`               | Public                                                                                                       | [Physionet](https://physionet.org/content/challenge-2019/)                                                             | TableShift |
| Unemployment            | `acsunemployment`         | Public                                                                                                       | [American Community Survey](https://www.census.gov/programs-surveys/acs) (via [folktables](http://folktables.org))       | TableShift |
| Utilization             | `meps`                    | Public ([source](https://meps.ahrq.gov/mepsweb/data_files/pufs/h216/h216dta.zip))                             | [Medical expenditure panel survey](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-216) |[Kim & Hardt (2023)](https://doi.org/10.1145/3617694.3623225) |
| Poverty                 | `sipp`                    | Public ([source](https://www2.census.gov/programs-surveys/sipp/data/datasets/2014/w1/pu2014w1_v13.dta.gz), [source](https://www2.census.gov/programs-surveys/sipp/data/datasets/2014/w2/pu2014w2_v13.dta.gz)) | [Survey of income and program participation](https://www.census.gov/sipp/)|[Kim & Hardt (2023)](https://doi.org/10.1145/3617694.3623225)|

TableShift includes the preprocessing of the data files in their implementation. For the tasks `Utilization` and `Poverty`, follow the instructions provided by [Kim & Hardt (2023)](https://doi.org/10.1145/3617694.3623225) in `backward_predictor/README.md`.

## Reproducing the experiments in the paper

The training script we run is located at `experiments_causal/run_experiment.py`.
It takes the following arguments:
* `experiment` (experiment to run)
* `model` (model to use)
* `cache_dir` (directory to cache raw data files to)
* `save_dir` (directory to save result files to)
  
The full list of model names is given below. For more details on each algorithm, see TableShift.
| Model                 | Name in TableShift |
|-----------------------|--------------------|
| XGBoost               | `xgb`              |
| LightGBM              | `lightgbm`         |
| SAINT                 | `saint`            |
| NODE                  | `node`             |
| Group DRO             | `group_dro`        |
| MLP                   | `mlp`              |
| Tabular ResNet        | `resnet`           |
| Adversarial Label DRO | `aldro`            |
| CORAL                 | `deepcoral`        |
| MMD                   | `mmd`              | 
| DRO                   | `dro`              |
| DANN                  | `dann`             | 
| TabTransformer        | `tabtransformer`   |
| MixUp                 | `mixup`            |
| Label Group DRO       | `label_group_dro`  |
| IRM                   | `irm`              |
| VREX                  | `vrex`             |
| FT-Transformer        | `ft_transformer`   |

All experiments were run as jobs submitted to a centralized cluster, running the open-source HTCondor scheduler.
The relevant script launching the jobs is located at `experiments_causal/launch_experiments.py`.

## Raw results of experiments
We provide the raw results of our experiments in the folder `experiments_causal/results/`. They contain a single `json` file for each task, feature selection and trained model.

## Reproducing the figures in the paper
Use the following Python scripts:
- Main result:
  - Figure in introduction: `experiments_causal/plot_paper_introduction_figure.py`
  - Figures in section "Empirical results": `experiments_causal/plot_paper_figures.py`
- Appendix: `experiments_causal/plot_paper_appendix_figures.py`, `experiments_causal/plot_paper_appendix_figures_extra.py`, and `experiments_causal/plot_paper_appendix_figures_extra2.py`
## Differences to TableShift
We list in the following which files/folders we changed for our experiments:
- created folder `experiments_causal` with python scripts to run experiments, launch experiments on a cluster, and plot figures for the paper
- created folder `backward_prediction` with preprocessing files adapted from [Kim & Hardt (2023)](https://doi.org/10.1145/3617694.3623225)
- added tasks `meps` and `sipp`, as well as causal feature selections of all tasks in their respective Python scripts in the folder `tableshift/datasets`
- added data source for `meps` and `sipp` in `tableshift/core/data_source.py`
- added tasks `meps` and `sipp`, as well as causal feature selection of all tasks in `tableshift/core/tasks.py`
- added configurations for tasks and their causal feature selections in `tableshift/configs/non_benchmark_configs.py`
- added computation of balanced accuracy in `tableshift/models/torchutils.py` and adapted `tableshift/models/compat.py` accordingly
- minor fixes in `tableshift/core/features.py`, `tableshift/core/tabular_dataset.py` and `tableshift/models/training.py`
- added the packages `paretoset==1.2.3` and `seaborn==0.13.0` in `requirements.txt`