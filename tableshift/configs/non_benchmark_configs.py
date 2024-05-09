"""
Experiment configs for the non-TableShift benchmark tasks, `Utilization` task, `Poverty` task and causal feature
selections specified in 'Predictors from Causal Features Do Not Generalize Better to New Domains'.


Modified for 'Predictors from Causal Features Do Not Generalize Better to New Domains'.
"""

from tableshift.configs.benchmark_configs import \
    _MIMIC_EXTRACT_PASSTHROUGH_COLUMNS
from tableshift.configs.experiment_config import ExperimentConfig
from tableshift.configs.experiment_defaults import DEFAULT_ID_TEST_SIZE, \
    DEFAULT_OOD_VAL_SIZE, DEFAULT_ID_VAL_SIZE, DEFAULT_RANDOM_STATE
from tableshift.core import RandomSplitter, Grouper, PreprocessorConfig, \
    DomainSplitter, FixedSplitter
from tableshift.datasets import *

GRINSTAJN_TEST_SIZE = 0.21
GRINSZTAJN_VAL_SIZE = 0.09

NON_BENCHMARK_CONFIGS = {
    "adult": ExperimentConfig(
        splitter=FixedSplitter(val_size=0.25, random_state=29746),
        grouper=Grouper({"Race": ["White", ], "Sex": ["Male", ]}, drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    "_debug": ExperimentConfig(
        splitter=DomainSplitter(
            val_size=0.01,
            id_test_size=0.2,
            ood_val_size=0.25,
            random_state=43406,
            domain_split_varname="purpose",
            # Counts by domain are below. We hold out all of the smallest
            # domains to avoid errors with very small domains during dev.
            # A48       9
            # A44      12
            # A410     12
            # A45      22
            # A46      50
            # A49      97
            # A41     103
            # A42     181
            # A40     234
            # A43     280
            domain_split_ood_values=["A44", "A410", "A45", "A46", "A48"]
        ),
        grouper=Grouper({"sex": ['1.0', ], "age_geq_median": ['1.0', ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"name": "german"}),

    "german": ExperimentConfig(
        splitter=RandomSplitter(val_size=0.01, test_size=0.2, random_state=832),
        grouper=Grouper({"sex": ['1.0', ], "age_geq_median": ['1.0', ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    "mooc": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="course_id",
                                domain_split_ood_values=[
                                    "HarvardX/CB22x/2013_Spring"]),
        grouper=Grouper({"gender": ["m", ],
                         "LoE_DI": ["Bachelor's", "Master's", "Doctorate"]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    ################### Grinsztajn et al. benchmark datasets ###################

    **{x: ExperimentConfig(
        splitter=RandomSplitter(val_size=GRINSZTAJN_VAL_SIZE,
                                test_size=GRINSTAJN_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"dataset_name": x}
    ) for x in ("electricity", "bank-marketing", "california",
                "covertype", "credit", 'default-of-credit-card-clients',
                'eye_movements', 'Higgs', 'MagicTelescope', 'MiniBooNE',
                'road-safety', 'pol', 'jannis', 'house_16H')},

    ################### MetaMIMIC datasets #####################################

    "metamimic_alcohol": ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_alcohol'}),

    'metamimic_anemia': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_anemia'}),

    'metamimic_atrial': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_atrial'}),

    'metamimic_diabetes': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_diabetes'}),

    'metamimic_heart': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_heart'}),

    'metamimic_hypertension': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_hypertension'}),

    'metamimic_hypotension': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_hypotension'}),

    'metamimic_ischematic': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_ischematic'}),

    'metamimic_lipoid': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_lipoid'}),

    'metamimic_overweight': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_overweight'}),

    'metamimic_purpura': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_purpura'}),

    'metamimic_respiratory': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_respiratory'}),

    ################### CatBoost benchmark datasets ########################

    "amazon": ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={
            "kaggle_dataset_name": "amazon-employee-access-challenge"}),

    **{k: ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        # categorical features in this dataset have *extremely* high cardinality
        preprocessor_config=PreprocessorConfig(
            categorical_features="passthrough"),
        tabular_dataset_kwargs={"task_name": k}) for k in
        ("appetency", "churn", "upselling")},

    "click": ExperimentConfig(
        splitter=FixedSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                               random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        # categorical features in this dataset have *extremely* high cardinality
        preprocessor_config=PreprocessorConfig(
            categorical_features="passthrough"),
        tabular_dataset_kwargs={
            "kaggle_dataset_name": "kddcup2012-track2"}),

    'kick': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            categorical_features="passthrough"),
        tabular_dataset_kwargs={"kaggle_dataset_name": "DontGetKicked"},
    ),

    ############# AutoML benchmark datasets (classification only) ##############
    **{x: ExperimentConfig(
        splitter=FixedSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                               random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            categorical_features="passthrough",
            dropna=None),
        tabular_dataset_kwargs={
            "automl_benchmark_dataset_name": x}) for x in (
        'product_sentiment_machine_hack', 'data_scientist_salary',
        'melbourne_airbnb', 'news_channel', 'wine_reviews',
        'imdb_genre_prediction', 'fake_job_postings2', 'kick_starter_funding',
        'jigsaw_unintended_bias100K',)},

    ######################## UCI datasets ######################################
    **{x: ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={}
    ) for x in ('iris', 'dry-bean', 'heart-disease', 'wine', 'wine-quality',
                'rice', 'cars', 'raisin', 'abalone')},

    # For breast cancer, mean/stc/worst values are already computed as features,
    # so we passthrough by default.
    'breast-cancer': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(numeric_features="passthrough"),
        tabular_dataset_kwargs={}),
    ######################## Kaggle datasets ###################################
    **{x: ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            categorical_features="passthrough", dropna=None),
        tabular_dataset_kwargs={}
    ) for x in ('otto-products', 'sf-crime', 'plasticc', 'walmart',
                'tradeshift', 'schizophrenia', 'titanic',
                'santander-transactions', 'home-credit-default-risk',
                'ieee-fraud-detection', 'safe-driver-prediction',
                'santander-customer-satisfaction', 'amex-default',
                'ad-fraud')},

    ############################################################################

    "mimic_extract_los_3_selected": ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS),
        tabular_dataset_kwargs={"task": "los_3",
                                "name": "mimic_extract_los_3_selected"}),

    "mimic_extract_mort_hosp_selected": ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS),
        tabular_dataset_kwargs={"task": "mort_hosp",
                                "name": "mimic_extract_mort_hosp_selected"}),

    "communities_and_crime": ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    "compas": ExperimentConfig(
        splitter=RandomSplitter(test_size=0.2, val_size=0.01,
                                random_state=90127),
        grouper=Grouper({"race": ["Caucasian", ], "sex": ["Male", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),
}

################################################################################
# Configuration for causal, arguably causal and (if applicable) anticausal features
################################################################################


# We passthrough all non-static columns because we use
# MIMIC-extract's default preprocessing/imputation and do not
# wish to modify it for these features
# (static features are not preprocessed by MIMIC-extract). See
# tableshift.datasets.mimic_extract.preprocess_mimic_extract().
_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS = [
    f for f in MIMIC_EXTRACT_SHARED_FEATURES.names
    if f not in MIMIC_EXTRACT_STATIC_FEATURES.names]

CAUSAL_BENCHMARK_CONFIGS = {
    "meps": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="INSCOV19",
                                domain_split_ood_values=[0]),
        grouper=Grouper({"SEX": ['1.0', ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={}),
    "meps_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="INSCOV19",
                                domain_split_ood_values=[0]),
        grouper=Grouper({"SEX": ['1.0', ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={}),
    "meps_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="INSCOV19",
                                domain_split_ood_values=[0]),
        grouper=Grouper({"SEX": ['1.0', ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={}),

    "sipp": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="CITIZENSHIP_STATUS",
                                domain_split_ood_values=['1.0']),
        grouper=Grouper({"GENDER": ['1.0', ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={}),
    "sipp_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="CITIZENSHIP_STATUS",
                                domain_split_ood_values=['1.0']),
        grouper=Grouper({"GENDER": ['1.0', ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={}),
    "sipp_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="CITIZENSHIP_STATUS",
                                domain_split_ood_values=['1.0']),
        grouper=Grouper({"GENDER": ['1.0', ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={}),
    "sipp_anticausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="CITIZENSHIP_STATUS",
                                domain_split_ood_values=['1.0']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={}),
    "acsfoodstamps_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['06']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsfoodstamps"}),
    "acsfoodstamps_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['06']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsfoodstamps"}),

    "acsincome_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['01']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsincome"}),
    "acsincome_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['01']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsincome"}),
    "acsincome_anticausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['01']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsincome"}),

    "acspubcov_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIS",
                                domain_split_ood_values=['1.0']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acspubcov", "name": "acspubcov_causal",
                                "years": ACS_YEARS}),
    "acspubcov_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIS",
                                domain_split_ood_values=['1.0']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acspubcov", "name": "acspubcov_arguablycausal",
                                "years": ACS_YEARS}),

    "acsunemployment_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='SCHL',
                                # No high school diploma vs. GED/diploma or higher.
                                domain_split_ood_values=['01', '02', '03', '04',
                                                         '05', '06', '07', '08',
                                                         '09', '10', '11', '12',
                                                         '13', '14', '15']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsunemployment"}),
    "acsunemployment_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='SCHL',
                                # No high school diploma vs. GED/diploma or higher.
                                domain_split_ood_values=['01', '02', '03', '04',
                                                         '05', '06', '07', '08',
                                                         '09', '10', '11', '12',
                                                         '13', '14', '15']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsunemployment"}),
    "acsunemployment_anticausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='SCHL',
                                # No high school diploma vs. GED/diploma or higher.
                                domain_split_ood_values=['01', '02', '03', '04',
                                                         '05', '06', '07', '08',
                                                         '09', '10', '11', '12',
                                                         '13', '14', '15']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsunemployment"}),

    "anes_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='VCF0112',
                                domain_split_ood_values=['3.0']),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={}),
    "anes_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='VCF0112',
                                domain_split_ood_values=['3.0']),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={}),

    "brfss_blood_pressure_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="BMI5CAT",
                                # OOD values: [1 underweight, 2 normal weight], [3 overweight, 4 obese]
                                domain_split_ood_values=['3.0', '4.0']),
        grouper=Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_blood_pressure_causal",
                                "task": "blood_pressure",
                                "years": BRFSS_YEARS},
    ),
    "brfss_blood_pressure_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="BMI5CAT",
                                # OOD values: [1 underweight, 2 normal weight], [3 overweight, 4 obese]
                                domain_split_ood_values=['3.0', '4.0']),
        grouper=Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_blood_pressure_arguablycausal",
                                "task": "blood_pressure",
                                "years": BRFSS_YEARS},
    ),
    "brfss_blood_pressure_anticausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="BMI5CAT",
                                # OOD values: [1 underweight, 2 normal weight], [3 overweight, 4 obese]
                                domain_split_ood_values=['3.0', '4.0']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_blood_pressure_anticausal",
                                "task": "blood_pressure",
                                "years": BRFSS_YEARS},
    ),

    "brfss_diabetes_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="PRACE1",
                                domain_split_ood_values=[2, 3, 4, 5, 6],
                                domain_split_id_values=[1, ]),
        grouper=Grouper({"SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_diabetes_causal",
                                "task": "diabetes", "years": BRFSS_YEARS},
    ),
    "brfss_diabetes_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="PRACE1",
                                domain_split_ood_values=[2, 3, 4, 5, 6],
                                domain_split_id_values=[1, ]),
        grouper=Grouper({"SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_diabetes_arguablycausal",
                                "task": "diabetes", "years": BRFSS_YEARS},
    ),
    "brfss_diabetes_anticausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="PRACE1",
                                domain_split_ood_values=[2, 3, 4, 5, 6],
                                domain_split_id_values=[1, ]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_diabetes_anticausal",
                                "task": "diabetes", "years": BRFSS_YEARS},
    ),

    "diabetes_readmission_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='admission_source_id',
                                domain_split_ood_values=[7, ]),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"race": ["Caucasian", ], "gender": ["Male", ]},
                        drop=False),
        # Note: using min_frequency=0.01 reduces data
        # dimensionality from ~2400 -> 169 columns.
        # This is due to high cardinality of 'diag_*' features.
        preprocessor_config=PreprocessorConfig(min_frequency=0.01),
        tabular_dataset_kwargs={}),
    "diabetes_readmission_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='admission_source_id',
                                domain_split_ood_values=[7, ]),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"race": ["Caucasian", ], "gender": ["Male", ]},
                        drop=False),
        # Note: using min_frequency=0.01 reduces data
        # dimensionality from ~2400 -> 169 columns.
        # This is due to high cardinality of 'diag_*' features.
        preprocessor_config=PreprocessorConfig(min_frequency=0.01),
        tabular_dataset_kwargs={}),

    "mimic_extract_los_3_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare"]),

        grouper=Grouper({"gender": ['M'], }, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[f for f in _MIMIC_EXTRACT_PASSTHROUGH_COLUMNS
                                 if f in MIMIC_EXTRACT_LOS_3_FEATURES_CAUSAL.names]),
        tabular_dataset_kwargs={"task": "los_3",
                                "name": "mimic_extract_los_3_causal"}),
    "mimic_extract_los_3_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare"]),

        grouper=Grouper({"gender": ['M'], }, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[f for f in _MIMIC_EXTRACT_PASSTHROUGH_COLUMNS
                                 if f in MIMIC_EXTRACT_LOS_3_FEATURES_ARGUABLYCAUSAL.names]),
        tabular_dataset_kwargs={"task": "los_3",
                                "name": "mimic_extract_los_3_arguablycausal"}),

    "mimic_extract_mort_hosp_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare",
                                                         "Medicaid"]),
        grouper=Grouper({"gender": ['M'], }, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[f for f in _MIMIC_EXTRACT_PASSTHROUGH_COLUMNS
                                 if f in MIMIC_EXTRACT_MORT_HOSP_FEATURES_CAUSAL.names]),
        tabular_dataset_kwargs={"task": "mort_hosp",
                                "name": "mimic_extract_mort_hosp_causal"}),
    "mimic_extract_mort_hosp_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare",
                                                         "Medicaid"]),
        grouper=Grouper({"gender": ['M'], }, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[f for f in _MIMIC_EXTRACT_PASSTHROUGH_COLUMNS
                                 if f in MIMIC_EXTRACT_MORT_HOSP_FEATURES_ARGUABLYCAUSAL.names]),
        tabular_dataset_kwargs={"task": "mort_hosp",
                                "name": "mimic_extract_mort_hosp_arguablycausal"}),
    "assistments_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='school_id',
                                domain_split_ood_values=[5040.0,
                                                         11502.0,
                                                         11318.0,
                                                         11976.0,
                                                         12421.0,
                                                         12379.0,
                                                         11791.0,
                                                         8359.0,
                                                         12406.0,
                                                         7594.0]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["skill_id", "bottom_hint", "first_action"],
        ),
        tabular_dataset_kwargs={},
    ),
    "assistments_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='school_id',
                                domain_split_ood_values=[5040.0,
                                                         11502.0,
                                                         11318.0,
                                                         11976.0,
                                                         12421.0,
                                                         12379.0,
                                                         11791.0,
                                                         8359.0,
                                                         12406.0,
                                                         7594.0]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["skill_id", "bottom_hint", "first_action"],
        ),
        tabular_dataset_kwargs={},
    ),

    "college_scorecard_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='CCBASIC',
                                domain_split_ood_values=[
                                    'Special Focus Institutions--Other special-focus institutions',
                                    'Special Focus Institutions--Theological seminaries, Bible colleges, and other faith-related institutions',
                                    "Associate's--Private For-profit 4-year Primarily Associate's",
                                    'Baccalaureate Colleges--Diverse Fields',
                                    'Special Focus Institutions--Schools of art, music, and design',
                                    "Associate's--Private Not-for-profit",
                                    "Baccalaureate/Associate's Colleges",
                                    "Master's Colleges and Universities (larger programs)"]
                                ),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            # Several categorical features in college scorecard have > 10k
            # unique values; so we label-encode instead of one-hot encoding.
            categorical_features="label_encode",
            # Some important numeric features are not reported by universities
            # in a way that could be systematic (and we would like these included
            # in the sample, not excluded), so we use kbins
            numeric_features="kbins",
            n_bins=100,
            dropna=None,
        ),
        tabular_dataset_kwargs={},
    ),
    "college_scorecard_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='CCBASIC',
                                domain_split_ood_values=[
                                    'Special Focus Institutions--Other special-focus institutions',
                                    'Special Focus Institutions--Theological seminaries, Bible colleges, and other faith-related institutions',
                                    "Associate's--Private For-profit 4-year Primarily Associate's",
                                    'Baccalaureate Colleges--Diverse Fields',
                                    'Special Focus Institutions--Schools of art, music, and design',
                                    "Associate's--Private Not-for-profit",
                                    "Baccalaureate/Associate's Colleges",
                                    "Master's Colleges and Universities (larger programs)"]
                                ),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            # Several categorical features in college scorecard have > 10k
            # unique values; so we label-encode instead of one-hot encoding.
            categorical_features="label_encode",
            # Some important numeric features are not reported by universities
            # in a way that could be systematic (and we would like these included
            # in the sample, not excluded), so we use kbins
            numeric_features="kbins",
            n_bins=100,
            dropna=None,
        ),
        tabular_dataset_kwargs={},
    ),

    "nhanes_lead_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='INDFMPIRBelowCutoff',
                                domain_split_ood_values=[1.]),
        # Race (non. hispanic white vs. all others; male vs. all others)
        grouper=Grouper({"RIDRETH_merged": [3, ], "RIAGENDR": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
        tabular_dataset_kwargs={"nhanes_task": "lead", "years": NHANES_YEARS}),
    "nhanes_lead_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='INDFMPIRBelowCutoff',
                                domain_split_ood_values=[1.]),
        # Race (non. hispanic white vs. all others; male vs. all others)
        grouper=Grouper({"RIDRETH_merged": [3, ], "RIAGENDR": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
        tabular_dataset_kwargs={"nhanes_task": "lead", "years": NHANES_YEARS}),

    "physionet_causal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='ICULOS',
                                domain_split_gt_thresh=47.0),
        grouper=None,
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={"name": "physionet_causal"}),
    "physionet_arguablycausal": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='ICULOS',
                                domain_split_gt_thresh=47.0),
        grouper=None,
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={"name": "physionet_arguablycausal"}),
}

NON_BENCHMARK_CONFIGS.update(CAUSAL_BENCHMARK_CONFIGS)

################################################################################
# Configuration for robustness tests
################################################################################

for index in range(SIPP_FEATURES_CAUSAL_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["sipp_causal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="CITIZENSHIP_STATUS",
                                domain_split_ood_values=['1.0']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={})

for index in range(SIPP_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["sipp_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="CITIZENSHIP_STATUS",
                                domain_split_ood_values=['1.0']),
        grouper=Grouper({"GENDER": ['1.0', ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={})

for index in range(MEPS_FEATURES_CAUSAL_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["meps_causal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="INSCOV19",
                                domain_split_ood_values=[0]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={})

for index in range(MEPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["meps_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="INSCOV19",
                                domain_split_ood_values=[0]),
        grouper=Grouper({"SEX": ['1.0', ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={})


for index in range(ACS_INCOME_FEATURES_CAUSAL_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["acsincome_causal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['01']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsincome"})

for index in range(ACS_INCOME_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["acsincome_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['01']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsincome"})

for index in range(ACS_FOODSTAMPS_FEATURES_CAUSAL_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["acsfoodstamps_causal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['06']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsfoodstamps"})

for index in range(ACS_FOODSTAMPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["acsfoodstamps_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['06']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsfoodstamps"})
    
for index in range(ACS_PUBCOV_FEATURES_CAUSAL_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["acspubcov_causal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIS",
                                domain_split_ood_values=['1.0']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acspubcov", "name": "acspubcov_causal_test_"+f"{index}",
                                "years": ACS_YEARS})

for index in range(ACS_PUBCOV_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["acspubcov_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIS",
                                domain_split_ood_values=['1.0']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acspubcov", "name": "acspubcov_arguablycausal_test_"+f"{index}",
                                "years": ACS_YEARS})

for index in range(ACS_UNEMPLOYMENT_FEATURES_CAUSAL_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["acsunemployment_causal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='SCHL',
                                # No high school diploma vs. GED/diploma or higher.
                                domain_split_ood_values=['01', '02', '03', '04',
                                                         '05', '06', '07', '08',
                                                         '09', '10', '11', '12',
                                                         '13', '14', '15']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsunemployment"})

for index in range(ACS_UNEMPLOYMENT_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["acsunemployment_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='SCHL',
                                # No high school diploma vs. GED/diploma or higher.
                                domain_split_ood_values=['01', '02', '03', '04',
                                                         '05', '06', '07', '08',
                                                         '09', '10', '11', '12',
                                                         '13', '14', '15']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsunemployment"})


for index in range(BRFSS_DIABETES_FEATURES_CAUSAL_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["brfss_diabetes_causal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="PRACE1",
                                domain_split_ood_values=[
                                    2, 3, 4, 5, 6],
                                domain_split_id_values=[1, ]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_diabetes_causal_test_"+f"{index}",
                                "task": "diabetes", "years": BRFSS_YEARS})
for index in range(BRFSS_DIABETES_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["brfss_diabetes_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="PRACE1",
                                domain_split_ood_values=[
                                    2, 3, 4, 5, 6],
                                domain_split_id_values=[1, ]),
        grouper=Grouper({"SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_diabetes_arguablycausal_test_"+f"{index}",
                                "task": "diabetes", "years": BRFSS_YEARS})

for index in range(BRFSS_BLOOD_PRESSURE_FEATURES_CAUSAL_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["brfss_blood_pressure_causal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="BMI5CAT",
                                # OOD values: [1 underweight, 2 normal weight], [3 overweight, 4 obese]
                                domain_split_ood_values=['3.0', '4.0']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_blood_pressure_causal_test_"+f"{index}",
                                "task": "blood_pressure",
                                "years": BRFSS_YEARS},
    )
for index in range(BRFSS_BLOOD_PRESSURE_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["brfss_blood_pressure_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="BMI5CAT",
                                # OOD values: [1 underweight, 2 normal weight], [3 overweight, 4 obese]
                                domain_split_ood_values=['3.0', '4.0']),
        grouper=Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_blood_pressure_arguablycausal_test_"+f"{index}",
                                "task": "blood_pressure",
                                "years": BRFSS_YEARS},
    )

for index in range(DIABETES_READMISSION_FEATURES_CAUSAL_NUMBER):
    NON_BENCHMARK_CONFIGS["diabetes_readmission_causal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='admission_source_id',
                                domain_split_ood_values=[7, ]),
        # male vs. all others; white non-hispanic vs. others
        grouper=None,
        # Note: using min_frequency=0.01 reduces data
        # dimensionality from ~2400 -> 169 columns.
        # This is due to high cardinality of 'diag_*' features.
        preprocessor_config=PreprocessorConfig(
            min_frequency=0.01),
        tabular_dataset_kwargs={})
for index in range(DIABETES_READMISSION_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["diabetes_readmission_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='admission_source_id',
                                domain_split_ood_values=[7, ]),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"race": ["Caucasian", ], "gender": ["Male", ]},
                        drop=False),
        # Note: using min_frequency=0.01 reduces data
        # dimensionality from ~2400 -> 169 columns.
        # This is due to high cardinality of 'diag_*' features.
        preprocessor_config=PreprocessorConfig(
            min_frequency=0.01),
        tabular_dataset_kwargs={})
for index in range(ANES_FEATURES_CAUSAL_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["anes_causal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='VCF0112',
                                domain_split_ood_values=['3.0']),
        # male vs. all others; white non-hispanic vs. others
        grouper=None,
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={})
for index in range(ANES_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["anes_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='VCF0112',
                                domain_split_ood_values=['3.0']),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={})

for index in range(ASSISTMENTS_FEATURES_CAUSAL_SUBSETS_NUMBER):
    list_passthrough = [feature.name for feature in ASSISTMENTS_FEATURES_CAUSAL_SUBSETS[index]
                        if feature.name in ["skill_id", "bottom_hint", "first_action"]]
    NON_BENCHMARK_CONFIGS["assistments_causal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='school_id',
                                domain_split_ood_values=[5040.0,
                                                         11502.0,
                                                         11318.0,
                                                         11976.0,
                                                         12421.0,
                                                         12379.0,
                                                         11791.0,
                                                         8359.0,
                                                         12406.0,
                                                         7594.0]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=list_passthrough,
        ),
        tabular_dataset_kwargs={},
    )
for index in range(ASSISTMENTS_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["assistments_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='school_id',
                                domain_split_ood_values=[5040.0,
                                                         11502.0,
                                                         11318.0,
                                                         11976.0,
                                                         12421.0,
                                                         12379.0,
                                                         11791.0,
                                                         8359.0,
                                                         12406.0,
                                                         7594.0]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[
                "skill_id", "bottom_hint", "first_action"],
        ),
        tabular_dataset_kwargs={},
    )


for index in range(COLLEGE_SCORECARD_FEATURES_CAUSAL_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["college_scorecard_causal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='CCBASIC',
                                domain_split_ood_values=[
                                    'Special Focus Institutions--Other special-focus institutions',
                                    'Special Focus Institutions--Theological seminaries, Bible colleges, and other faith-related institutions',
                                    "Associate's--Private For-profit 4-year Primarily Associate's",
                                    'Baccalaureate Colleges--Diverse Fields',
                                    'Special Focus Institutions--Schools of art, music, and design',
                                    "Associate's--Private Not-for-profit",
                                    "Baccalaureate/Associate's Colleges",
                                    "Master's Colleges and Universities (larger programs)"]
                                ),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            # Several categorical features in college scorecard have > 10k
            # unique values; so we label-encode instead of one-hot encoding.
            categorical_features="label_encode",
            # Some important numeric features are not reported by universities
            # in a way that could be systematic (and we would like these included
            # in the sample, not excluded), so we use kbins
            numeric_features="kbins",
            n_bins=100,
            dropna=None,
        ),
        tabular_dataset_kwargs={},
    )
for index in range(COLLEGE_SCORECARD_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["college_scorecard_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='CCBASIC',
                                domain_split_ood_values=[
                                    'Special Focus Institutions--Other special-focus institutions',
                                    'Special Focus Institutions--Theological seminaries, Bible colleges, and other faith-related institutions',
                                    "Associate's--Private For-profit 4-year Primarily Associate's",
                                    'Baccalaureate Colleges--Diverse Fields',
                                    'Special Focus Institutions--Schools of art, music, and design',
                                    "Associate's--Private Not-for-profit",
                                    "Baccalaureate/Associate's Colleges",
                                    "Master's Colleges and Universities (larger programs)"]
                                ),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            # Several categorical features in college scorecard have > 10k
            # unique values; so we label-encode instead of one-hot encoding.
            categorical_features="label_encode",
            # Some important numeric features are not reported by universities
            # in a way that could be systematic (and we would like these included
            # in the sample, not excluded), so we use kbins
            numeric_features="kbins",
            n_bins=100,
            dropna=None,
        ),
        tabular_dataset_kwargs={},
    )

for index in range(MIMIC_EXTRACT_LOS_3_FEATURES_CAUSAL_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["mimic_extract_los_3_causal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare"]),

        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[f for f in _MIMIC_EXTRACT_PASSTHROUGH_COLUMNS
                                 if f in MIMIC_EXTRACT_LOS_3_FEATURES_CAUSAL_SUBSETS[index].names]),
        tabular_dataset_kwargs={"task": "los_3",
                                "name": "mimic_extract_los_3_causal_test_"+f"{index}"})
# for index in range(MIMIC_EXTRACT_LOS_3_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
#     NON_BENCHMARK_CONFIGS["mimic_extract_los_3_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
    # splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
    #                         ood_val_size=DEFAULT_OOD_VAL_SIZE,
    #                         random_state=DEFAULT_RANDOM_STATE,
    #                         id_test_size=DEFAULT_ID_TEST_SIZE,
    #                         domain_split_varname="insurance",
    #                         domain_split_ood_values=["Medicare"]),
    # grouper=Grouper({"gender": ['M'], }, drop=False),
    # preprocessor_config=PreprocessorConfig(
    #     passthrough_columns=[f for f in _MIMIC_EXTRACT_PASSTHROUGH_COLUMNS
    #                          if f in MIMIC_EXTRACT_LOS_3_FEATURES_ARGUABLYCAUSAL_SUPERSETS[index].names]),
    # tabular_dataset_kwargs={"task": "los_3",
    #                         "name": "mimic_extract_los_3_arguablycausal_test_"+f"{index}"})


for index in range(MIMIC_EXTRACT_MORT_HOSP_FEATURES_CAUSAL_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["mimic_extract_mort_hosp_causal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare"]),

        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[f for f in _MIMIC_EXTRACT_PASSTHROUGH_COLUMNS
                                 if f in MIMIC_EXTRACT_MORT_HOSP_FEATURES_CAUSAL_SUBSETS[index].names]),
        tabular_dataset_kwargs={"task": "mort_hosp",
                                "name": "mimic_extract_mort_hosp_causal_test_"+f"{index}"})

# for index in range(MIMIC_EXTRACT_MORT_HOSP_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
#     NON_BENCHMARK_CONFIGS["mimic_extract_mort_hosp_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
    # splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
    #                         ood_val_size=DEFAULT_OOD_VAL_SIZE,
    #                         random_state=DEFAULT_RANDOM_STATE,
    #                         id_test_size=DEFAULT_ID_TEST_SIZE,
    #                         domain_split_varname="insurance",
    #                         domain_split_ood_values=["Medicare"]),

    # grouper=Grouper({"gender": ['M'], }, drop=False),
    # preprocessor_config=PreprocessorConfig(
    #     passthrough_columns=[f for f in _MIMIC_EXTRACT_PASSTHROUGH_COLUMNS
    #                          if f in MIMIC_EXTRACT_MORT_HOSP_FEATURES_ARGUABLYCAUSAL_SUPERSETS[index].names]),
    # tabular_dataset_kwargs={"task": "mort_hosp",
    #                         "name": "mimic_extract_mort_hosp_arguablycausal_test_"+f"{index}"})

for index in range(PHYSIONET_FEATURES_CAUSAL_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["physionet_causal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='ICULOS',
                                domain_split_gt_thresh=47.0),
        grouper=None,
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={"name": "physionet_causal_test_"+f"{index}"})
for index in range(PHYSIONET_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["physionet_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='ICULOS',
                                domain_split_gt_thresh=47.0),
        grouper=None,
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={"name": "physionet_arguablycausal_test_"+f"{index}"})


for index in range(NHANES_LEAD_FEATURES_CAUSAL_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["nhanes_lead_causal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='INDFMPIRBelowCutoff',
                                domain_split_ood_values=[1.]),
        # Race (non. hispanic white vs. all others; male vs. all others)
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
        tabular_dataset_kwargs={"nhanes_task": "lead", "years": NHANES_YEARS})

for index in range(NHANES_LEAD_FEATURES_ARGUABLYCAUSAL_SUPERSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["nhanes_lead_arguablycausal_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='INDFMPIRBelowCutoff',
                                domain_split_ood_values=[1.]),
        # Race (non. hispanic white vs. all others; male vs. all others)
        grouper=Grouper({"RIDRETH_merged": [3, ], "RIAGENDR": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
        tabular_dataset_kwargs={"nhanes_task": "lead", "years": NHANES_YEARS})

################################################################################
# Configuration for additional robustness tests
################################################################################

for index in range(ACS_INCOME_FEATURES_RANDOM_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["acsincome_random_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['01']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsincome"})

for index in range(ACS_UNEMPLOYMENT_FEATURES_RANDOM_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["acsunemployment_random_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='SCHL',
                                # No high school diploma vs. GED/diploma or higher.
                                domain_split_ood_values=['01', '02', '03', '04',
                                                         '05', '06', '07', '08',
                                                         '09', '10', '11', '12',
                                                         '13', '14', '15']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsunemployment"})
    
for index in range(ACS_FOODSTAMPS_FEATURES_RANDOM_SUBSETS_NUMBER): 
    NON_BENCHMARK_CONFIGS["acsfoodstamps_random_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['06']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsfoodstamps"})

for index in range(ACS_PUBCOV_FEATURES_RANDOM_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["acspubcov_random_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIS",
                                domain_split_ood_values=['1.0']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acspubcov", "name": "acspubcov_random_test_"+f"{index}",
                                "years": ACS_YEARS})
    
for index in range(BRFSS_DIABETES_FEATURES_RANDOM_SUBSETS_NUMBER):
     NON_BENCHMARK_CONFIGS["brfss_diabetes_random_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="PRACE1",
                                domain_split_ood_values=[
                                    2, 3, 4, 5, 6],
                                domain_split_id_values=[1, ]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_diabetes_random_test_"+f"{index}",
                                "task": "diabetes", "years": BRFSS_YEARS})

for index in range(BRFSS_BLOOD_PRESSURE_FEATURES_RANDOM_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["brfss_blood_pressure_random_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="BMI5CAT",
                                # OOD values: [1 underweight, 2 normal weight], [3 overweight, 4 obese]
                                domain_split_ood_values=['3.0', '4.0']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_blood_pressure_random_test_"+f"{index}",
                                "task": "blood_pressure",
                                "years": BRFSS_YEARS},
    )

for index in range(ANES_FEATURES_RANDOM_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["anes_random_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='VCF0112',
                                domain_split_ood_values=['3.0']),
        # male vs. all others; white non-hispanic vs. others
        grouper=None,
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={})

for index in range(SIPP_FEATURES_RANDOM_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["sipp_random_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="CITIZENSHIP_STATUS",
                                domain_split_ood_values=['1.0']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={})

for index in range(DIABETES_READMISSION_FEATURES_RANDOM_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["diabetes_readmission_random_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='admission_source_id',
                                domain_split_ood_values=[7, ]),
        # male vs. all others; white non-hispanic vs. others
        grouper=None,
        # Note: using min_frequency=0.01 reduces data
        # dimensionality from ~2400 -> 169 columns.
        # This is due to high cardinality of 'diag_*' features.
        preprocessor_config=PreprocessorConfig(
            min_frequency=0.01),
        tabular_dataset_kwargs={})
    
for index in range(MEPS_FEATURES_RANDOM_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["meps_random_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="INSCOV19",
                                domain_split_ood_values=[0]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={})

for index in range(COLLEGE_SCORECARD_FEATURES_RANDOM_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["college_scorecard_random_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='CCBASIC',
                                domain_split_ood_values=[
                                    'Special Focus Institutions--Other special-focus institutions',
                                    'Special Focus Institutions--Theological seminaries, Bible colleges, and other faith-related institutions',
                                    "Associate's--Private For-profit 4-year Primarily Associate's",
                                    'Baccalaureate Colleges--Diverse Fields',
                                    'Special Focus Institutions--Schools of art, music, and design',
                                    "Associate's--Private Not-for-profit",
                                    "Baccalaureate/Associate's Colleges",
                                    "Master's Colleges and Universities (larger programs)"]
                                ),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            # Several categorical features in college scorecard have > 10k
            # unique values; so we label-encode instead of one-hot encoding.
            categorical_features="label_encode",
            # Some important numeric features are not reported by universities
            # in a way that could be systematic (and we would like these included
            # in the sample, not excluded), so we use kbins
            numeric_features="kbins",
            n_bins=100,
            dropna=None,
        ),
        tabular_dataset_kwargs={},
    )

for index in range(NHANES_LEAD_FEATURES_RANDOM_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["nhanes_lead_random_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='INDFMPIRBelowCutoff',
                                domain_split_ood_values=[1.]),
        # Race (non. hispanic white vs. all others; male vs. all others)
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
        tabular_dataset_kwargs={"nhanes_task": "lead", "years": NHANES_YEARS})

for index in range(PHYSIONET_FEATURES_RANDOM_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["physionet_random_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='ICULOS',
                                domain_split_gt_thresh=47.0),
        grouper=None,
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={"name": "physionet_random_test_"+f"{index}"})

for index in range(ASSISTMENTS_FEATURES_RANDOM_SUBSETS_NUMBER):
    NON_BENCHMARK_CONFIGS["assistments_random_test_"+f"{index}"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='school_id',
                                domain_split_ood_values=[5040.0,
                                                         11502.0,
                                                         11318.0,
                                                         11976.0,
                                                         12421.0,
                                                         12379.0,
                                                         11791.0,
                                                         8359.0,
                                                         12406.0,
                                                         7594.0]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[
                "skill_id", "bottom_hint", "first_action"],
        ),
        tabular_dataset_kwargs={},
    )


#####################################s###########################################
# Configuration for invariant causal prediction
################################################################################

NON_BENCHMARK_CONFIGS["acsunemployment_icp"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='SCHL',
                                # No high school diploma vs. GED/diploma or higher.
                                domain_split_ood_values=['01', '02', '03', '04',
                                                         '05', '06', '07', '08',
                                                         '09', '10', '11', '12',
                                                         '13', '14', '15']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsunemployment"})

################################################################################
# Configuration for pc causal discovery
################################################################################
NON_BENCHMARK_CONFIGS["brfss_diabetes_pc"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="PRACE1",
                                domain_split_ood_values=[
                                    2, 3, 4, 5, 6],
                                domain_split_id_values=[1, ]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_diabetes_pc",
                                "task": "diabetes", "years": BRFSS_YEARS})
    
NON_BENCHMARK_CONFIGS["acsunemployment_pc"] = ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='SCHL',
                                # No high school diploma vs. GED/diploma or higher.
                                domain_split_ood_values=['01', '02', '03', '04',
                                                         '05', '06', '07', '08',
                                                         '09', '10', '11', '12',
                                                         '13', '14', '15']),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsunemployment"})

