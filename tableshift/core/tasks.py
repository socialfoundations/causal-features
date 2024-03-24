"""
Contains task configurations.

A task is a set of features (including both predictors and a target variable)
along with a DataSource. Tasks are the fundamental benchmarks that comprise
the tableshift benchmark, as well as causal feature selections of these fundamental benchmarks.

Modified for 'Predictors from Causal Features Do Not Generalize Better to New Domains'.
"""

from dataclasses import dataclass
from typing import Any, Dict
from .data_source import *
from .features import FeatureList

from tableshift.datasets import *


@dataclass
class TaskConfig:
    # The data_source_cls instantiates the DataSource,
    # which fetches data and preprocesses it using a preprocess_fn.
    data_source_cls: Any
    # The feature_list describes the schema of the data *after* the
    # preprocess_fn is applied. It is used to check the output of the
    # preprocess_fn, and features are dropped or type-cast as necessary.
    feature_list: FeatureList


# Mapping of task names to their configs. An arbitrary number of tasks
# can be created from a single data source, by specifying different
# preprocess_fn and features.
_TASK_REGISTRY: Dict[str, TaskConfig] = {
    "acsincome":
        TaskConfig(ACSDataSource,
                   ACS_INCOME_FEATURES + ACS_SHARED_FEATURES),
    "acsincome_causal":
        TaskConfig(ACSDataSource,
                   ACS_INCOME_FEATURES_CAUSAL),
    "acsincome_arguablycausal":
        TaskConfig(ACSDataSource,
                   ACS_INCOME_FEATURES_ARGUABLYCAUSAL),
    "acsincome_anticausal":
        TaskConfig(ACSDataSource,
                   ACS_INCOME_FEATURES_ANTICAUSAL),
    "acsfoodstamps":
        TaskConfig(ACSDataSource,
                   ACS_FOODSTAMPS_FEATURES + ACS_SHARED_FEATURES),
    "acsfoodstamps_causal":
        TaskConfig(ACSDataSource,
                   ACS_FOODSTAMPS_FEATURES_CAUSAL),
    "acsfoodstamps_arguablycausal":
        TaskConfig(ACSDataSource,
                   ACS_FOODSTAMPS_FEATURES_ARGUABLYCAUSAL),
    "acspubcov":
        TaskConfig(ACSDataSource,
                   ACS_PUBCOV_FEATURES + ACS_SHARED_FEATURES),
    "acspubcov_causal":
        TaskConfig(ACSDataSource,
                   ACS_PUBCOV_FEATURES_CAUSAL),
    "acspubcov_arguablycausal":
        TaskConfig(ACSDataSource,
                   ACS_PUBCOV_FEATURES_ARGUABLYCAUSAL),
    "acsunemployment":
        TaskConfig(ACSDataSource,
                   ACS_UNEMPLOYMENT_FEATURES + ACS_SHARED_FEATURES),
    "acsunemployment_causal":
        TaskConfig(ACSDataSource,
                   ACS_UNEMPLOYMENT_FEATURES_CAUSAL),
    "acsunemployment_arguablycausal":
        TaskConfig(ACSDataSource,
                   ACS_UNEMPLOYMENT_FEATURES_ARGUABLYCAUSAL),
    "acsunemployment_anticausal":
        TaskConfig(ACSDataSource,
                   ACS_UNEMPLOYMENT_FEATURES_ANTICAUSAL),
    "adult":
        TaskConfig(AdultDataSource, ADULT_FEATURES),
    "anes":
        TaskConfig(ANESDataSource, ANES_FEATURES),
    "anes_causal":
        TaskConfig(ANESDataSource, ANES_FEATURES_CAUSAL),
    "anes_arguablycausal":
        TaskConfig(ANESDataSource, ANES_FEATURES_ARGUABLYCAUSAL),
    "assistments":
        TaskConfig(AssistmentsDataSource, ASSISTMENTS_FEATURES),
    "assistments_causal":
        TaskConfig(AssistmentsDataSource, ASSISTMENTS_FEATURES_CAUSAL),
    "assistments_arguablycausal":
        TaskConfig(AssistmentsDataSource, ASSISTMENTS_FEATURES_ARGUABLYCAUSAL),
    "brfss_diabetes":
        TaskConfig(BRFSSDataSource, BRFSS_DIABETES_FEATURES),
    "brfss_diabetes_causal":
        TaskConfig(BRFSSDataSource, BRFSS_DIABETES_FEATURES_CAUSAL),
    "brfss_diabetes_arguablycausal":
        TaskConfig(BRFSSDataSource, BRFSS_DIABETES_FEATURES_ARGUABLYCAUSAL),
    "brfss_diabetes_anticausal":
        TaskConfig(BRFSSDataSource, BRFSS_DIABETES_FEATURES_ANTICAUSAL),
    "brfss_blood_pressure":
        TaskConfig(BRFSSDataSource, BRFSS_BLOOD_PRESSURE_FEATURES),
    "brfss_blood_pressure_causal":
        TaskConfig(BRFSSDataSource, BRFSS_BLOOD_PRESSURE_FEATURES_CAUSAL),
    "brfss_blood_pressure_arguablycausal":
        TaskConfig(BRFSSDataSource, BRFSS_BLOOD_PRESSURE_FEATURES_ARGUABLYCAUSAL),
    "brfss_blood_pressure_anticausal":
        TaskConfig(BRFSSDataSource, BRFSS_BLOOD_PRESSURE_FEATURES_ANTICAUSAL),
    "communities_and_crime":
        TaskConfig(CommunitiesAndCrimeDataSource, CANDC_FEATURES),
    "compas":
        TaskConfig(COMPASDataSource, COMPAS_FEATURES),
    "diabetes_readmission":
        TaskConfig(DiabetesReadmissionDataSource,
                   DIABETES_READMISSION_FEATURES),
    "diabetes_readmission_causal":
        TaskConfig(DiabetesReadmissionDataSource,
                   DIABETES_READMISSION_FEATURES_CAUSAL),
    "diabetes_readmission_arguablycausal":
        TaskConfig(DiabetesReadmissionDataSource,
                   DIABETES_READMISSION_FEATURES_ARGUABLYCAUSAL),
    "german":
        TaskConfig(GermanDataSource, GERMAN_FEATURES),
    "heloc":
        TaskConfig(HELOCDataSource, HELOC_FEATURES),
    "metamimic_alcohol":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_ALCOHOL_FEATURES),
    "metamimic_anemia":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_ANEMIA_FEATURES),
    "metamimic_atrial":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_ATRIAL_FEATURES),
    "metamimic_diabetes":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_DIABETES_FEATURES),
    "metamimic_heart":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_HEART_FEATURES),
    "metamimic_hypertension":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_HYPERTENSIVE_FEATURES),
    "metamimic_hypotension":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_HYPOTENSION_FEATURES),
    "metamimic_ischematic":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_ISCHEMATIC_FEATURES),
    "metamimic_lipoid":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_LIPOID_FEATURES),
    "metamimic_overweight":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_OVERWEIGHT_FEATURES),
    "metamimic_purpura":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_PURPURA_FEATURES),
    "metamimic_respiratory":
        TaskConfig(MetaMIMICDataSource, METAMIMIC_RESPIRATORY_FEATURES),
    "mimic_extract_los_3":
        TaskConfig(MIMICExtractDataSource, MIMIC_EXTRACT_LOS_3_FEATURES),
    "mimic_extract_los_3_causal":
        TaskConfig(MIMICExtractDataSource, MIMIC_EXTRACT_LOS_3_FEATURES_CAUSAL),
    "mimic_extract_los_3_arguablycausal":
        TaskConfig(MIMICExtractDataSource, MIMIC_EXTRACT_LOS_3_FEATURES_ARGUABLYCAUSAL),
    "mimic_extract_los_3_selected":
        TaskConfig(MIMICExtractDataSource, MIMIC_EXTRACT_LOS_3_SELECTED_FEATURES),
    "mimic_extract_mort_hosp":
        TaskConfig(MIMICExtractDataSource, MIMIC_EXTRACT_MORT_HOSP_FEATURES),
    "mimic_extract_mort_hosp_causal":
        TaskConfig(MIMICExtractDataSource, MIMIC_EXTRACT_MORT_HOSP_FEATURES_CAUSAL),
    "mimic_extract_mort_hosp_arguablycausal":
        TaskConfig(MIMICExtractDataSource, MIMIC_EXTRACT_MORT_HOSP_FEATURES_ARGUABLYCAUSAL),
    "mimic_extract_mort_hosp_selected":
        TaskConfig(MIMICExtractDataSource, MIMIC_EXTRACT_MORT_HOSP_SELECTED_FEATURES),
    "mooc":
        TaskConfig(MOOCDataSource, MOOC_FEATURES),
    "nhanes_cholesterol":
        TaskConfig(NHANESDataSource,
                   NHANES_CHOLESTEROL_FEATURES +
                   NHANES_SHARED_FEATURES),
    "nhanes_lead":
        TaskConfig(NHANESDataSource,
                   NHANES_SHARED_FEATURES +
                   NHANES_LEAD_FEATURES),
    "nhanes_lead_causal":
        TaskConfig(NHANESDataSource, NHANES_LEAD_FEATURES_CAUSAL),
    "nhanes_lead_arguablycausal":
        TaskConfig(NHANESDataSource, NHANES_LEAD_FEATURES_ARGUABLYCAUSAL),
    "physionet":
        TaskConfig(PhysioNetDataSource, PHYSIONET_FEATURES),
    "physionet_causal":
        TaskConfig(PhysioNetDataSource, PHYSIONET_FEATURES_CAUSAL),
    "physionet_arguablycausal":
        TaskConfig(PhysioNetDataSource, PHYSIONET_FEATURES_ARGUABLYCAUSAL),
    "electricity":
        TaskConfig(GrinstajnHFDataSource, ELECTRICITY_FEATURES),
    "bank-marketing":
        TaskConfig(GrinstajnHFDataSource, BANK_MARKETING_FEATURES),
    "california":
        TaskConfig(GrinstajnHFDataSource, CALIFORNIA_FEATURES),
    'covertype':
        TaskConfig(GrinstajnHFDataSource, COVERTYPE_FEATURES),
    'credit':
        TaskConfig(GrinstajnHFDataSource, CREDIT_FEATURES),
    'default-of-credit-card-clients':
        TaskConfig(GrinstajnHFDataSource, DEFAULT_OF_CREDIT_CLIENTS_FEATURES),
    'eye_movements':
        TaskConfig(GrinstajnHFDataSource, EYE_MOVEMENTS_FEATURES),
    'Higgs':
        TaskConfig(GrinstajnHFDataSource, HIGGS_FEATURES),
    'MagicTelescope':
        TaskConfig(GrinstajnHFDataSource, MAGIC_TELESCOPE_FEATURES),
    'MiniBooNE':
        TaskConfig(GrinstajnHFDataSource, MINI_BOONE_FEATURES),
    'road-safety':
        TaskConfig(GrinstajnHFDataSource, ROAD_SAFETY_FEATURES),
    'pol':
        TaskConfig(GrinstajnHFDataSource, POL_FEATURES),
    'jannis':
        TaskConfig(GrinstajnHFDataSource, JANNIS_FEATURES),
    'house_16H':
        TaskConfig(GrinstajnHFDataSource, HOUSE_16H_FEATURES),
    'amazon':
        TaskConfig(AmazonDataSource, AMAZON_FEATURES),
    'appetency':
        TaskConfig(KddCup2009DataSource, APPETENCY_FEATURES),
    'churn':
        TaskConfig(KddCup2009DataSource, CHURN_FEATURES),
    'upselling':
        TaskConfig(KddCup2009DataSource, UPSELLING_FEATURES),
    'click':
        TaskConfig(ClickDataSource, CLICK_FEATURES),
    'kick':
        TaskConfig(KickDataSource, KICK_FEATURES),
    'product_sentiment_machine_hack':
        TaskConfig(AutoMLBenchmarkDataSource, PROD_FEATURES),
    'data_scientist_salary':
        TaskConfig(AutoMLBenchmarkDataSource, SALARY_FEATURES),
    'melbourne_airbnb':
        TaskConfig(AutoMLBenchmarkDataSource, AIRBNB_FEATURES),
    'news_channel':
        TaskConfig(AutoMLBenchmarkDataSource, NEWS_CHANNEL_FEATURES),
    'wine_reviews':
        TaskConfig(AutoMLBenchmarkDataSource, WINE_REVIEWS_FEATURES),
    'imdb_genre_prediction':
        TaskConfig(AutoMLBenchmarkDataSource, IMDB_FEATURES),
    'fake_job_postings2':
        TaskConfig(AutoMLBenchmarkDataSource, FAKE_JOBS_FEATURES),
    'kick_starter_funding':
        TaskConfig(AutoMLBenchmarkDataSource, KICKSTARTER_FEATURES),
    'jigsaw_unintended_bias100K':
        TaskConfig(AutoMLBenchmarkDataSource, JIGSAW_FEATURES),
    'iris':
        TaskConfig(IrisDataSource, IRIS_FEATURES),
    'dry-bean':
        TaskConfig(DryBeanDataSource, DRY_BEAN_FEATURES),
    'heart-disease':
        TaskConfig(HeartDiseaseDataSource, HEART_DISEASE_FEATURES),
    'wine':
        TaskConfig(WineCultivarsDataSource, WINE_CULTIVARS_FEATURES),
    'wine-quality':
        TaskConfig(WineQualityDataSource, WINE_QUALITY_FEATURES),
    'rice':
        TaskConfig(RiceDataSource, RICE_FEATURES),
    'breast-cancer':
        TaskConfig(BreastCancerDataSource, BREAST_CANCER_FEATURES),
    'cars':
        TaskConfig(CarDataSource, CAR_FEATURES),
    'raisin':
        TaskConfig(RaisinDataSource, RAISIN_FEATURES),
    'abalone':
        TaskConfig(AbaloneDataSource, ABALONE_FEATURES),
    'otto-products':
        TaskConfig(OttoProductsDataSource, OTTO_FEATURES),
    'sf-crime':
        TaskConfig(SfCrimeDataSource, SF_CRIME_FEATURES),
    'plasticc':
        TaskConfig(PlasticcDataSource, PLASTICC_FEATURES),
    'walmart':
        TaskConfig(WalmartDataSource, WALMART_FEATURES),
    'tradeshift':
        TaskConfig(TradeShiftDataSource, TRADESHIFT_FEATURES),
    'schizophrenia':
        TaskConfig(SchizophreniaDataSource, SCHIZOPHRENIA_FEATURES),
    'titanic':
        TaskConfig(TitanicDataSource, TITANIC_FEATURES),
    'santander-transactions':
        TaskConfig(SantanderTransactionDataSource,
                   SANTANDER_TRANSACTION_FEATURES),
    'home-credit-default-risk':
        TaskConfig(HomeCreditDefaultDataSource, HOME_CREDIT_DEFAULT_FEATURES),
    'ieee-fraud-detection':
        TaskConfig(IeeFraudDetectionDataSource, IEEE_FRAUD_DETECTION_FEATURES),
    'safe-driver-prediction':
        TaskConfig(SafeDriverPredictionDataSource,
                   SAFE_DRIVER_PREDICTION_FEATURES),
    'santander-customer-satisfaction':
        TaskConfig(SantanderCustomerSatisfactionDataSource,
                   SANTANDER_CUSTOMER_SATISFACTION_FEATURES),
    'amex-default':
        TaskConfig(AmexDefaultDataSource, AMEX_DEFAULT_FEATURES),
    'ad-fraud':
        TaskConfig(AdFraudDataSource, AD_FRAUD_FEATURES),
    'college_scorecard':
        TaskConfig(CollegeScorecardDataSource, COLLEGE_SCORECARD_FEATURES),
    'college_scorecard_causal':
        TaskConfig(CollegeScorecardDataSource, COLLEGE_SCORECARD_FEATURES_CAUSAL),
    'college_scorecard_arguablycausal':
        TaskConfig(CollegeScorecardDataSource, COLLEGE_SCORECARD_FEATURES_ARGUABLYCAUSAL),
    'meps':
        TaskConfig(MEPSDataSource, MEPS_FEATURES),
    'meps_causal':
        TaskConfig(MEPSDataSource, MEPS_FEATURES_CAUSAL),
    'meps_arguablycausal':
        TaskConfig(MEPSDataSource, MEPS_FEATURES_ARGUABLYCAUSAL),
    'sipp':
        TaskConfig(SIPPDataSource, SIPP_FEATURES),
    'sipp_causal':
        TaskConfig(SIPPDataSource, SIPP_FEATURES_CAUSAL),
    'sipp_arguablycausal':
        TaskConfig(SIPPDataSource, SIPP_FEATURES_ARGUABLYCAUSAL),
    'sipp_anticausal':
        TaskConfig(SIPPDataSource, SIPP_FEATURES_ANTICAUSAL),
}

################################################################################
# Tasks for robustness tests
################################################################################

# for index, subset in enumerate(ACS_INCOME_FEATURES_CAUSAL_SUBSETS):
#     _TASK_REGISTRY["acsincome_causal_test_"+f"{index}"] = TaskConfig(ACSDataSource, subset)

# for index, superset in enumerate(ACS_INCOME_FEATURES_ARGUABLYCAUSAL_SUPERSETS):
#     _TASK_REGISTRY["acsincome_arguablycausal_test_"+f"{index}"] = TaskConfig(ACSDataSource, superset)

# for index, subset in enumerate(ACS_FOODSTAMPS_FEATURES_CAUSAL_SUBSETS):
#     _TASK_REGISTRY["acsfoodstamps_causal_test_"+f"{index}"] = TaskConfig(ACSDataSource, subset)

# for index, superset in enumerate(ACS_FOODSTAMPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS):
#     _TASK_REGISTRY["acsfoodstamps_arguablycausal_test_"+f"{index}"] = TaskConfig(ACSDataSource, superset)

# for index, subset in enumerate(ACS_PUBCOV_FEATURES_CAUSAL_SUBSETS):
#     _TASK_REGISTRY["acspubcov_causal_test_"+f"{index}"] = TaskConfig(ACSDataSource, subset)

# for index, superset in enumerate(ACS_PUBCOV_FEATURES_ARGUABLYCAUSAL_SUPERSETS):
#     _TASK_REGISTRY["acspubcov_arguablycausal_test_"+f"{index}"] = TaskConfig(ACSDataSource, superset)

# for index, subset in enumerate(ACS_UNEMPLOYMENT_FEATURES_CAUSAL_SUBSETS):
#     _TASK_REGISTRY["acsunemployment_causal_test_"+f"{index}"] = TaskConfig(ACSDataSource, subset)

# for index, superset in enumerate(ACS_UNEMPLOYMENT_FEATURES_ARGUABLYCAUSAL_SUPERSETS):
#     _TASK_REGISTRY["acsunemployment_arguablycausal_test_"+f"{index}"] = TaskConfig(ACSDataSource, superset)

# for index, subset in enumerate(BRFSS_DIABETES_FEATURES_CAUSAL_SUBSETS):
#     _TASK_REGISTRY["brfss_diabetes_causal_test_"+f"{index}"] = TaskConfig(BRFSSDataSource, subset)

# for index, superset in enumerate(BRFSS_DIABETES_FEATURES_ARGUABLYCAUSAL_SUPERSETS):
#     _TASK_REGISTRY["brfss_diabetes_arguablycausal_test_"+f"{index}"] = TaskConfig(BRFSSDataSource, superset)

# for index, subset in enumerate(BRFSS_BLOOD_PRESSURE_FEATURES_CAUSAL_SUBSETS):
#     _TASK_REGISTRY["brfss_blood_pressure_causal_test_"+f"{index}"] = TaskConfig(BRFSSDataSource, subset)

# for index, superset in enumerate(BRFSS_BLOOD_PRESSURE_FEATURES_ARGUABLYCAUSAL_SUPERSETS):
#     _TASK_REGISTRY["brfss_blood_pressure_arguablycausal_test_"+f"{index}"] = TaskConfig(BRFSSDataSource, superset)

# for index, subset in enumerate(DIABETES_READMISSION_FEATURES_CAUSAL_SUBSETS):
#     _TASK_REGISTRY["diabetes_readmission_causal_test_"+f"{index}"] = TaskConfig(DiabetesReadmissionDataSource, subset)

# for index, superset in enumerate(DIABETES_READMISSION_FEATURES_ARGUABLYCAUSAL_SUPERSETS):
#     _TASK_REGISTRY["diabetes_readmission_arguablycausal_test_" +
#                    f"{index}"] = TaskConfig(DiabetesReadmissionDataSource, superset)

# for index, subset in enumerate(ANES_FEATURES_CAUSAL_SUBSETS):
#     _TASK_REGISTRY["anes_causal_test_"+f"{index}"] = TaskConfig(ANESDataSource, subset)

# for index, superset in enumerate(ANES_FEATURES_ARGUABLYCAUSAL_SUPERSETS):
#     _TASK_REGISTRY["anes_arguablycausal_test_"+f"{index}"] = TaskConfig(ANESDataSource, superset)

# for index, subset in enumerate(ASSISTMENTS_FEATURES_CAUSAL_SUBSETS):
#     _TASK_REGISTRY["assistments_causal_test_"+f"{index}"] = TaskConfig(AssistmentsDataSource, subset)

# for index, superset in enumerate(ASSISTMENTS_FEATURES_ARGUABLYCAUSAL_SUPERSETS):
#     _TASK_REGISTRY["assistments_arguablycausal_test_"+f"{index}"] = TaskConfig(AssistmentsDataSource, superset)

# for index, subset in enumerate(COLLEGE_SCORECARD_FEATURES_CAUSAL_SUBSETS):
#     _TASK_REGISTRY["college_scorecard_causal_test_"+f"{index}"] = TaskConfig(CollegeScorecardDataSource, subset)

# for index, superset in enumerate(COLLEGE_SCORECARD_FEATURES_ARGUABLYCAUSAL_SUPERSETS):
#     _TASK_REGISTRY["college_scorecard_arguablycausal_test_" +
#                    f"{index}"] = TaskConfig(CollegeScorecardDataSource, superset)

# for index, subset in enumerate(MIMIC_EXTRACT_LOS_3_FEATURES_CAUSAL_SUBSETS):
#     _TASK_REGISTRY["mimic_extract_los_3_causal_test_"+f"{index}"] = TaskConfig(MIMICExtractDataSource, subset)

# for index, subset in enumerate(MIMIC_EXTRACT_MORT_HOSP_FEATURES_CAUSAL_SUBSETS):
#     _TASK_REGISTRY["mimic_extract_mort_hosp_causal_test_"+f"{index}"] = TaskConfig(MIMICExtractDataSource, subset)


# for index, subset in enumerate(SIPP_FEATURES_CAUSAL_SUBSETS):
#     _TASK_REGISTRY["sipp_causal_test_"+f"{index}"] = TaskConfig(SIPPDataSource, subset)

# for index, superset in enumerate(SIPP_FEATURES_ARGUABLYCAUSAL_SUPERSETS):
#     _TASK_REGISTRY["sipp_arguablycausal_test_"+f"{index}"] = TaskConfig(SIPPDataSource, superset)

# for index, subset in enumerate(MEPS_FEATURES_CAUSAL_SUBSETS):
#     _TASK_REGISTRY["meps_causal_test_"+f"{index}"] = TaskConfig(MEPSDataSource, subset)

# for index, superset in enumerate(MEPS_FEATURES_ARGUABLYCAUSAL_SUPERSETS):
#     _TASK_REGISTRY["meps_arguablycausal_test_"+f"{index}"] = TaskConfig(MEPSDataSource, superset)

# for index, subset in enumerate(PHYSIONET_FEATURES_CAUSAL_SUBSETS):
#     _TASK_REGISTRY["physionet_causal_test_"+f"{index}"] = TaskConfig(PhysioNetDataSource, subset)

# for index, superset in enumerate(PHYSIONET_FEATURES_ARGUABLYCAUSAL_SUPERSETS):
#     _TASK_REGISTRY["physionet_arguablycausal_test_"+f"{index}"] = TaskConfig(PhysioNetDataSource, superset)

# for index, subset in enumerate(NHANES_LEAD_FEATURES_CAUSAL_SUBSETS):
#     _TASK_REGISTRY["nhanes_lead_causal_test_"+f"{index}"] = TaskConfig(NHANESDataSource, subset)

# for index, superset in enumerate(NHANES_LEAD_FEATURES_ARGUABLYCAUSAL_SUPERSETS):
#     _TASK_REGISTRY["nhanes_lead_arguablycausal_test_"+f"{index}"] = TaskConfig(NHANESDataSource, superset)

################################################################################
# Tasks for additional robustness tests
################################################################################

# for index, subset in enumerate(ACS_INCOME_FEATURES_RANDOM_SUBSETS):
#     _TASK_REGISTRY["acsincome_random_test_"+f"{index}"] = TaskConfig(ACSDataSource, subset)
    
for index, subset in enumerate(BRFSS_DIABETES_FEATURES_RANDOM_SUBSETS):
    _TASK_REGISTRY["brfss_diabetes_random_test_"+f"{index}"] = TaskConfig(BRFSSDataSource, subset)

def get_task_config(name: str) -> TaskConfig:
    if name in _TASK_REGISTRY:
        return _TASK_REGISTRY[name]
    else:
        raise NotImplementedError(f"task {name} not implemented.")
