"""Load preprocessed SIPP data."""

import pandas as pd


def load_sipp(wave_1_file='sipp/data/sipp_2014_wave_1.csv',
              wave_2_file='sipp/data/sipp_2014_wave_2.csv'):
    """Load sipp data from preprocessed csv files."""
    w1 = pd.read_csv(wave_1_file)
    w2 = pd.read_csv(wave_2_file)
    X = w1[w1.columns[5:]]
    y = w2['OPM_RATIO'] #1.0*(w2['OPM_RATIO'] >= 3)

    return X, y

if __name__ == "__main__":
    import os
    import json

    X, y = load_sipp()
    data = pd.concat([y,X], axis = 1)

    data.to_csv("sipp/sipp_2014.csv",index=False)

    cols = {"LIVING_QUARTERS_TYPE": "category",
            "MARITAL_STATUS": "category",
            "CITIZENSHIP_STATUS": "category",
            "FAMILY_SIZE_AVG": "float64",
            "ORIGIN": "category",
            "HOUSEHOLD_INC": "float64",
            "RECEIVED_WORK_COMP": "float64",
            "TANF_ASSISTANCE": "float64",
            "UNEMPLOYMENT_COMP": "category",
            "SEVERANCE_PAY_PENSION": "category",
            "FOSTER_CHILD_CARE_AMT": "float64",
            "CHILD_SUPPORT_AMT": "float64",
            "ALIMONY_AMT": "float64",
            "INCOME_FROM_ASSISTANCE": "float64",
            "INCOME": "float64",
            "SAVINGS_INV_AMOUNT": "float64",
            "UNEMPLOYMENT_COMP_AMOUNT": "float64",
            "VA_BENEFITS_AMOUNT": "float64",
            "RETIREMENT_INCOME_AMOUNT": "float64",
            "SURVIVOR_INCOME_AMOUNT": "float64",
            "DISABILITY_BENEFITS_AMOUNT": "float64",
            "FOOD_ASSISTANCE": "category",
            "EDUCATION": "category",
            "RACE": "category",
            "GENDER": "category",
            "AGE": "float64",
            "LIVING_OWNERSHIP": "category",
            "SNAP_ASSISTANCE": "float64",
            "WIC_ASSISTANCE": "float64",
            "MEDICARE_ASSISTANCE": "float64",
            "MEDICAID_ASSISTANCE": "float64",
            "HEALTHDISAB": "category",
            "DAYS_SICK": "float64",
            "HOSPITAL_NIGHTS": "float64",
            "PRESCRIPTION_MEDS": "category",
            "VISIT_DENTIST_NUM": "float64",
            "TRANSPORTATION_ASSISTANCE": "category",
            "VISIT_DOCTOR_NUM": "float64",
            "HEALTH_OVER_THE_COUNTER_PRODUCTS_PAY": "float64",
            "HEALTH_MEDICAL_CARE_PAY": "float64",
            "HEALTH_HEARING": "category",
            "HEALTH_SEEING": "category",
            "HEALTH_COGNITIVE": "category",
            "HEALTH_AMBULATORY": "category",
            "HEALTH_SELF_CARE": "category",
            "HEALTH_ERRANDS_DIFFICULTY": "category",
            "HEALTH_CORE_DISABILITY": "category",
            "HEALTH_SUPPLEMENTAL_DISABILITY": "category",
            "HEALTH_INSURANCE_PREMIUMS": "float64",
            "SOCIAL_SEC_BENEFITS": "category"}

    with open("sipp/sipp_feature_types.json", "w") as file:
        # Use json.dump to write the dictionary to the file
        json.dump(cols, file)

    demographic_features = ['AGE', 'GENDER', 'RACE', 'EDUCATION', 'MARITAL_STATUS',
                            'CITIZENSHIP_STATUS']
    
    cols = {key: value for key, value in cols.items() if key in demographic_features}

    with open("sipp/sipp_feature_demographic_types.json", "w") as file:
        # Use json.dump to write the dictionary to the file
        json.dump(cols, file)
