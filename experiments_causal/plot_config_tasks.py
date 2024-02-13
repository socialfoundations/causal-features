"""Python script to set configuration for tasks in plots."""

# Dictionary to map tasks to variable name of domain
dic_domain_label = {
    "acsemployment": "SCHL",
    "acsfoodstamps": "DIVISION",
    "acsincome": "DIVISION",
    "acspubcov": "DIS",
    "acsunemployment": "SCHL",
    "anes": "VCF0112",  # region
    "assistments": "school_id",
    "brfss_blood_pressure": "BMI5CAT",
    "brfss_diabetes": "PRACE1",
    "college_scorecard": "CCBASIC",
    "diabetes_readmission": "admission_source_id",
    "meps": "INSCOV19",
    "mimic_extract_los_3": "insurance",
    "mimic_extract_mort_hosp": "insurance",
    "nhanes_lead": "INDFMPIRBelowCutoff",
    "physionet": "ICULOS",  # ICU length of stay
    "sipp": "CITIZENSHIP_STATUS",
}

# Dictionary to map tasks to in-domain description
dic_id_domain = {
    "acsemployment": "High school diploma or higher",
    "acsfoodstamps": "Other U.S. Census divisions",
    "acsincome": "Other U.S. Census divisions",
    "acspubcov": "Without disability",
    "acsunemployment": "High school diploma or higher",
    "anes": "Other U.S. Census regions",
    "assistments": "approximately 700 schools",
    "brfss_blood_pressure": "Underweight and normal weight",
    "brfss_diabetes": "White",
    "college_scorecard": "Carnegie Classification: other institutional types",
    "diabetes_readmission": "Other admission sources",
    "meps": "Public insurance",
    "mimic_extract_los_3": "Private, Medicaid, Government, Self Pay",
    "mimic_extract_mort_hosp": "Private, Medicaid, Government, Self Pay",
    "nhanes_lead": "poverty-income ratio > 1.3",
    "physionet": "ICU length of stay <= 47 hours",
    "sipp": "U.S. citizen",
}

# Dictionary to map tasks to out-of-domain description
dic_ood_domain = {
    "acsemployment": "No high school diploma",
    "acsfoodstamps": "East South Central",
    "acsincome": "New England",
    "acspubcov": "With disability",
    "acsunemployment": "No high school diploma",
    "anes": "South",  # region
    "assistments": "10 new schools",
    "brfss_blood_pressure": "Overweight and obese",
    "brfss_diabetes": "Non white",
    "college_scorecard": "Special Focus Institutions [Faith-related, art & design and other fields],\n Baccalaureate/Associates Colleges,\n Master's Colleges and Universities [larger programs]",
    "diabetes_readmission": "Emergency Room",
    "meps": "Private insurance",
    "mimic_extract_los_3": "Medicare",
    "mimic_extract_mort_hosp": "Medicare",
    "nhanes_lead": "poverty-income ratio <= 1.3",
    "physionet": "ICU length of stay > 47 hours",  # ICU length of stay
    "sipp": "non U.S. citizen",
}

# Dictionary to map tasks to titles
dic_title = {
    "acsemployment": "Employment",
    "acsfoodstamps": "Food Stamps",
    "acsincome": "Income",
    "acspubcov": "PublicCoverage",
    "acsunemployment": "Unemployment",
    "anes": "Voting",
    "assistments": "ASSISTments",
    "brfss_blood_pressure": "Hypertension",
    "brfss_diabetes": "Diabetes",
    "college_scorecard": "College Scorecard",
    "diabetes_readmission": "Hospital Readmission",
    "meps": "Utilization",
    "mimic_extract_los_3": "ICU Length of Stay",
    "mimic_extract_mort_hosp": "Hospital Mortality",
    "nhanes_lead": "Childhood Lead",
    "physionet": "Sepsis",
    "sipp": "Poverty",
}
# Dictonart to map tasks to TableShift names
dic_tableshift = {
    "acsfoodstamps": "Food Stamps",
    "acsincome": "Income",
    "acspubcov": "Public Health Ins.",
    "acsunemployment": "Unemployment",
    "anes": "Voting",
    "assistments": "ASSISTments",
    "brfss_blood_pressure": "Hypertension",
    "brfss_diabetes": "Diabetes",
    "college_scorecard": "College Scorecard",
    "diabetes_readmission": "Hospital Readmission",
    "mimic_extract_los_3": "ICU Length of Stay",
    "mimic_extract_mort_hosp": "ICU Hospital Mortality",
    "nhanes_lead": "Childhood Lead",
    "physionet": "Sepsis",
}
