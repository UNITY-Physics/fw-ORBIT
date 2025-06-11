"""Parser module to parse gear config.json."""

import time
from typing import Tuple
import os
import re
from flywheel_gear_toolkit import GearToolkitContext
import flywheel
import pandas as pd
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


log = logging.getLogger(__name__)


def parse_config(context):
    """Parse the config and other options from the context, both gear and app options.

    Returns:
        gear_inputs
        gear_options: options for the gear
        app_options: options to pass to the app
    """

    # -------------- Get the gear configuration -------------- #

    api_key = context.get_input("api-key").get("key")
    fw = flywheel.Client(api_key=api_key)
    user = f"{fw.get_current_user().firstname} {fw.get_current_user().lastname} [{fw.get_current_user().email}]"

    print(f"Logged in as {fw.get_current_user().email}")

    input_container = context.client.get_analysis(context.destination["id"])
    proj_id = input_container.parents["project"]
    project_container = context.client.get(proj_id)
    project_label = project_container.label
    print("project label: ", project_label)

    # -------------------  Get Input Data -------------------  #

    df = context.get_input_path("input")
    if df:
        log.info(f"Loaded {df}")
        inputs_provided = True
    else:
        log.info("Session spreadsheet not provided")
        inputs_provided = False

    age_min = context.config.get("age_min_months")
    age_max = context.config.get("age_max_months")
    age_range = context.config.get("age_range")
    threshold = context.config.get("threshold")
    age_unit = context.config.get("age_unit")
    growth_curve = context.config.get("growth_curve")
    birth_weight_icv = context.config.get("birth_weight_icv")

    # -------------------  Get Input label -------------------  #

    # Specify the directory you want to list files from
    input_path = '/flywheel/v0/input/input'
    work_path = Path('/flywheel/v0/work')

    if not work_path.exists():
        work_path.mkdir(parents = True)

    # List all files in the specified directory
    #recon-all output, QC output , [add more as needed]
    input_labels = {} 
    name_key_maping = {
    "recon-all-clinical": "volumetric",
    "synthseg": "volumetric",
    "mrr_axireg": "volumetric",
    "minimorph":"volumetric"
}

    for filename in os.listdir(input_path):
    #for filename in file_inputs: #This line was used when debugging locally and specifying filenames
        for keyword, key in name_key_maping.items():
            if keyword in filename:
                # if key == "qc":
                #     input_labels['qc'] = filename
                # else:
                input_labels['volumetric'] = filename
                #break

    log.info(f"Input files found: {input_labels}")

    df_path = impute_information(context,input_labels['volumetric'])
    df_path = rename_columns (input_labels['volumetric'])
    df_path = os.path.join(work_path,filename)
    
    config_context = {"age_min_months": age_min, "age_max_months": age_max, "age_range": age_range,"age_unit": age_unit, "threshold": threshold, "growth_curve": growth_curve, "birth_weight_icv": birth_weight_icv}
    
    return user, df_path, input_labels, config_context, project_container, api_key

def impute_row(index, row, project, columns, log):
    subject_label = row['subject']
    session_label = row['session']
    result = {}

    try:
        subject = next((s for s in project.subjects() if s.label == subject_label), None)
        if not subject:
            return index, result
        subject = subject.reload()

        session = next((s for s in subject.sessions() if s.label == session_label), None)
        if not session:
            return index, result
        session = session.reload()

        for column_name in columns:
            try:
                if column_name == 'sex':
                    if subject.sex is not None:
                        result[column_name] = subject.sex
                        log.info(f"Imputing {column_name} for subject: {subject_label}: {subject.sex}")
                    elif session.info and "sex_at_birth" in session.info:
                        result[column_name] = session.info["sex_at_birth"]
                        log.info(f"Imputing {column_name} for subject: {subject_label}: {session.info['sex_at_birth']}")

                elif column_name == 'age' and (row['age'] is None or row['age'] < 0 or pd.isna(row['age'])):
                    if session.info and "age_at_scan_months" in session.info and session.info["age_at_scan_months"] != 0:
                        result[column_name] = session.info["age_at_scan_months"]
                        log.info(f"Imputing {column_name} for subject: {subject_label}: {session.info['age_at_scan_months']}")
                        #key_with_age = [k for k in session.info if 'age' in k and 'months' in k]
                        #if key_with_age:
                        #    age = session.info[key_with_age[0]]
                        #    result[column_name] = age
                            
                    elif session.age_years is not None:
                        result[column_name] = session.age_years * 12
                        log.info(f"Imputing {column_name} for subject: {subject_label}: {session.age_years * 12}")
                else:
                    result[column_name] = session.info.get(column_name)
                    log.info(f"Imputing {column_name} for subject: {subject_label}: {result[column_name]}")
            except Exception as e:
                log.info(f"Error imputing {column_name} for subject: {subject_label}: {e}")
    except Exception as e:
        log.info(f"General error for subject {subject_label}: {e}")
    
    return index, result

def impute_information(context,vols):

    """Imputes missing information needed for the plotting functions. 
    Currently it handles sex and age

    Returns:
        Dataframe with imputed information
    """

    log.info('Imputing missing information...')

    api_key = context.get_input("api-key").get("key")
    fw = flywheel.Client(api_key=api_key)

    input_container = context.client.get_analysis(context.destination["id"])
    proj_id = input_container.parents["project"]
    project_container = context.client.get(proj_id)
    project_label = project_container.label
    
    project = fw.projects.find_first(f'label="{project_label}"')

    input_path = '/flywheel/v0/input/input'
    output_path = '/flywheel/v0/output'
    work_path = '/flywheel/v0/work'

    df = pd.read_csv(os.path.join(input_path,vols))
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


    # Check if the sex or age column has empty values
    columns = ['sex','age','birth_length_cm','birth_weight_kg','current_hc_cm']
    futures = []
    
    #minimorph's output is dicom_age_in_months
    for age_column in ['dicom_age_in_months', 'template_age']:
        if age_column in df.columns:
            log.info(f"Extracting age from {age_column} column")
            # Step 1: Apply regex to extract digits
            df['age'] = df[age_column].str.replace(r'[a-zA-Z]', '', regex=True)
            # Step 2: Convert to numeric (this will turn non-convertible entries to NaN)
            df['age'] = pd.to_numeric(df[age_column], errors='coerce')

        
    df.loc[df['age'] < 0, 'age'] = pd.NA #some sites fill in age_at_scan with one month before DOB
    mask = df['sex'].isna() | (df['age'] < 0) | (df['age'].isna())
    
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(impute_row, index, row, project, columns, log)
            for index, row in df.iterrows()
        ]

        for future in as_completed(futures):
            index, result = future.result()
            for col, val in result.items():
                if val is not None:
                    df.at[index, col] = val
                    print("Updating the value in the dataframe" + str(index) + " " + col + " " + str(val))
    
    #count time elapsed
    log.info(f"Time elapsed: {time.time() - start_time} seconds")

    #Harmonise sex values
    sex_map = {
    'female': 'F', 'male': 'M',
    'f': 'F', 'm': 'M',
    'n/a': 'N/A'
    }
    df['sex'] = df['sex'].fillna('N/A').str.lower().map(sex_map)
    
    print(df['sex'].value_counts())
    #Printing the modified csv to the input directory to be used for plotting
    df_path = os.path.join(work_path,vols)
    df.to_csv(df_path,index=False)
    log.info("File saved..." + df_path)

    #saving those with missing sex/age information to a separate file
    missing_info = df[df.isnull().any(axis=1)][['subject','session','age','sex','acquisition']]

    #if data frame is not empty, save it
    if missing_info.empty == False:
        missing_info.to_csv(os.path.join(output_path,"missing_sex_age_info.csv"),index=False)

    return df_path

def pull_metadata(context,vols):

    """Imputes missing information needed for the plotting functions. 
    Currently it handles sex (Add as needed)

    Returns:
        imputed file
    """
def rename_columns (vols):

    log.info('Renaming ICV columns....')

    input_path = '/flywheel/v0/input/input'
    work_path = '/flywheel/v0/work/'

    df = pd.read_csv(os.path.join(work_path,vols))
    

    name_key_maping = {
    "recon-all":{'total intracranial': 'total intracranial'},
    "synthseg": {'total intracranial': 'total intracranial'},
    "mrr_axireg":{'icv': 'total intracranial'},
    "minimorph":{'icv': 'total intracranial'}}


    for keyword, key in name_key_maping.items():
        if keyword in vols:
            column_mapping = name_key_maping[keyword]
            df.rename(columns=column_mapping,inplace=True)

            log.info('Column has been renamed')
        

    df.columns = df.columns.str.replace('_', ' ').str.replace('-', ' ').str.lower()
    df_path = os.path.join(work_path,vols)

    df.to_csv(df_path,index=False)

    #print(os.path.join(input_path,vols))
    #df.to_csv(os.path.join(input_path,"updated_headers_.csv"),index=False)

    log.info("File saved..." + df_path)
    return df_path

