"""Parser module to parse gear config.json."""

from typing import Tuple
import os
import re
from flywheel_gear_toolkit import GearToolkitContext
import flywheel
import pandas as pd
import logging

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

    age_min = context.config.get("age_min")
    age_max = context.config.get("age_max")
    age_range = context.config.get("age_range")
    threshold = context.config.get("threshold")

    # -------------------  Get Input label -------------------  #

    # Specify the directory you want to list files from
    input_path = '/flywheel/v0/input/input'
    # List all files in the specified directory
    #recon-all output, QC output , [add more as needed]
    input_labels = {} 
    name_key_maping = {
    "recon-all-clinical": "volumetric",
    "synthseg": "volumetric",
    "mrr_axireg": "volumetric"
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

    print("Input files found: ", input_labels)

    impute_information(context,input_labels['volumetric'])
    rename_columns (input_labels['volumetric'])

    return user, df, input_labels, age_range, age_min, age_max, threshold, project_container, input_path, api_key


def impute_information(context,vols):

    """Imputes missing information needed for the plotting functions. 
    Currently it handles sex (Add as needed)

    Returns:
        imputed file
    """

    
    api_key = context.get_input("api-key").get("key")
    fw = flywheel.Client(api_key=api_key)

    input_container = context.client.get_analysis(context.destination["id"])
    proj_id = input_container.parents["project"]
    project_container = context.client.get(proj_id)
    project_label = project_container.label
    
    project = fw.projects.find_first(f'label="{project_label}"')

    input_path = '/flywheel/v0/input/input'
    output_path = '/flywheel/v0/output'
    df = pd.read_csv(os.path.join(input_path,vols))
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


    # Check if the sex or age column has empty values
    columns = ['sex','age']
    #has_empty_values = df[column_name].isnull().any()

    

    #For every session get the sex information for those with missing
    for index, row in df.iterrows():
        subject_label = row['subject']
        session_label = row['session']
        # Find the subject by label
        subject = next((s for s in project.subjects() if s.label == subject_label), None)

        
        subject = subject.reload()
        session = next((s for s in subject.sessions() if s.label == session_label), None)
        session = session.reload()

        for column_name in columns:
            if column_name == 'sex':
                df.at[index, column_name] = subject.sex
                print(f"Imputing {column_name} for subject: {subject_label}: {subject.sex}")
            else:
                #get either session information (age_months) session
                if session.info != {}:
                    key_with_age = [key for key in session.info if 'age' in key]
                    age = session.info[key_with_age[0]]
                    df.at[index, column_name] = age
                    
                    print(f"Imputing {column_name} for subject: {subject_label}: {age}")
                else:  
                    df.at[index, column_name] = session.age_years * 12 #Flywheel session age is in years, convert to months
                    print(f"Imputing {column_name} for subject: {subject_label}: {session.age_years * 12}")

    #Printing the modified csv to the input directory to be used for plotting
    df.to_csv(os.path.join(input_path,vols),index=False)

    #saving those with missing sex/age information to a separate file
    missing_info = df[df.isnull().any(axis=1)][['subject','session','age','sex','acquisition']]

    #if data frame is not empty, save it
    if missing_info.empty == False:
        missing_info.to_csv(os.path.join(output_path,"missing_sex_age_info.csv"),index=False)

def rename_columns (vols):

    print('RENAMING COLUMNS....')

    input_path = '/flywheel/v0/input/input'
    df = pd.read_csv(os.path.join(input_path,vols))
    

    name_key_maping = {
    "recon-all":{'total intracranial': 'total intracranial'},
    "synthseg": {'total intracranial': 'total intracranial'},
    "mrr_axireg":{'icv': 'total intracranial'}}


    for keyword, key in name_key_maping.items():
        if keyword in vols:
            column_mapping = name_key_maping[keyword]
            df.rename(columns=column_mapping,inplace=True)
            print('Column has been renamed')
        

    df.columns = df.columns.str.replace('_', ' ').str.replace('-', ' ').str.lower()
    df.to_csv(os.path.join(input_path,vols),index=False)
    print(os.path.join(input_path,vols))
    df.to_csv(os.path.join(input_path,"updated_headers.csv"),index=False)

    print("file saved...")

