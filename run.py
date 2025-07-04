#!/usr/bin/env python
import logging
from app.parser import parse_config
from app.main import create_cover_page, parse_csv, create_data_report, merge_pdfs
from datetime import datetime
import os

# import flywheel functions
from flywheel_gear_toolkit import GearToolkitContext


"""The run script.

This script is the entry point for the gear. It is responsible for setting up the
gear environment and executing the main function.
- logging is initialized
- the gear context is created
- the main function is executed

"""

# Initialize the logger
log = logging.getLogger(__name__)

# Define the main function
def main(context: GearToolkitContext) -> None:

    output_dir= "/flywheel/v0/output/"
    work_dir= "/flywheel/v0/work/"

    # Step 0: Parse the configuration file
    user, filepath, input_labels,outlier_thresholds, volumetric_columns, config_context, project,api_key = parse_config(context)
    #age_range, age_min, age_max, age_unit, threshold
    # Step 1: Create the cover page
    cover , age_min, age_max = create_cover_page(user, input_labels, config_context, project,work_dir)
    project_label = project.label
    print(filepath)
    # Step 2: Parse the CSV file
    df, summary_table, filtered_df, n, n_projects, n_sessions, n_clean_sessions, outlier_n, outliers_per_region, project_labels, labels = parse_csv(filepath, outlier_thresholds, volumetric_columns, project_label, config_context)

    # Step 3: Create the data report using the parsed CSV, and the QC csv    
    report = create_data_report(df, summary_table, filtered_df, n, n_projects, n_sessions, n_clean_sessions,  outlier_n, outliers_per_region, project_labels, config_context,output_dir, api_key)
    # qc = generate_qc_report(directory_path, input_labels, output_dir,project_labels)
    # Step 4: Merge cover page and data report
    # Get the current timestamp
    current_timestamp = datetime.now()
    # Format the timestamp as a string
    formatted_timestamp = current_timestamp.strftime('%Y-%m-%d_%H-%M-%S')
    final_report = os.path.join(output_dir, f"{project_label}_{formatted_timestamp}_report.pdf")
    merge_pdfs(project_label, api_key, cover, report, final_report)




# Only execute if file is run as main, not when imported by another module
if __name__ == "__main__":  # pragma: no cover
    # Get access to gear config, inputs, and sdk client if enabled.
    with GearToolkitContext() as gear_context:

        # Initialize logging, set logging level based on `debug` configuration
        # key in gear config.
        gear_context.init_logging()

        # Pass the gear context into main function defined above.
        main(gear_context)
