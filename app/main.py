from fpdf import FPDF
from datetime import datetime
from PyPDF2 import PdfMerger
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Frame,SimpleDocTemplate, Table, TableStyle, PageBreak, Spacer,  PageTemplate, Frame
from reportlab.lib.utils import ImageReader
import yaml
#import statsmodels.api as sm

import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline
from patsy import dmatrix


import flywheel
import os
import itertools
import warnings
import logging


from utils.format import beautify_report, scale_image, simplify_label, generate_on_page
from utils.outliers import outlier_detection

output_dir ='/flywheel/v0/output/'
workdir = '/flywheel/v0/work/'

log = logging.getLogger(__name__)

# Define the bins and labels
# These have been setup with finer granularity early on due to rapid growth and then coarser granularity later
global bins 
global labels
global range_mapping
global label_mapping
global bins_mapping
global age_min
global age_max
global age_range
global threshold
global age_unit
global growth_curve
global birth_weight_icv

bins = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 24, 30, 36, 48, 60, 72, 84, 96, 108, 120, 144, 168, 192, 216, 252, 300]
labels = ['0-1 month', '1-2 months', '2-3 months', '3-4 months', '4-5 months', '5-6 months',
        '6-8 months', '8-10 months', '10-12 months', '12-15 months', '15-18 months', 
        '18-21 months', '21-24 months', '24-30 months', '30-36 months','3-4 years', 
        '4-5 years', '5-6 years', '6-7 years', '7-8 years', '8-9 years', '9-10 years', 
        '10-12 years', '12-14 years', '14-16 years', '16-18 years', '18-21 years', '21-25 years']


range_mapping =  {"Infants (0-12 months)": (0, 12),
"1st 1000 Days (0-32 months)": (0,32),
"Toddlers (1-3 years)": (12, 36),
"Preschool (3-6 years)": (36, 72),
"School-age Children (6-12 years)": (72, 144),
"Adolescents (12-18 years)": (144, 216),
"Young Adults (18-34 years)": (216, 408),
"Adults (35-89 years)": (420, 1068),  
"All Ages (0-100 years)": (0, 1200) 
}

label_mapping = {
"Infants (0-12 months)": ['0-1 month', '1-2 months', '2-3 months', '3-4 months', '4-5 months', '5-6 months','6-8 months', '8-10 months', '10-12 months'],
"1st 1000 Days (0-32 months)" : ['0-1 month', '1-2 months', '2-3 months', '3-4 months', '4-5 months', '5-6 months','6-8 months', '8-10 months', '10-12 months', '12-15 months', '15-18 months', '18-21 months', '21-24 months', '24-30 months', '30-36 months'],
"Toddlers (1-3 years)": ['12-15 months', '15-18 months', '18-21 months', '21-24 months', '24-30 months', '30-36 months'],
"Preschool (3-6 years)": ['3-4 years','4-5 years', '5-6 years'],
"School-age Children (6-12 years)": ['6-7 years', '7-8 years', '8-9 years', '9-10 years', '10-12 years'],
"Adolescents (12-18 years)": ['12-14 years', '14-16 years', '16-18 years'],
"Young Adults (18-34 years)": ['18-21 years', '21-24 years','25-29 years', '30-34 years'],
"Adults (35-89 years)":['35-39 years', '40-44 years','45-49 years','50-54 years','55-59 years','60-64 years','65-69 years','70-74 years','75-79 years','80-84 years','85-89 years'],
"All Ages (0-100 years)":["0-12 months", "12-36 months", "3-6 years", "6-10 years", "10-12 years", "12-18 years", "18-25 years", "25-34 years", "34-50 years", "50-60 years", "60-70 years", "70-80 years", "80-90 years", "90-100 years"]

}

bins_mapping = {
"Infants (0-12 months)": [0, 1, 2, 3, 4, 5, 6, 8, 10, 12],
"1st 1000 Days (0-32 months)": [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 24, 30, 36],
"Toddlers (1-3 years)": [12, 15, 18, 21, 24, 30, 36],
"Preschool (3-6 years)": [36, 48, 60, 72],
"School-age Children (6-12 years)": [72, 84, 96, 108, 120, 144],
"Adolescents (12-18 years)": [144, 168, 192, 216],
"Young Adults (18-34 years)": [216, 240, 264, 288, 312, 336],
"Adults (35-89 years)": [420, 444, 468, 492, 516, 540, 564, 588, 612, 636, 660, 684, 708, 732, 756, 780, 804, 828, 852, 876, 900, 924, 948, 972, 996, 1020, 1044, 1068],
"All Ages (0-100 years)": [0, 12, 36, 72, 120, 144, 216, 300, 408, 600, 720, 840, 960, 1080, 1200]  # Covers all from 0 months to 100 years
}


# 1. Generate Cover Page
def create_cover_page(user, input_labels, config_context, project,output_dir):

    global bins 
    global labels
    global range_mapping
    global label_mapping
    global bins_mapping
    global age_min
    global age_max
    global age_range
    global threshold
    global age_unit
    global growth_curve
    global birth_weight_icv


    age_min = config_context["age_min_months"]
    age_max = config_context["age_max_months"]
    age_range = config_context["age_range"]
    threshold = config_context["threshold"]
    age_unit = config_context["age_unit"]
    growth_curve = config_context["growth_curve"]
    birth_weight_icv = config_context["birth_weight_icv"]

    if age_range != "":


        age_min = range_mapping[age_range][0] 
        age_max = range_mapping[age_range][1] 
        labels = label_mapping[age_range]
        bins = bins_mapping[age_range]

    log.info(f"Age Min: {age_min} | Age Max: {age_max} | Age Range: {age_range} | Age Bins: {bins}")

    filename = 'cover_page'
    cover = f"{output_dir}{filename}.pdf"

    # Ensure the directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a new PDF canvas
    doc = SimpleDocTemplate(os.path.join(output_dir, f"{filename}.pdf"), pagesize=A4)
    page_width, page_height = A4

    # Styles
    styles = getSampleStyleSheet()
    styleN = styles['Normal']
    styleN.fontSize = 12  # override fontsize because default stylesheet is too small
    styleN.leading = 15
    # Add left and right indentation
    styleN.leftIndent = 20  # Set left indentation
    styleN.rightIndent = 20  # Set right indentation

    # Create a custom style
    custom_style = ParagraphStyle(name="CustomStyle", parent=styleN,
                              fontSize=12,
                              leading=15,
                              alignment=0,  # Centered
                              leftIndent=20,
                              rightIndent=20,
                              spaceBefore=10,
                              spaceAfter=10)


    # Main text (equivalent to `multi_cell` in FPDF)
    text = ("This report provides a detailed summary of the input-derived data. "
            "The data are analyzed by age group and sex. Analyses include the calculation of brain volume z-scores for different age groups, summary descriptive statistics of the total intracranial volume (TICV), and the age distribution in the cohort. of brain volume z-scores for different age groups."
            f"List of outliers has been generated based on z-scores outside of ±{threshold} SD. "
            "Custom options such as age filtering and cubic spline fitting have been applied to the data."
            f"<b><br/><br/>Project Description:</b> {project.description}")
    
    custom_options_text = ( f"<br/><br/>Custom Options used:<br />"
                            f"1. Age Range: {age_min}-{age_max} months<br/>"
                            f"2. Outlier Threshold: ±{threshold} SD<br/>"
                            "3. Cubic Spline Regression Fit: Degree 3<br/>"
                            "4. Confidence Interval: 95% <br/><br/>"
                            f"<i>Input file used: {input_labels['volumetric']}</i>")
    
    stylesheet = getSampleStyleSheet()
    stylesheet.add(ParagraphStyle(name='Paragraph', spaceAfter=20))
    elements = []
    elements.append(Paragraph(text + custom_options_text, stylesheet['Paragraph']))

    # Define a frame for the content to flow into
    margin = 40
    frame = Frame(margin, -60, page_width - 2 * margin, page_height - 2 * margin, id='normal')

    # Define the PageTemplate with the custom "beautify_report" function for adding logo/border
    template = PageTemplate(id='CustomPage', frames=[frame], onPage=generate_on_page(user,project.label,age_min,age_max, age_unit, threshold,input_labels))

    # Build the document
    doc.addPageTemplates([template])
    doc.build(elements)
    log.info("Cover page has been generated.")

    return cover, age_min, age_max


# 2. Parse the volumetric CSV File
def parse_csv(filepath, outlier_thresholds, volumetric_columns, project_label,  config_context):

    """Parse the input CSV file.

    Returns:
        filtered_df (pd.DataFrame): Filtered DataFrame based on age range.
        n (int): Number of observations in the filtered data.
        n_projects (int): Number of unique projects in the filtered data.
        project_labels (list): Unique project labels in the filtered data.
        n_sessions (int): Number of unique sessions in the original data.
        n_clean_sessions (int): Number of unique sessions in the clean data after removing outliers.
        outlier_n (int): Number of participants flagged as outliers based on
    """

    global bins 
    global labels
    global range_mapping
    global label_mapping
    global bins_mapping
    global age_min
    global age_max
    global age_range
    global threshold
    global age_unit
    global growth_curve
    global birth_weight_icv
        
    # Example DataFrame with ages in months
    df = pd.read_csv(filepath) #
    n_sessions = df['session'].nunique()  # Number of unique sessions
    log.info(f"Number of unique sessions processed:  {n_sessions}")

    if age_range != "":
        age_min = range_mapping[age_range][0] 
        age_max = range_mapping[age_range][1] 
        labels = label_mapping[age_range]
        bins = bins_mapping[age_range]
    
    # A simple heuristic: if the maximum value is above a threshold, consider it as days
    #threshold = 100  # adjust based on your dataset context
    
    # if df['age'].max() > threshold:
    #     print("The age values are likely in days.")
    #     # Rename the 'age' column to 'age_in_days'
    #     df.rename(columns={'age': 'age_in_days'}, inplace=True)
    #     df['age_in_months'] = df['age_in_days'] / 30.44
    # else:
    #     print("The age values are likely in months.")
    #     df.rename(columns={'age': 'age_in_months'}, inplace=True)

    if age_unit == "days" :
        df['age_in_months'] = df['age'] / 30.44
    elif age_unit == "weeks":
        df['age_in_months'] = df['age'] / 4
    elif age_unit == "months":
        df['age_in_months'] = df['age']
    elif age_unit == "years" : 
        df['age_in_months'] = df['age'] * 12
            
    df['age_group'] = pd.cut(df['age_in_months'], bins=bins, labels=labels, right=False)

    log.info(f"Age unit specified: {age_unit}")
    log.info(f"NA Sex: {df.sex.isna().sum()}")
    log.info(f"NA Age: {df.age_in_months.isna().sum()}")
    log.info(f"NA TICV: {df['total intracranial'].isna().sum()}")

    # Group by sex and age group
    df['sex'] = df['sex'].fillna('N/A')

    grouped = df.groupby(['sex', 'age_group'])


    # Calculate mean and std for each group
    df['mean_total_intracranial'] = grouped['total intracranial'].transform('mean').fillna(0)
    df['std_total_intracranial'] = grouped['total intracranial'].transform('std').fillna(0)

    # Calculate z-scores
    df['z_score'] = (df['total intracranial'] - df['mean_total_intracranial']) / df['std_total_intracranial']
    # Check if 'project_label' exists, if not, assign a default value
    
    df['project_label'] = project_label  # Or any default value like None

    
    # Calculate other volumes
    # df['total cerebral white matter'] = df['left cerebral white matter'] + df['right cerebral white matter']
    # df['total cerebral cortex'] = df['left cerebral cortex'] + df['right cerebral cortex']
    # df['hippocampus'] = df['left hippocampus'] + df['right hippocampus']
    # df['thalamus'] = df['left thalamus'] + df['right thalamus']
    # df['amygdala'] = df['left amygdala'] + df['right amygdala']
    # df['putamen'] = df['left putamen'] + df['right putamen']
    # df['caudate'] = df['left caudate'] + df['right caudate']


    used_age_groups = [age for age in labels if age in df['age_group'].unique()]
    # Calculate the count of participants per age group``
    age_group_counts = df['age_group'].value_counts().sort_index()

    # Ensure that 'age_group' is treated as a categorical variable with the correct order (only for used categories)
    df['age_group'] = pd.Categorical(df['age_group'], categories=used_age_groups, ordered=True)
    
    # Define the list of columns you want to retain


    
    # volumetric_cols = [ 'supratentorial_tissue',
    #    'supratentorial_csf', 'ventricles', 'cerebellum', 'cerebellum_csf',
    #    'brainstem', 'brainstem_csf', 'left_thalamus', 'left_caudate',
    #    'left_putamen', 'left_globus_pallidus', 'right_thalamus',
    #    'right_caudate', 'right_putamen', 'right_globus_pallidus',
    #    'posterior_callosum', 'mid_posterior_callosum', 'central_callosum',
    #    'mid_anterior_callosum', 'anterior_callosum', 'icv']

    # Define the columns to keep in the DataFrame

    columns_to_keep = ['project_label', 'subject',	'session',	'age_in_months', 'sex',	'acquisition',"age_group", "z_score"]  + volumetric_columns

    # Define thresholds for outlier detection


    
    # # define z-score thresholds 
    # zscore_thresholds  = {}
    # for col in volumetric_cols:
    #     zscore_thresholds[col] = 2  # Set a z-score threshold of 2 for all volumetric columns

    # Retrieve outliers using the outlier_detection function
    df, outliers_df = outlier_detection(df[columns_to_keep], age_column = 'age_in_months',volumetric_columns=volumetric_columns, cov_thresholds = outlier_thresholds, zscore_thresholds = outlier_thresholds)

    # Save the filtered DataFrame to a CSV file
    outliers_df.to_csv(os.path.join(output_dir,'outliers_list.csv'), index=False)




    outlier_n = outliers_df['session'].nunique()


    outliers_per_region = {}
    for region, threshold in outlier_thresholds.items():
        outliers_per_region[region] = len(outliers_df[outliers_df[region].abs() > threshold])
    outliers_per_region = pd.DataFrame.from_dict(outliers_per_region, orient='index')


    # Step 3: Create a clean DataFrame by excluding the outliers
    clean_df = df[df['is_outlier'] == False]
    clean_df.drop(columns = ['is_outlier'], inplace=True)


    n_clean_sessions = clean_df['session'].nunique()  # Number of unique sessions in the clean data
    #print(clean_df['session'].nunique())
    #print(clean_df.shape)

    # Optional: Save the clean DataFrame to a CSV file
    clean_df.to_csv(os.path.join(workdir,'clean_data.csv'), index=False)


    # Set limit for the age range to be included in the analysis
    upper_age_limit = int(age_max)
    lower_age_limit = int(age_min) 

    # Filter the data to include only observations up to the requested limit
    filtered_df = clean_df[(clean_df['age_in_months'] <= upper_age_limit) & (clean_df['age_in_months'] >= lower_age_limit)]

    n = len(filtered_df)  # Number of observations in the filtered data
    print("@@@@ Number of observations in the filtered data:", n)
    if n == 0:
        log.warning("No data available after filtering. Please check the age range and input data.")
        #use the max and min of age_in_months instead 
        filtered_df = clean_df
        upper_age_limit = int(clean_df['age_in_months'].max())
        lower_age_limit = int(clean_df['age_in_months'].min())

        #update the new values
        age_max = upper_age_limit
        age_min = lower_age_limit

        n = len(filtered_df)  # Recalculate number of observations in the filtered data
    

    n_projects = filtered_df['project_label'].nunique()  # Number of unique projects in the filtered data
    project_labels = filtered_df['project_label'].unique()  # Unique project labels in the filtered data

    # --- Generate a summary report with plots and tables --- #

    # Calculate the count (n) for each age group
    age_group_counts = clean_df['age_group'].value_counts().sort_index()
    # Filter out age groups with a count of 0
    age_group_counts = age_group_counts[age_group_counts > 0]


    # Group by sex and age group and calculate the necessary statistics
    summary_table = clean_df.groupby(['age_group', 'sex']).agg({
        'subject': 'nunique',  # Count the number of unique participants
        'session': 'nunique',  # Count the number of unique sessions
        'total intracranial': ['mean', 'std']  # Mean and std of brain volume
    }).reset_index()

    # Remove rows where the mean of 'total intracranial' is NaN
    summary_table = summary_table.dropna(subset=[('total intracranial', 'mean')])
    # Pivot the table to have Sex as columns and Age Group as a single row index
    summary_table = summary_table.pivot(index='age_group', columns='sex')

  
    # Flatten the multi-level columns
    summary_table.columns = ['_'.join(col).strip() for col in summary_table.columns.values]

    # Reset index to make 'age_group' a column
    summary_table.reset_index(inplace=True)

    # Renaming columns for better readability
    summary_table.rename(columns={"age_group":"Age Group","subject_nunique_F":"n sub (F)","subject_nunique_M":"n sub (M)","subject_nunique_N/A":"n sub (N/A)",
                                  "session_nunique_M":'n ses (M)', "session_nunique_F":'n ses (F)', "session_nunique_N/A":'n ses (N/A)',
                                  "total intracranial_mean_M": 'Mean TICV (M)', "total intracranial_mean_F":'Mean TICV (F)', "total intracranial_mean_N/A":'Mean TICV (N/A)',
                                  "total intracranial_std_M":'Std TICV (M)',"total intracranial_std_F":'Std TICV (F)',"total intracranial_std_N/A":'Std TICV (N/A)'},inplace=True)
                                  

    # Round the numerical columns to 2 decimal places
    summary_table = summary_table.round(2)
    summary_table.to_csv(os.path.join(output_dir,'summary_table.csv'),index=False)


    #### Plotting outliers #####
    #outliers_df
    sns.kdeplot(df.loc[df['is_outlier'] == False, 'total intracranial'], label='Non-Outliers')
    sns.kdeplot(df.loc[df['is_outlier'] == True, 'total intracranial'], label='Outliers', color='red')
    plt.title('Distribution of TICV of outliers vs non-outliers')
    plt.legend()
    plt.savefig(os.path.join(workdir, "outlier_icv_plot.png"))

    # Count missing values
    #missing_counts = df[["age_in_months", "sex"]].isna().sum()
    
    # Plot
    # plt.figure(figsize=(6, 4))
    # missing_counts.plot(kind="bar", color="#D96B6B", linewidth=1)

    # # Styling
    # plt.ylabel("Number of observations")
    # plt.title("Missing metadata")
    # plt.xticks(rotation=0)  # Keep labels horizontal

    # # Add explanation text below the plot
    # plt.figtext(0.13, 0.28, 
    #             "Missing sex and age information affects the usability of this report.\n"
    #             "Please ensure the metadata is available in your project.\nn"
    #            )  # Added padding for better spacing


    # plt.savefig(os.path.join(output_dir, "missing_metadata.png"))


    return df, summary_table, filtered_df, n, n_projects, n_sessions, n_clean_sessions, outlier_n, outliers_per_region, project_labels, labels


# 3. Generate the Data Report
def create_data_report(df, summary_table, filtered_df, n, n_projects, n_sessions, n_clean_sessions, outlier_n, outliers_per_region, project_labels, config_context,output_dir,api_key):

    """Generate a data report with multiple plots and a summary table in a PDF format.

    Returns: report filename
        
    """

    global bins 
    global labels
    global range_mapping
    global label_mapping
    global bins_mapping
    global age_min
    global age_max
    global age_range
    global threshold
    global age_unit
    global growth_curve
    global birth_weight_icv

    filename = "data_report"
    report = f'{workdir}{filename}.pdf'
    pdf = canvas.Canvas((f'{workdir}{filename}.pdf') )
    a4_fig_size = (8.27, 11.69)  # A4 size
    # Define the page size
    page_width, page_height = A4
   
    # --- Plot 1: Boxplot of all Z-Scores by Age Group with Sample Sizes --- #

    # Drop observations where 'age_group' is NaN
    df = df.dropna(subset=['age_group'])

    used_age_groups = [age for age in labels if age in df['age_group'].unique()]

    # Ensure that 'age_group' is treated as a categorical variable with the correct order (only for used categories)
    df['age_group'] = pd.Categorical(df['age_group'], categories=used_age_groups, ordered=True)

    # Calculate the count of participants per age group
    age_group_counts = df['age_group'].value_counts().sort_index()

    # Create new labels with counts
    age_group_labels = [f"{label}\n(n={age_group_counts[label]})" for label in used_age_groups]


    # Dynamically adjust font size based on the number of labeé&ls
    n_labels = len(used_age_groups)
    font_size = max(6, 8 - n_labels // 3)  # Scale the font size down as the number of labels increases. [Not used]

    # Create figure with full A4 size using plt.figure() (not plt.subplots)
    fig = plt.figure(figsize= a4_fig_size)

    # Define the position and size of the smaller figure within the A4 page
    # The numbers in add_axes([left, bottom, width, height]) are relative to the figure size, between 0 and 1
    ax = fig.add_axes([0.125, 0.5, 0.8, 0.4])  # Left, bottom, width, height (adjust these as needed)

    # Set the plot size and create the boxplot
    # fig, ax = plt.subplots(figsize=a4_fig_size)
    # sns.boxplot(x='age_group', y='z_score', data=df, ax=ax, order=used_age_groups, palette='Set2', legend=False, hue='age_group')
    # ax.set_title('Z-Scores by Age Group')
    # ax.set_xlabel('Age Group')
    # ax.set_ylabel('Z-Score')
    
    # # Set x-axis tick labels to show the age group labels in the correct order
    # ax.set_xticklabels(age_group_labels, rotation=45)
    # plt.setp(ax.get_xticklabels(), rotation=45, fontsize=10)  # Shift labels slightly to the left
    # ax.grid(True)

    outliers_per_region[0].plot(kind='bar')
    ax.set_xlabel('Region')
    ax.set_ylabel('Number of Outliers')
    ax.set_title('Number of Outliers per Region')
    ax.set_xticklabels(rotation=45)
    plt.setp(ax.get_xticklabels(), rotation=45, fontsize=10)  # Shift labels slightly to the left

    plt.show()


    # Add explanation text below the plot
    plt.figtext(0.08, 0.30, 
                "This boxplot displays how many outliers were flagged per region.\n"
                "Each bin represent how many values were above or below the \n"
                f"threshold for that region when transformed using the covariance approach\n"
                f"Unique sessions: N = {n_sessions}."
                # f"Number of sessions after removing outliers = {n_clean_sessions}\n"
                f"\n{outlier_n} session(s) fell outside the thresholds and were flagged for further review.",
                wrap=True, horizontalalignment='left', fontsize=12,
                bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 11})  # Added padding for better spacing

    # Adjust layout to ensure no overlap
    plt.subplots_adjust(top=0.85, bottom=0.4)  # Adjust to fit title and text properly
    # Save the plot only
    zscore_plot_path = os.path.join(workdir, "zscores_agegroup_plot.png")
    #plt.tight_layout()
    plt.savefig(zscore_plot_path)
    plt.close()

  
    # --- Plot 2: Summary Table of all Participants --- # 
    summary_table = pd.read_csv(os.path.join(output_dir,'summary_table.csv'))    
    summary_table.fillna(0,inplace=True)

    sexes = ["F","M","N/A"]
    long_rows = []

    prev_age_group = None
    for idx, row in summary_table.iterrows():
        current_age_group = row['Age Group']
        for sex in sexes:
            
            if current_age_group == prev_age_group:
                age_group = ""
            else:
                prev_age_group = current_age_group
                age_group  = current_age_group

            long_rows.append({
                'Age Group': age_group,
                'Sex': sex,
                'n sub' : str(int(row.get(f"n sub ({sex})", "0"))),
                'n ses': str(int(row.get(f"n ses ({sex})", "0"))),
                #'n (subs/ses)': str(int(row.get(f"n sub ({sex})", "0")))+" / "+str(int(row.get(f"n ses ({sex})", "0"))),
                'Mean TICV': row.get(f"Mean TICV ({sex})", 0),
                'Std TICV': row.get(f"Std TICV ({sex})", 0),
            })

            

    # Create a new long-format DataFrame
    long_summary_table = pd.DataFrame(long_rows)

    nsub_cols = [col for col in long_summary_table.columns if col.lower().startswith('n sub')]



    # Drop rows where any of those columns == 0
    long_summary_table = long_summary_table[~(long_summary_table[nsub_cols] == "0").any(axis=1)]

    ######

    fig = plt.figure(figsize=(9, 5))  # smaller now!
    ax = fig.add_axes([0.05, 0.3, 0.9, 0.6])
    ax.axis('tight')
    ax.axis('off')



    #long_summary_table = summary_table
    table = ax.table(
        cellText=long_summary_table.values,
        colLabels=long_summary_table.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style as before
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            # Header row
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        else:
            # All data rows
            cell.set_facecolor('#f0f0f0')

            if col_idx == 0:
                # First column in data rows
                cell.set_text_props(weight='bold', color='#303070')  # or any color you want

    # Add title and footnote
    #ax.set_title('Summary Descriptive Statistics', fontdict={'fontsize': 13, 'weight': 'bold'})
    #fig.suptitle('Summary Descriptive Statistics', fontsize=13, fontweight='bold')

    plt.tight_layout()
    table_plot_path = os.path.join(workdir, "descriptive_stats_long.png")
    plt.savefig(table_plot_path, bbox_inches='tight')

    # --- Plot 3: Histogram of Z-Scores --- #
    
    # Create figure with full A4 size using plt.figure() (not plt.subplots)
    fig = plt.figure(figsize=a4_fig_size)

    # Define the position and size of the smaller figure within the A4 page
    # The numbers in add_axes([left, bottom, width, height]) are relative to the figure size, between 0 and 1
    ax = fig.add_axes([0.125, 0.5, 0.8, 0.4])  # Left, bottom, width, height (adjust these as needed)

    # fig, ax = plt.subplots(figsize=a4_fig_size)
    sns.histplot(filtered_df['age_in_months'], bins=20, kde=True, ax=ax)
    ax.set_title('Distribution of Age in Months')
    ax.set_xlabel('Age (months)')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    
    # Add explanation text below the plot
    plt.figtext(0.15, 0.35, "This plot shows the distribution of participant ages in months.\n"
                        "The KDE curve provides a smoothed estimate of the age distribution.\n"
                        f"Plot limits set to {age_min}-{age_max} months, N = {n}.\n"
                        f"Included projects = {', '.join(project_labels)}",
                wrap=True, horizontalalignment='left', fontsize=12,
                bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 15})  # Added padding for better spacing

    # Adjust layout to ensure no overlap
    plt.subplots_adjust(top=0.85, bottom=0.2)  # Adjust to fit title and text properly
    age_plot_path = os.path.join(workdir, "agedist_plot.png")
    #plt.tight_layout()
    plt.savefig(age_plot_path)
    plt.close()

    padding = 60  # Space between plots

    scaled_width1, scaled_height1 = scale_image(zscore_plot_path, 500, 500)
    
    plot1_x = (page_width - scaled_width1) / 2  # Centered horizontally
    plot1_y = page_height - scaled_height1 - padding #- 50

    #NO longer drawing the summary table. This will be output as a csv
    #Age distribution plot
    scaled_width2, scaled_height2 = scale_image(age_plot_path, 500, 500)
    plot2_x = (page_width - scaled_width2) / 2  # Centered horizontally
    plot2_y = plot1_y - scaled_height2 + 100  #- plot1_y - padding 


    
    pdf.drawImage(zscore_plot_path, plot1_x, plot1_y, width= scaled_width1, height=scaled_height1)
    pdf.drawImage(age_plot_path, plot2_x, plot2_y, width= scaled_width2, height=scaled_height2)   # Position plot higher on the page
    
    pdf = beautify_report(pdf,False,True)
    pdf.showPage()

    

    # --- Plot 4: Polynomial fit with degree 3 (cubic) using sns.regplot --- #
    # Create figure with full A4 size using plt.figure() (not plt.subplots)
    fig = plt.figure(figsize=a4_fig_size)
    ax = fig.add_axes([0.125, 0.5, 0.8, 0.4])  # Left, bottom, width, height (adjust these as needed)
    ax.grid(True)

    print(filtered_df['sex'].value_counts())

    for sex in filtered_df['sex'].unique():
        df_sex = filtered_df[filtered_df['sex'] == sex]
        if df_sex.shape[0] < 5:  # fewer than your n_knots (5) → skip
            print(f"Skipping {sex}: only {df_sex.shape[0]} samples")
            continue

        X = df_sex[['age_in_months']].values
        y = df_sex['total intracranial'].values

        model = make_pipeline(
            SplineTransformer(degree=3, n_knots=5, include_bias=False),
            LinearRegression()
        )
        model.fit(X, y)

        X_test = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        y_pred = model.predict(X_test)

        plt.scatter(X, y, alpha=0.5, label=f"{sex} data")
        plt.plot(X_test, y_pred, label=f"{sex} spline fit", linewidth=2)

    plt.xlabel("Age (Months)")
    plt.ylabel("Brain Volume vs. Age, split by Sex")
    plt.title("Cubic Spline Regression by Sex")
    plt.legend()
    # Add explanation text below the plot
    plt.figtext(0.06, 0.28, f"This scatter plot shows the relationship between age and total intracranial volume (TICV),\n"
                            f"with a spline-based regression model (B-splines, degree 3) fitted separately for each sex.\n"
                            f"The smoothed trends allow for non-linear changes in TICV across age, and individual data points "
                            f"are overlaid to illustrate distribution.\n"
                            f"Data points outside the initial study {threshold} IQR range are excluded from the plot.\n\n"
                            f"Plot limits set to {age_min}-{age_max} months, N = {n}.\n"
                            f"Included projects = {', '.join(project_labels)}",
                wrap=True, horizontalalignment='left', fontsize=12,
                bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 15})  # Added padding for better spacing

    # Adjust layout to ensure no overlap
    plt.subplots_adjust(top=0.85, bottom=0.2)  # Adjust to fit title and text properly
    plt.legend()
    scatter_plot_path = os.path.join(workdir, "ageVol_scatter_plot.png")
    plt.savefig(scatter_plot_path)
    plt.close()


    # volumetrics_to_plot = ['total cerebral white matter', 'total cerebral cortex', 'hippocampus', 
    #                'thalamus', 'amygdala', 'putamen', 'caudate']

    
    scaled_width1, scaled_height1 = scale_image(scatter_plot_path, 500, 500)
    
    plot1_x = (page_width - scaled_width1) / 2  # Centered horizontally
    plot1_y = page_height - scaled_height1 - 50 #- 50

    scaled_width2, scaled_height2 = scale_image(scatter_plot_path, 500, 500)
    plot2_x = (page_width - scaled_width2) / 2  # Centered horizontally
    plot2_y = plot1_y - (scaled_height2 / 2) - 100 #- plot1_y - padding 


    pdf.drawImage(scatter_plot_path, plot1_x, plot1_y, width=scaled_width1, height=scaled_height1)   # Position plot higher on the page    
    #pdf.drawImage(scatter_plot_path, plot2_x, plot2_y, width= scaled_width2, height=scaled_height2)   # Position plot higher on the page    

    pdf = beautify_report(pdf,False,True)
    pdf.showPage()    
    

    # --- Plot 5: Growth curves--- #

    if growth_curve:

        var = 'birth weight kg'
        var2 = 'total intracranial'

        # Define birth weight bins and labels
        bw_bins = [0, 2.5, 3.5, df[var].max()]
        bw_labels = ['Low (<2.5 kg)', 'Normal (2.5–3.5 kg)', 'High (>3.5 kg)']

        # Assign birth weight groups
        df['bw_group'] = pd.cut(df[var], bins=bw_bins, labels=bw_labels)

        palette = sns.color_palette("Set2")

        # Prepare legend elements
        bin_legend_elements = [Patch(facecolor=col, label=label) for col, label in zip(palette, bw_labels)]
        scatter_legend_elements = []

        # Initialize the plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='age_in_months', y=var2, hue='bw_group', estimator=None, units='subject', lw=1, alpha=0.1)

        # Fit spline and scatter per group
        for i, group in enumerate(bw_labels):
            sub_df = df[df['bw_group'] == group].dropna(subset=["age", var2])
            if len(sub_df) < 5:
                print(f"Skipping group '{group}' due to insufficient data ({len(sub_df)} rows).")
                continue

            try:
                color = palette[i]
                n_samples = len(sub_df)

                # Plot scatter
                plt.scatter(sub_df["age"], sub_df[var2], color=color, alpha=0.05)
                

                # Add visible patch for legend
                scatter_legend_elements.append(
                    Patch(facecolor=color, edgecolor='black', label=f"{group} (n={n_samples})", alpha=0.8)
                )

                # Fit and plot spline
                x = dmatrix("bs(age, df=4)", {"age": sub_df["age"]}, return_type='dataframe')
                model = LinearRegression().fit(x, sub_df[var2])
                #model = sm.OLS(sub_df[var2], x).fit()
                pred_x = np.linspace(sub_df["age"].min(), sub_df["age"].max(), 100)
                pred_x_spline = dmatrix("bs(age, df=4)", {"age": pred_x}, return_type='dataframe')

                plt.plot(pred_x, model.predict(pred_x_spline), linewidth=2, linestyle='--', color=color)

            except Exception as e:
                print(f"Error fitting spline for group '{group}': {e}")

        # Combine legend entries (scatter with n + bin info)
        fig = plt.gcf()
        combined_legend = scatter_legend_elements 
        fig.legend(
            handles=combined_legend,
            title="Birth Weight Groups",
            loc="center right",
            bbox_to_anchor=(1.25, 0.5),
            borderaxespad=0.0,
            frameon=True
        )

        plt.gca().get_legend().remove()


        # Final plot labels
        plt.title("ICV Growth by Birth Weight Group")
        plt.xlabel("Age (Months)")
        plt.ylabel(var2.title())
        #plt.tight_layout()

        growth_plot_path = os.path.join(workdir, "birthWeight_icv_growthCurve.png")
        plt.savefig(growth_plot_path,bbox_inches='tight')



        scaled_width1, scaled_height1 = scale_image(growth_plot_path, 500, 500)
    
        plot1_x = (page_width - scaled_width1) / 2  # Centered horizontally
        plot1_y = page_height - scaled_height1 - 50 #- 50


        pdf.drawImage(growth_plot_path, plot1_x, plot1_y, width=scaled_width1, height=scaled_height1)   # Position plot higher on the page    

        pdf = beautify_report(pdf,False,True)
        pdf.showPage()    
    

    pdf.save()  # Save the PDF
    print("PDF summary report has been generated.")
    return report

# 5. Merge the Cover Page and Data Report
def merge_pdfs(project_label, api_key, cover, report, final_report):
    merger = PdfMerger()

    fw = flywheel.Client(api_key=api_key)

    project  = fw.projects.find_one(f'label={project_label}')
    project = project.reload() 
   
    print("Merging the cover page and data report...")
    print("Cover Page: ", cover)
    print("Data Report: ", report)
    print("Final Report: ", final_report)

    # Append the cover page
    merger.append(cover)

    # Append the data report
    merger.append(report)

    # Write to a final PDF
    merger.write(final_report)
    merger.close()


    custom_name = final_report.split('/')[-1]
    project.upload_file(final_report, filename=custom_name,classification="Report")
    log.info("Report has been uploaded to the project's information tab.")


    
    