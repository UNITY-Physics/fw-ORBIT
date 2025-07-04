import pandas as pd
import numpy as np


def covariance_difference(data):
    """
    Compute the difference between each point and the covariate prediction.

    Parameters:
    - data: DataFrame with standardized values.
    - method: Method to use for outlier detection ('covariance' or 'linear_regression').

    Returns:
    - DataFrame with an differences between actual z-score and predicted z-score.
    """


    #compute covariance matrix
    newdata = np.array([]).reshape(0, data.shape[0])
    

    # compute the "predicted" values for each column based on differences from mean and covariance matrix
    for i in range(len(data.T)):

        copy = data.copy()

        # include only covariance that does not include the i-th column
        cov_matrix_i = np.cov(copy.T)

        #return cov_matrix_i
        cov_column_i = cov_matrix_i.T[i,:]
        cov_column_i = np.delete(cov_column_i, i).reshape(-1,1)



        #drop the i-th column of the data
        copy = copy.drop(copy.columns[i], axis=1)


        # compute the mean of the i-th column based on the covariance with the other columns
        # the i-th column is the dot product of the covariance with the other columns
        #  divided by the number of columns to normalize
        newdata = np.vstack([newdata, np.dot(copy, cov_column_i).squeeze()/copy.shape[1]])

    # return the difference between the original data and the predicted values
    
    return data - newdata.T

def threshold_outlier_detection(data,skip_covariance = False, thresholds: dict = {}):
    """
    Detect outliers using covariance method.
    
    Parameters:
    - data: DataFrame with standardized values.
    - kwargs: Additional parameters for the method.
    
    Returns:
    - DataFrame with outliers detected.
    """

    if not skip_covariance:
        cov_differences = covariance_difference(data)
    else:
        cov_differences = data.copy()
    #cov_differences.set_index(data.index.values, inplace=True)
    #cov_differences["subject"] = data["subject"].values
    
    for key, value in thresholds.items():
        if key not in cov_differences.columns:
            raise ValueError(f"Key '{key}' not found in the DataFrame columns.")
        
        if "outliers" not in cov_differences.columns:
            cov_differences["outliers"] = cov_differences[key].apply(lambda x: 1 if x > value or x < -value else 0)

        else:
            outliers = cov_differences[key].apply(lambda x: 1 if x > value or x < -value else 0)
            cov_differences["outliers"] = cov_differences["outliers"] + outliers

    
        
    return cov_differences[list(thresholds.keys())+["outliers"]].reset_index(drop=True)
        

def outlier_detection(df: pd.DataFrame, age_column: str, volumetric_columns: list, cov_thresholds: dict = {},zscore_thresholds: dict = {}) -> pd.DataFrame:
    """
    Detect outliers in a DataFrame using various methods.
    
    Parameters:
    - df: pd.DataFrame, the input data.
    - age_column: str, the name of the age column to analyze.
    - volumetric_columns: list, the names of the volumetric columns to analyze.
    - method: str, the method to use for outlier detection. Options are 'zscore', 'pca', 'lof', 'isolation_forest'.
    - plot: bool, whether to plot the results.
    - explained: bool, whether to compute explainable differences.
    - kwargs: dict, additional parameters for the chosen method.
    -   For 'zscore': {'threshold': float} - the z-score threshold for outlier detection.
    -   For 'lof': {'n_neighbors': int} - the number of neighbors for
        Local Outlier Factor.
    -   For 'isolation_forest': {'contamination': float} - the proportion of outliers in the data.
    -   For 'pca': {'n_components': int, 'threshold': float} - number of PCA components and threshold for outlier detection.
    -   For 'mahalanobis': {'threshold': float} - the threshold for Mahalanobis distance outlier detection.
    -   For 'explain_method': {'method': str} - the method to use for explainable differences, options are 'covariance' or 'linear_regression'.
    
    Returns:
    - outliers: pd.DataFrame, the detected outliers.
    """
    df = df.copy()
    outliers = pd.DataFrame()
    z_score_agg = pd.DataFrame()

    #Perform outlier detection for each age group
    for age in df[age_column].unique():
        age_df = df[df[age_column] == age]


        if not age_df.empty:
            # perform z-score normalization
            z_scores = (age_df[volumetric_columns] - age_df[volumetric_columns].mean()) / age_df[volumetric_columns].std()
            


            # perform outlier detection based on the covariance
            outliers_grouped = threshold_outlier_detection(z_scores, thresholds=cov_thresholds)
            zscore_outliers = threshold_outlier_detection(z_scores, skip_covariance=True, thresholds=zscore_thresholds)
            outliers_grouped["zscore_outliers"] = zscore_outliers["outliers"]
            outliers_grouped["subject"] = age_df["subject"].values
            outliers_grouped["session"] = age_df["session"].values


            # filter to keep only rows with outliers
            outliers_grouped = outliers_grouped[outliers_grouped["outliers"] > 0]


        if not outliers.empty:
            outliers = pd.concat([outliers, outliers_grouped], ignore_index=True)
        else:
            outliers = outliers_grouped
        
        # Aggregate z-scores for all ages

        # Not used, but can be useful for further analysis
        z_scores['subject'] = age_df['subject']

        if not z_score_agg.empty:
            # Aggregate z-scores for all ages
            z_score_agg = pd.concat([z_score_agg, z_scores], ignore_index=True)
        else:
            z_score_agg = z_scores.copy()
    
    #flag outliers
    outliers["is_outlier"] = True
    tag_only = outliers[["subject", "is_outlier"]].drop_duplicates()
    df = df.copy().merge(tag_only, how='left', on='subject')
    df['is_outlier'] = df['is_outlier'].fillna(0).astype(bool)

    # cleanup
    df.drop(columns = ["Unnamed: 0"], inplace=True, errors='ignore')
    outliers.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    
    return df, outliers
