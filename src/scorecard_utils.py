import pandas as pd
import scorecardpy as sc
import math
import numpy as np
from sklearn.model_selection import train_test_split

def filter_features_by_iv(df: pd.DataFrame, target_col: str = 'TARGET', min_iv: float = 0.02, max_iv: float = 0.50):
    """
    Calculates the Information Value (IV) for all features in the dataset
    and filters them based on industry standard thresholds.
    
    Industry Standards for IV:
    < 0.02 : Useless for prediction
    0.02 to 0.1 : Weak predictor
    0.1 to 0.3 : Medium predictor
    0.3 to 0.5 : Strong predictor
    > 0.5 : Suspicious or too good to be true (possible data leakage)
    
    Args:
        df (pd.DataFrame): The dataset containing features and the target variable.
        target_col (str): The name of the target column.
        min_iv (float): Minimum IV threshold for feature selection.
        max_iv (float): Maximum IV threshold to prevent data leakage.
        
    Returns:
        tuple: A list of selected feature names, and a DataFrame containing IV metrics for all features.
    """
    
    iv_df = sc.iv(df, y=target_col)
    
    iv_df = iv_df.sort_values(by='info_value', ascending=False).reset_index(drop=True)
    
    valid_features_df = iv_df[(iv_df['info_value'] >= min_iv) & (iv_df['info_value'] <= max_iv)]
    selected_features = valid_features_df['variable'].tolist()
    
    suspicious_features = iv_df[iv_df['info_value'] > max_iv]['variable'].tolist()
    useless_features = iv_df[iv_df['info_value'] < min_iv]['variable'].tolist()
    
    print(f"Total features analyzed : {len(iv_df)}")
    print(f"Features selected       : {len(selected_features)} (IV between {min_iv} and {max_iv})")
    print(f"Suspicious features     : {len(suspicious_features)} (IV > {max_iv})")
    print(f"Useless features dropped: {len(useless_features)} (IV < {min_iv})")
    
    return selected_features, iv_df

def get_statistical_sample(df: pd.DataFrame, target_col: str, confidence_level: float = 0.95, margin_of_error: float = 0.01):
    """
    Calculates the statistically significant sample size and returns a stratified sample.
    
    Args:
        df (pd.DataFrame): The full population dataset.
        target_col (str): The target variable for stratification.
        confidence_level (float): 0.95 or 0.99
        margin_of_error (float): 0.01 (1%) or 0.005 (0.5%)
        
    Returns:
        pd.DataFrame: A representative stratified sample of the data.
    """
    N = len(df)
    
    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    Z = z_map.get(confidence_level, 1.96)
    p = 0.5  
    E = margin_of_error
    
    # Cochran's formula for large populations
    n_0 = (Z**2 * p * (1-p)) / (E**2)
    
    n = n_0 / (1 + (n_0 - 1) / N)
    n = math.ceil(n) 
    
    print(f"Population Size (N)   : {N}")
    print(f"Confidence Level      : {confidence_level * 100}%")
    print(f"Margin of Error       : {margin_of_error * 100}%")
    print(f"Calculated Sample Size (n): {n}")
    
    sample_fraction = n / N
    
    _, sample_df = train_test_split(
        df, 
        test_size=sample_fraction, 
        stratify=df[target_col], 
        random_state=42
    )
    
    return sample_df
