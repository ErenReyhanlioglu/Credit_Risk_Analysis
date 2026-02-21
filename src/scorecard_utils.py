import pandas as pd
import scorecardpy as sc
import math
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

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

def run_feature_health_check(binning_results):
    """
    Analyzes feature health while strictly excluding summary rows.
    Fixed ValueError by enforcing string casting for comparisons.
    """
    health_data = []

    for col, optb in binning_results.items():
        bt = optb.binning_table.build()
        
        bin_labels = bt['Bin'].astype(str)
        
        is_missing = bin_labels.str.contains("Missing", na=False, case=False)
        is_special = bin_labels.str.contains("Special", na=False, case=False)
        is_total = bin_labels.str.contains("Total", na=False, case=False) | (bin_labels == "") | (bin_labels == "nan")
        
        missing_rate = bt[is_missing]['Count (%)'].values[0] if any(is_missing) else 0
        
        actual_bins = bt[~(is_missing | is_special | is_total)].copy()
        
        max_dist = actual_bins['Count (%)'].max() if not actual_bins.empty else 0
        bin_count = len(actual_bins)
        iv = optb.binning_table.iv
        status = optb.status
        
        reasons = []
        if missing_rate > 0.50: reasons.append("High Missing")
        if max_dist > 0.85: reasons.append("Over-Concentrated")
        if status != "OPTIMAL": reasons.append("Non-Optimal Trend")
        if bin_count < 2: reasons.append("Insufficient Bins")
        
        health_status = "STABLE" if not reasons else "CRITICAL: " + ", ".join(reasons)
        
        health_data.append({
            "Feature": col,
            "IV": round(iv, 4),
            "Missing_Rate": round(missing_rate, 4),
            "Max_Bin_Dist": round(max_dist, 4),
            "Bin_Count": bin_count,
            "Health_Score": health_status
        })

    return pd.DataFrame(health_data).sort_values(by="IV", ascending=False)

def get_health_summary(df_health):
    print("\n" + "="*50)
    print("      FEATURE STABILITY AUDIT SUMMARY")
    print("="*50)
    summary = df_health["Health_Score"].value_counts()
    print(summary)
    print("-" * 50)
    
    critical_df = df_health[df_health["Health_Score"] != "STABLE"]
    return critical_df

import pandas as pd
import numpy as np

def filter_by_correlation(df_woe, binning_results, threshold=0.80):
    """
    Removes highly correlated features by keeping the one with the highest IV.
    
    Args:
        df_woe (pd.DataFrame): WoE transformed features (including TARGET).
        binning_results (dict): Dictionary containing fitted OptimalBinning objects.
        threshold (float): Correlation threshold (default 0.80).
        
    Returns:
        list: List of uncorrelated features.
    """
    features = [col for col in df_woe.columns if col != 'TARGET']
    corr_matrix = df_woe[features].corr().abs()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    
    for col in upper.columns:
        correlated_with_col = upper.index[upper[col] > threshold].tolist()
        
        for correlated_feat in correlated_with_col:
            iv_col = binning_results[col].binning_table.iv
            iv_corr = binning_results[correlated_feat].binning_table.iv
            
            if iv_col >= iv_corr:
                if correlated_feat not in to_drop:
                    to_drop.add(correlated_feat)
            else:
                if col not in to_drop:
                    to_drop.add(col)
                    
    final_features = [f for f in features if f not in to_drop]
    
    print(f"Initial features: {len(features)}")
    print(f"Features dropped due to high correlation: {len(to_drop)}")
    print(f"Remaining features: {len(final_features)}")
    
    return final_features

def calculate_vif(df_woe, feature_list):
    """
    Calculates VIF for a given list of features.
    
    Args:
        df_woe (pd.DataFrame): WoE transformed dataframe.
        feature_list (list): List of features to check for VIF.
        
    Returns:
        pd.DataFrame: VIF values for each feature.
    """
    # VIF requires a constant/intercept, but since we use WoE, 
    # the relative differences are what matters. 
    # We create a temporary X dataframe.
    X = df_woe[feature_list].copy()
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                       for i in range(len(X.columns))]
    
    return vif_data.sort_values(by="VIF", ascending=False)

def fit_logistic_regression(X, y):
    """
    Fits a Logistic Regression model using statsmodels.
    Adds a constant term (intercept) to the features.
    """
    # Statsmodels requires adding a constant for the intercept term
    X_with_const = sm.add_constant(X)
    
    # Fit the Logit model
    model = sm.Logit(y, X_with_const)
    result = model.fit()
    
    return result

def iterative_p_value_elimination(X, y, threshold=0.05):
    """
    Performs backward elimination based on P-Values.
    In each step, fits the model and removes the feature with the highest 
    P-Value that is above the significance threshold.
    """
    features = list(X.columns)
    iteration = 1
    
    while len(features) > 0:
        X_const = sm.add_constant(X[features])
        model = sm.Logit(y, X_const).fit(disp=False)
        
        # Get p-values excluding the constant/intercept
        p_values = model.pvalues.drop('const')
        max_p = p_values.max()
        
        if max_p > threshold:
            feature_to_remove = p_values.idxmax()
            features.remove(feature_to_remove)
            print(f"Iteration {iteration}: Removed '{feature_to_remove}' with P-Value: {max_p:.4f}")
            iteration += 1
        else:
            print("-" * 30)
            print(f"Pruning finished. Final feature count: {len(features)}")
            break
            
    return features, model