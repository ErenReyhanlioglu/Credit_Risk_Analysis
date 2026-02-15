import pandas as pd
import numpy as np

def get_bureau_balance_status_dummies(df):
    """
    Performs One-Hot Encoding on the STATUS column.
    STATUS values range from 0 (no DPD) to 5 (120+ DPD), 
    plus 'C' (Closed) and 'X' (Unknown).
    """
    return pd.get_dummies(df, columns=['STATUS'], prefix='STATUS')

def get_bb_agg_logic(df):
    """
    Defines the standard risk aggregation dictionary.
    Includes time-depth (MONTHS_BALANCE) and behavior (STATUS).
    """
    # Time-based features
    agg_logic = {
        'MONTHS_BALANCE': ['min', 'max', 'size']
    }
    
    # Status-based features (sum for volume, mean for ratio)
    status_cols = [col for col in df.columns if 'STATUS_' in col]
    for col in status_cols:
        agg_logic[col] = ['mean', 'sum']
        
    return agg_logic

def aggregate_bureau_balance(df, agg_logic):
    """
    Groups by SK_ID_BUREAU and flattens column names.
    """
    # Perform Grouping
    bb_agg = df.groupby('SK_ID_BUREAU').agg(agg_logic)
    
    # Professional Naming: BB_STATUS_0_MEAN etc.
    bb_agg.columns = [f"BB_{col[0]}_{col[1].upper()}" for col in bb_agg.columns]
    
    return bb_agg.reset_index()

def get_bureau_dummies(df):
    """
    Applies One-Hot Encoding to categorical columns in the bureau table.
    Targets CREDIT_ACTIVE (Active/Closed) and CREDIT_TYPE (Loan Types).
    """
    categorical_cols = ['CREDIT_ACTIVE', 'CREDIT_TYPE']
    return pd.get_dummies(df, columns=categorical_cols, prefix_sep='_')

def get_bureau_aggregation_logic(df):
    """
    Defines a dictionary for mapping features to their respective aggregation functions.
    Based on the 5C's of Credit Risk.
    """
    agg_logic = {
            'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'CNT_CREDIT_PROLONG': ['sum'],
            'AMT_ANNUITY': ['max', 'mean'],
            'DAYS_ENDDATE_FACT': ['min', 'max', 'mean']
        }
    
    cat_cols = [col for col in df.columns if 'CREDIT_ACTIVE_' in col or 'CREDIT_TYPE_' in col]
    for col in cat_cols:
        agg_logic[col] = ['mean', 'sum']
        

    bb_cols = [col for col in df.columns if col.startswith('BB_')]
    for col in bb_cols:
        agg_logic[col] = ['mean']
        
    return agg_logic

def aggregate_bureau_data(df, agg_logic):
    """
    Groups the dataframe by customer ID (SK_ID_CURR) and applies aggregation logic.
    Renames columns for clear traceability.
    """
    bureau_agg = df.groupby('SK_ID_CURR').agg(agg_logic)
    
    bureau_agg.columns = [f"BURO_{col[0]}_{col[1].upper()}" for col in bureau_agg.columns]
    
    return bureau_agg.reset_index()
