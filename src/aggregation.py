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


def get_installments_aggregation_logic(df):
    agg_logic = {
        'SK_ID_CURR': ['first'], 
        
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'NUM_INSTALMENT_NUMBER': ['max', 'mean'],
        'DAYS_INSTALMENT': ['min', 'max', 'mean'],
        'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'INS_DPD': ['max', 'mean', 'sum'],
        'INS_DBD': ['max', 'mean', 'sum'],
        'INS_AMT_DIFF': ['max', 'mean', 'sum'],
        'INS_AMT_RATIO': ['mean']
    }
    return agg_logic

def aggregate_installments(df, agg_logic):
    df['INS_DPD'] = (df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']).clip(lower=0)
    df['INS_DBD'] = (df['DAYS_INSTALMENT'] - df['DAYS_ENTRY_PAYMENT']).clip(lower=0)
    df['INS_AMT_DIFF'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
    df['INS_AMT_RATIO'] = df['AMT_PAYMENT'] / (df['AMT_INSTALMENT'] + 0.00001)

    ins_agg = df.groupby('SK_ID_PREV').agg(agg_logic)
    
    ins_agg.columns = [f"INS_{col[0]}_{col[1].upper()}" for col in ins_agg.columns]
    
    ins_agg = ins_agg.rename(columns={'INS_SK_ID_CURR_FIRST': 'SK_ID_CURR'})
    return ins_agg.reset_index()

def get_pos_cash_agg_logic(status_ohe_cols):
    """
    Defines the aggregation dictionary for POS_CASH features.
    Separating logic from execution for cleaner maintenance.
    """
    agg_rules = {
        'SK_ID_CURR': ['first'], 
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'CNT_INSTALMENT': ['max', 'last'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    
    for col in status_ohe_cols:
        agg_rules[col] = ['mean', 'sum']
        
    return agg_rules

def aggregate_pos_cash(df):
    """
    Main orchestrator: Performs OHE, calls rules, and groups by Loan ID (SK_ID_PREV).
    """
    # Internal Preprocessing (One-Hot Encoding)
    df = pd.get_dummies(df, columns=['NAME_CONTRACT_STATUS'], prefix='POS_STAT')
    status_ohe_cols = [col for col in df.columns if col.startswith('POS_STAT_')]
    
    agg_logic = get_pos_cash_agg_logic(status_ohe_cols)
    
    pos_agg = df.groupby('SK_ID_PREV').agg(agg_logic)
    
    # Flatten MultiIndex columns (e.g., ('SK_DPD', 'max') -> 'POS_SK_DPD_MAX')
    pos_agg.columns = [f"POS_{col[0]}_{col[1].upper()}" for col in pos_agg.columns]
    
    pos_agg = pos_agg.rename(columns={'POS_SK_ID_CURR_FIRST': 'SK_ID_CURR'})
    
    return pos_agg.reset_index()

def get_credit_card_agg_rules(status_ohe_cols):
    """
    Optimized aggregation logic based on correlation and sparsity analysis.
    Removed redundant receivables, overlapped payments, and sparse 'other' drawings.
    """
    agg_rules = {
        'SK_ID_CURR': ['first'],
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        
        # Core Balance and Limit
        'AMT_BALANCE': ['max', 'mean', 'sum', 'var'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean'],
        
        # Specific Drawing Types (Excluded 'Other')
        'AMT_DRAWINGS_ATM_CURRENT': ['max', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['max', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['max', 'sum'],
        
        # Frequency of Usage
        'CNT_DRAWINGS_ATM_CURRENT': ['max', 'sum'],
        'CNT_DRAWINGS_CURRENT': ['max', 'sum'],
        'CNT_DRAWINGS_POS_CURRENT': ['max', 'sum'],
        
        # Installment and Payment behavior
        'AMT_INST_MIN_REGULARITY': ['max', 'mean'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['max', 'mean', 'sum'], # Kept Total over Current
        'CNT_INSTALMENT_MATURE_CUM': ['max', 'mean'],
        
        # Final consolidated Receivable
        'AMT_TOTAL_RECEIVABLE': ['max', 'mean', 'sum'],
        
        # Risk Indicators
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    
    # Status flags
    for col in status_ohe_cols:
        agg_rules[col] = ['mean', 'sum']
        
    return agg_rules

def aggregate_credit_card(df):
    """
    Main orchestrator for credit card data. 
    Includes feature engineering for utilization rates.
    """
    # How much of the limit is the customer using?
    df['CC_UTILIZATION'] = df['AMT_BALANCE'] / (df['AMT_CREDIT_LIMIT_ACTUAL'] + 0.00001)
    
    # 2. Preprocessing: One-Hot Encoding for statuses
    df = pd.get_dummies(df, columns=['NAME_CONTRACT_STATUS'], prefix='CC_STAT')
    status_ohe_cols = [col for col in df.columns if col.startswith('CC_STAT_')]
    
    # 3. Get rules and add the new utilization feature
    agg_rules = get_credit_card_agg_rules(status_ohe_cols)
    agg_rules['CC_UTILIZATION'] = ['max', 'mean', 'var']
    
    # 4. Grouping by Loan ID (SK_ID_PREV)
    cc_agg = df.groupby('SK_ID_PREV').agg(agg_rules)
    
    # 5. Flatten MultiIndex and Clean Up
    cc_agg.columns = [f"CC_{col[0]}_{col[1].upper()}" for col in cc_agg.columns]
    cc_agg = cc_agg.rename(columns={'CC_SK_ID_CURR_FIRST': 'SK_ID_CURR'})
    
    return cc_agg.reset_index()

def add_prev_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds financial ratios based on Siddiqi/FICO standards to the previous applications table.
    Uses a small epsilon (eps) to prevent division by zero errors.
    """
    df = df.copy()
    eps = 1e-5 
    
    # 1. Application/Origination Ratios
    df['APP_CREDIT_RATIO'] = df['AMT_CREDIT'] / (df['AMT_APPLICATION'] + eps)
    df['LTV_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + eps)
    df['DOWN_PAYMENT_RATIO'] = df['AMT_DOWN_PAYMENT'] / (df['AMT_GOODS_PRICE'] + eps)
    df['ESTIMATED_TERM'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + eps)
    
    # 2. Behavioral/Repayment Ratios (Installments)
    if 'INS_AMT_PAYMENT_SUM' in df.columns and 'INS_AMT_INSTALMENT_SUM' in df.columns:
        df['PAYMENT_DISCIPLINE_RATIO'] = df['INS_AMT_PAYMENT_SUM'] / (df['INS_AMT_INSTALMENT_SUM'] + eps)
        
    # 3. Revolving/Credit Card Ratios
    if 'CC_AMT_BALANCE_MEAN' in df.columns and 'CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN' in df.columns:
        df['CC_UTILIZATION_RATIO'] = df['CC_AMT_BALANCE_MEAN'] / (df['CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN'] + eps)
    
    if 'CC_AMT_DRAWINGS_ATM_CURRENT_SUM' in df.columns and 'CC_AMT_DRAWINGS_CURRENT_SUM' in df.columns:
        df['CC_CASH_ADVANCE_RATIO'] = df['CC_AMT_DRAWINGS_ATM_CURRENT_SUM'] / (df['CC_AMT_DRAWINGS_CURRENT_SUM'] + eps)
        
    return df

def get_prev_aggregation_rules(df: pd.DataFrame) -> dict:
    """
    Determines aggregation rules (min, max, mean, sum) dynamically 
    based on column names and data types.
    """
    agg_rules = {}
    
    # Select numeric columns excluding IDs
    num_cols = [c for c in df.select_dtypes(include=['number']).columns 
                if c not in ['SK_ID_CURR', 'SK_ID_PREV']]
    
    for col in num_cols:
        # A. Monetary amounts and ratios
        if any(keyword in col for keyword in ['AMT', 'RATIO', 'RATE', 'PERC']):
            agg_rules[col] = ['min', 'max', 'mean', 'sum']
        
        # B. Days Past Due (DPD) - Penalties
        elif 'DPD' in col:
            agg_rules[col] = ['max', 'mean']
            
        # C. Timelines (Days/Months)
        elif 'DAYS' in col or 'MONTHS' in col:
            agg_rules[col] = ['min', 'max', 'mean']
            
        # D. OHE (0-1) Categorical Columns
        elif df[col].nunique() <= 2:
            agg_rules[col] = ['mean', 'sum'] 
            
        # E. Fallback for everything else
        else:
            agg_rules[col] = ['min', 'max', 'mean']
            
    return agg_rules

def aggregate_prev_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main runner function:
    1. Adds domain features.
    2. Determines dynamic aggregation rules.
    3. Collapses table at customer level (SK_ID_CURR) via groupby.
    4. Cleans and flattens column names.
    """
    df_featured = add_prev_domain_features(df)
    
    agg_rules = get_prev_aggregation_rules(df_featured)
    
    agg_df = df_featured.groupby('SK_ID_CURR').agg(agg_rules)
    
    agg_df.columns = [f"PREV_{col[0]}_{col[1].upper()}" for col in agg_df.columns]
    
    agg_df = agg_df.reset_index()
    
    return agg_df