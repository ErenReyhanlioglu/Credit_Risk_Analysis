import pandas as pd

def get_column_descriptions(table_name, description_path='../data/raw/HomeCredit_columns_description.csv'):
    """
    Filters and returns column descriptions for a specific table 
    from the HomeCredit_columns_description.csv file.
    
    Args:
        table_name (str): The name of the table (e.g., 'bureau.csv').
        description_path (str): Path to the column description CSV file.
        
    Returns:
        pd.DataFrame: Filtered rows containing 'Row', 'Description', and 'Special' columns.
    """
    try:
        descriptions = pd.read_csv(description_path, encoding='ISO-8859-1')
        
        if 'application' in table_name:
            target_table = 'application_{train|test}.csv'
        else:
            target_table = table_name

        filtered_data = descriptions[descriptions['Table'] == target_table][['Row', 'Description', 'Special']]
        
        if filtered_data.empty:
            return f"No descriptions found for table: '{table_name}'. Please check the input string."
            
        return filtered_data
        
    except FileNotFoundError:
        return f"Error: The description file was not found at {description_path}."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def add_financial_ratios(df):
    """
    Expert-level feature engineering based on FICO and Naeem Siddiqi standards.
    Focuses on Debt-to-Income (DTI), Solvency, and External Score synergy.
    """
    df = df.copy()
    eps = 1e-5  

    # --- 1. DEBT-TO-INCOME & BURDEN ---
    # The 'Annuity' is the monthly/yearly burden. How much does it eat from total income?
    # Equation: $$DTI = \frac{AMT\_ANNUITY}{AMT\_INCOME\_TOTAL}$$
    df['APP_DTI_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + eps)
    
    # Credit to Income: Overall debt burden relative to earning power.
    df['APP_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + eps)

    # --- 2. EXTERNAL SOURCE SYNERGY (The Kaggle Holy Grail) ---
    # External sources are powerful. Their mean, product, and weighted interaction are key.
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['APP_EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
    df['APP_EXT_SOURCE_NAN_COUNT'] = df[ext_cols].isnull().sum(axis=1)
    df['APP_EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    
    # Weighted Score: Often EXT_SOURCE_3 and 2 are more important in this dataset.
    df['APP_EXT_SOURCE_WEIGHTED'] = (df['EXT_SOURCE_1'] * 2 + df['EXT_SOURCE_2'] * 3 + df['EXT_SOURCE_3'] * 5) / 10

    # --- 3. SOLVENCY & LIQUIDITY ---
    # Disposable Income: What's left for the family after paying the loan?
    df['APP_DISPOSABLE_INCOME'] = df['AMT_INCOME_TOTAL'] - df['AMT_ANNUITY']
    df['APP_INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + eps)
    df['APP_DISPOSABLE_PER_PERSON'] = df['APP_DISPOSABLE_INCOME'] / (df['CNT_FAM_MEMBERS'] + eps)

    # --- 4. LIFE-STAGE & STABILITY ---
    # Employment Ratio: Percentage of adult life spent working.
    df['APP_EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + eps)
    
    # Registration Stability: Has the customer lived in the same place for long?
    df['APP_REGISTRATION_TO_BIRTH_RATIO'] = df['DAYS_REGISTRATION'] / (df['DAYS_BIRTH'] + eps)
    df['APP_ID_PUBLISH_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / (df['DAYS_BIRTH'] + eps)

    # --- 5. ASSET RISK ---
    # Car age vs life stage. Old cars might imply hidden maintenance costs or lower wealth.
    if 'OWN_CAR_AGE' in df.columns:
        df['APP_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / (df['DAYS_BIRTH'] / -365.0 + eps)
        df['APP_CAR_TO_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / (df['DAYS_EMPLOYED'] / -365.0 + eps)

    return df