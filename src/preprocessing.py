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