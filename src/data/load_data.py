import pandas as pd 
import os 

def load_data(file_path: str) -> pd.DataFrame:
    """
    Docstring for load_data
    
    :param file_path: Description
    :type file_path: str
    :return: Description
    :rtype: DataFrame
    
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}") 
    
    return pd.read_csv(file_path)