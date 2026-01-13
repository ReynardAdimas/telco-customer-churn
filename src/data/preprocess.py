import pandas as pd 

def preprocess_data(df: pd.DataFrame, target_col: str = "Churn"):
    
    """
    Docstring for preprocess_data
    
    :param df: Description
    :type df: pd.DataFrame
    :param target_col: Description
    :type target_col: str 

     - trim column names 
     - drop obvious ID cols 
     - fix TotalCharges to numeric 
     - map target Churn to 0/1 
     - simple NA handling
    """
    df.columns = df.columns.str.strip() 

    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col]) 
    
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes" : 1})
    
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce") 
    
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df['SeniorCitizen'].fillna(0).astype(int) 
    
    num_cols = df.select_dtypes(include=["number"]).columns 
    df[num_cols] = df[num_cols].fillna(0) 

    return df