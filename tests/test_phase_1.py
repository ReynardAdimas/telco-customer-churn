import os 
import pandas 
import sys 
sys.path.append(os.path.abspath("src"))

from data.load_data import load_data 
from data.preprocess import preprocess_data 
from features.build_features import build_features 

data_path = "D:/Coding/Hands on ML/telco-customer-churn/data/raw/Telco-Customer-Churn.csv" 
target_col = "Churn" 

def main():
    print("Testing P1: Load -> Preprocessed -> Build Feature") 

    print("1. Loading Data") 
    df = load_data(data_path) 
    print(f"Data Loaded, Data Shape: {df.shape}")
    print(df.head(1)) 

    print("\n2. Preprocessing Data")
    df_clean = preprocess_data(df, target_col=target_col)
    print(f"Data after preprocessing, Shape: {df_clean.shape}")
    print(df_clean.head(3)) 

    print("\n3. Build Features")
    df_features = build_features(df_clean, target_col=target_col)
    print(f"Data after feature engineering, shape: {df_features.shape}")
    print(df_features.head(3)) 

    print("Done")

if __name__ == "__main__":
    main()