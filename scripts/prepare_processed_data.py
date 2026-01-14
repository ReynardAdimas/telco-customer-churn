import os, sys 
import pandas as pd 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 

from src.data.preprocess import preprocess_data 
from src.features.build_features import build_features 

raw = "data/raw/Telco-Customer-Churn.csv"
out = "data/processed/telco-churn-processed.csv" 

df = pd.read_csv(raw) 

df = preprocess_data(df, target_col="Churn") 

if "Churn" in df.columns and df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"No":0, "Yes":1}).astype(int) 

assert df["Churn"].isna().sum() == 0, "Churn has NaNs after preprocessed" 
assert set(df["Churn"].unique()) <= {0,1}, "Churn not 0/1 after preprocess" 

df_prcess = build_features(df, target_col="Churn")

os.makedirs(os.path.dirname(out), exist_ok=True)
df_prcess.to_csv(out, index=False)
print(f"Proccesed dataset saved to {out}")