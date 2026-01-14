import os 
import sys 
import time 
import argparse 
import pandas as pd 
import mlflow 
import mlflow.sklearn 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score, roc_auc_score
) 
from xgboost import XGBClassifier 
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data 

def main(args):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path_obj = Path(project_root) / "mlruns"
    # mlruns_path = args.mlflow_uri or f"file://{project_root}/mlruns"
    mlruns_path = args.mlflow_uri or path_obj.as_uri()
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment) 

    with mlflow.start_run():
        mlflow.log_param("model", "xgboost") 
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size) 

        print("1. Loading Data")
        df = load_data(args.input) 
        print(f"Data Loaded: {df.shape[0]}")
        print("2. Validating data quality")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))

        if not is_valid:
            import json 
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"Data Quality check failed. Issues: {failed}")
        else:
            print("Data validation passed") 
        
        print("3. Preprocessing data")
        df = preprocess_data(df) 

        processed_path = os.path.join(project_root, "data", "processed", "telco-churn-processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True) 
        df.to_csv(processed_path, index=False)
        print(f"Processed dataset saved to {processed_path}") 

        print("4. Building Feature")
        target = args.target 
        if target not in df.columns:
            raise ValueError(f"Target column not found in data")

        df_enc = build_features(df, target_col=target) 

        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int) 
        print(f"Feature engineering completed: {df_enc.shape[1]} features") 

        import json, joblib 
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True) 

        feature_cols = list(df_enc.drop(columns=[target]).columns)

        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f) 

        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        preprocessing_artifact = {
            "feature_columns": feature_cols, 
            "target" : target
        }  

        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"Saved {len(feature_cols)} feature columns for serving consistency") 

        print("5. Splitting Data") 
        X = df_enc.drop(columns=[target])
        y = df[target] 

        X_train, X_test, y_train, y_test = train_test_split(
            X,y, 
            test_size=args.test_size, 
            stratify=y, 
            random_state=42
        ) 
        print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
        scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()
        print(f"Class imbalance ratio: {scale_pos_weight:.2f} applied to positive class") 

        print("6. Training Model")
        model = XGBClassifier(
            n_estimators=394, 
            learning_rate=0.10472571613778275, 
            max_depth=4, 
            subsample=0.9128488190674419,
            colsample_bytree=0.665022422325231, 
            min_child_weight=3, 
            gamma=4.992344710206348, 
            reg_alpha=0.1818683352868308, 
            reg_lambda=4.186922280533751,
            n_jobs=-1, 
            random_state=42, 
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight
        ) 

        mlflow.log_param("model_type", "xgboost")
        mlflow.log_params(model.get_params())

        t0 = time.time() 
        try: 
            model.fit(X_train, y_train) 
        except Exception as e:
            mlflow.set_tag("status", "failed")
            raise e
        train_time = time.time() - t0 
        mlflow.log_metric("train time", train_time)
        print(f"Model trained in {train_time:.2f} seconds") 

        print("7. Evaluating model performance")

        t1 = time.time() 
        proba = model.predict_proba(X_test)[:,1]
        y_pred = (proba >= args.threshold).astype(int)
        pred_time = time.time() - t1 
        mlflow.log_metric("Pred_time", pred_time)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, proba)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print("Model Performance: ")
        print(f"Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f} | ROC AUC: {roc_auc:.2f}") 

        print("8. Saving model to MLflow")
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model"
        )

        print("Model saved to MLflow") 

        print("Performance Summary:")
        print(f"Training Time: {train_time:.2f} s")
        print(f"Inference time: {pred_time:.2f} s")
        print(f"Sample per second: {len(X_test)/pred_time:.0f}")

        print("\n Detailed Classification Report")
        print(classification_report(y_test, y_pred, digits=3))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")  
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/raw/Telco-Customer-Churn.csv)")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args) 

"""
# Use this below to run the pipeline:

python scripts/run_pipeline.py ^ --input data/raw/Telco-Customer-Churn.csv ^ --target Churn

"""