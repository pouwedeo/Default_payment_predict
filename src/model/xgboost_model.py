import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
from src.data.data_preped import data_preped
from src.mlflow_Tracker.mlflowTracker import MLflowTracker
import subprocess
import time

mlflow_process = subprocess.Popen(["mlflow", "ui"])
time.sleep(2)

# Data initialization
X_train_balanced, X_test, y_train_balanced, y_test = data_preped()

# XGBOOST Moel
params = {
    "use_label_encoder": False,
    "eval_metric": 'mlogloss'
}
xgb_model = xgb.XGBClassifier(**params)
xgb_model.fit(X_train_balanced, y_train_balanced)
y_pred = xgb_model.predict(X_test)

recall_metrics = classification_report(y_test, y_pred, output_dict=True)

roc_auc = roc_auc_score(y_test, y_pred)

metrics = {
           "precision_0": recall_metrics["0"]["precision"],
           "recall_0": recall_metrics["0"]["recall"],
           "f1_score_0": recall_metrics["0"]["f1-score"],
           "precision_1": recall_metrics["1"]["precision"],
           "recall_1": recall_metrics["1"]["recall"],
           "f1_score_1": recall_metrics["1"]["f1-score"],
           "roc_auc": roc_auc
           }


# Mlflow Tracker
tracker = MLflowTracker()
tracker.train_and_log(
                     run_name="XGBOOST", params=params,
                     metrics=metrics, model_name=xgb_model,
                     X_val=X_test, artifacts_path="XGB_predict",
                     experiment_name="Loan_Predict_XGBOOST"
                      )
mlflow_process.wait()