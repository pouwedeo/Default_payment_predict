import xgboost as xgb
from sklearn.metrics import accuracy_score
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
xgb_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, xgb_pred)

metrics = accuracy

# Mlflow Tracker
tracker = MLflowTracker()
tracker.train_and_log(
                     run_name="XGBOOST", params=params,
                     metrics=metrics, model_name=xgb_model,
                     X_val=X_test, artifacts_path="XGB_predict"
                      )
mlflow_process.wait()