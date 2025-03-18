import catboost as cb
from sklearn.metrics import accuracy_score
from src.data.data_preped import data_preped
from src.mlflow_Tracker.mlflowTracker import MLflowTracker
import subprocess
import time

mlflow_process = subprocess.Popen(["mlflow", "ui"])
time.sleep(2)

# Data initialization
X_train_balanced, X_test, y_train_balanced, y_test = data_preped()

# CatBoost Model
cb_model = cb.CatBoostClassifier(verbose=0)
cb_model.fit(X_train_balanced, y_train_balanced)
cb_pred = cb_model.predict(X_test)
accuracy = accuracy_score(y_test, cb_pred)
metrics = accuracy

# Mlflow Tracker
tracker = MLflowTracker()
tracker.train_and_log(
                     run_name="CatBoost", params="null",
                     metrics=metrics, model_name=cb_model,
                     X_val=X_test, artifacts_path="CatBoost_predict"
                      )
mlflow_process.wait()