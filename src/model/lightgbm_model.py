import lightgbm as lgb
from sklearn.metrics import accuracy_score
from src.data.data_preped import data_preped
from src.mlflow_Tracker.mlflowTracker import MLflowTracker
import subprocess
import time

mlflow_process = subprocess.Popen(["mlflow", "ui"])
time.sleep(2)

# Data initialization
X_train_balanced, X_test, y_train_balanced, y_test = data_preped()


lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train_balanced, y_train_balanced)
lgb_pred = lgb_model.predict(X_test)
accuracy = accuracy_score(y_test, lgb_pred)
metrics = accuracy


# Mlflow Tracker
tracker = MLflowTracker()
tracker.train_and_log(
                     run_name="lightgbm", params="null",
                     metrics=metrics, model_name=lgb_model,
                     X_val=X_test, artifacts_path="Lgb_predict"
                      )
mlflow_process.wait()