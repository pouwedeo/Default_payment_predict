from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.data.data_preped import data_preped
from src.mlflow_Tracker.mlflowTracker import MLflowTracker
import subprocess
import time

mlflow_process = subprocess.Popen(["mlflow", "ui"])
time.sleep(2)

# Data initialization
X_train_balanced, X_test, y_train_balanced, y_test = data_preped()

# RandomForest Model
params = { 
          "n_estimators": 100,
          "random_state": 42
          }
rf_model = RandomForestClassifier(**params)
rf_model.fit(X_train_balanced, y_train_balanced)
rf_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, rf_pred)
metrics = accuracy

# Mlflow Tracker
tracker = MLflowTracker()
tracker.train_and_log(
                     run_name="RandomForest", params=params,
                     metrics=metrics, model_name=rf_model,
                     X_val=X_test, artifacts_path="RandomForest_predict"
                      )
mlflow_process.wait()