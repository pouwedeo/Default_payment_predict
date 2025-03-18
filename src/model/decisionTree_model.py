from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from src.data.data_preped import data_preped
from src.mlflow_Tracker.mlflowTracker import MLflowTracker
import subprocess
import time

mlflow_process = subprocess.Popen(["mlflow", "ui"])
time.sleep(2)

# Data initialization
X_train_balanced, X_test, y_train_balanced, y_test = data_preped()

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_balanced, y_train_balanced)
dt_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, dt_pred)
metrics = accuracy

# Mlflow Tracker
tracker = MLflowTracker()
tracker.train_and_log(
                     run_name="DecisionTree", params="null",
                     metrics=metrics, model_name=dt_model,
                     X_val=X_test, artifacts_path="DecisionTree_predict"
                      )
mlflow_process.wait()
