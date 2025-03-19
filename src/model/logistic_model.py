from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from src.data.data_preped import data_preped
from src.mlflow_Tracker.mlflowTracker import MLflowTracker
import subprocess
import time

mlflow_process = subprocess.Popen(["mlflow", "ui"])
time.sleep(2)

# Data initialization
X_train_balanced, X_test, y_train_balanced, y_test = data_preped()

# Entrainement du modèle logistic
model_lr = LogisticRegression()
model_lr.fit(X_train_balanced, y_train_balanced)

# Prédiction sur les données de test
y_pred = model_lr.predict(X_test)

# Probabilité d'apparition de la class 1

y_1_prob = model_lr.predict_proba(X_test)[:, 1]

# Metrics Precision, Recall, F1-score

recall_metrics = classification_report(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
confusion_matrixs = confusion_matrix(y_test, y_pred)

metrics = {"Recall": recall_metrics[0], "Precision": recall_metrics[1],
           "F1-score": recall_metrics[2], "Accuracy": accuracy
           }

# Mlflow Tracker
tracker = MLflowTracker()
tracker.train_and_log(
                     run_name="LogisticRegression", params="null",
                     metrics=accuracy, model_name=model_lr,
                     X_val=X_test, artifacts_path="Logistic_predict"
                      )
mlflow_process.wait()