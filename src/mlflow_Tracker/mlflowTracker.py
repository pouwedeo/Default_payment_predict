import mlflow


class MLflowTracker:
    def __init__(self, experiment_name="Loan_Predict"):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(experiment_name)
        mlflow.set_experiment_tag("project_name", "Model Comparison")
        mlflow.set_experiment_tag("team", "DataShow")
        mlflow.set_experiment_tag("version", "1.0")

    def train_and_log(self, run_name, params, metrics, model_name, X_val,
                      artifacts_path):
        with mlflow.start_run(run_name=run_name):
            # Log des paramètres et métriques
            mlflow.log_param("Params", params)
            mlflow.log_metric("Metrics", metrics)
            mlflow.sklearn.log_model(sk_model=model_name, input_example=X_val,
                                     artifact_path=artifacts_path)

            print(f"Modèle {model_name} enregistré")
