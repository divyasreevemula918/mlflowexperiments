import warnings
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, Lasso
import mlflow
import mlflow.sklearn
import dagshub

warnings.filterwarnings("ignore")
np.random.seed(40)

# -------------------------------
# DagsHub + MLflow setup
# -------------------------------
dagshub.init(repo_owner="divyasreevemula918", repo_name="mlflowexperiments", mlflow=True)

# Optional: set experiment name
mlflow.set_experiment("Wine_Quality_Experiments")

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# -------------------------------
# Load dataset
# -------------------------------
url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
data = pd.read_csv(url, sep=";")

train, test = train_test_split(data, test_size=0.2, random_state=42)

train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train["quality"]
test_y = test["quality"]

# -------------------------------
# Models to compare
# -------------------------------
models = {
    "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42),
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=0.5),
    "Lasso": Lasso(alpha=0.01)
}

# -------------------------------
# Run experiments
# -------------------------------
for model_name, model in models.items():
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name=model_name):
        model.fit(train_x, train_y)
        predicted_qualities = model.predict(test_x)

        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        mlflow.log_param("model_name", model_name)

        if hasattr(model, "alpha"):
            mlflow.log_param("alpha", model.alpha)

        if hasattr(model, "l1_ratio"):
            mlflow.log_param("l1_ratio", model.l1_ratio)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # Save model artifact to DagsHub MLflow tracking
        mlflow.sklearn.log_model(model, artifact_path=model_name.lower())

        print(f"{model_name}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R2: {r2:.4f}")
        print("-" * 40)