import streamlit as st
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="MLflow Wine Quality App", page_icon="🍷")

st.title("🍷 Wine Quality Prediction with MLflow")
st.write("Train and compare regression models on the Wine Quality dataset.")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    data = pd.read_csv(url, sep=";")
    return data

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

data = load_data()
st.subheader("Dataset Preview")
st.dataframe(data.head())

X = data.drop(columns=["quality"])
y = data["quality"]

test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)

model_name = st.selectbox(
    "Choose Model",
    ["ElasticNet", "LinearRegression", "Ridge", "Lasso"]
)

alpha = st.slider("Alpha", 0.01, 1.0, 0.5, 0.01)
l1_ratio = st.slider("L1 Ratio (ElasticNet only)", 0.0, 1.0, 0.5, 0.01)

if st.button("Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if model_name == "ElasticNet":
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    elif model_name == "LinearRegression":
        model = LinearRegression()
    elif model_name == "Ridge":
        model = Ridge(alpha=alpha)
    else:
        model = Lasso(alpha=alpha)

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rmse, mae, r2 = eval_metrics(y_test, predictions)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", test_size)

        if model_name in ["ElasticNet", "Ridge", "Lasso"]:
            mlflow.log_param("alpha", alpha)

        if model_name == "ElasticNet":
            mlflow.log_param("l1_ratio", l1_ratio)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

    st.success(f"{model_name} trained successfully 🎉")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**MAE:** {mae:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")