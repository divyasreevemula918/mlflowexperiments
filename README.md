<<<<<<< HEAD
# 🍷 MLflow Wine Quality App

An interactive **Streamlit + MLflow** project for training and comparing regression models on the **Wine Quality** dataset.  
This app lets users choose a model, adjust hyperparameters, train it, and view evaluation metrics in a simple UI.

---

## 🚀 Live Demo

🔗 **Deployment Link:**  
👉 https://mlflowexperiments-i6fssqfvpnqkdx3gxxkbue.streamlit.app/

---

## 📌 Project Overview

This project demonstrates how to:

- Build a regression-based machine learning workflow
- Compare multiple regression models
- Use **Streamlit** for an interactive web interface
- Use **MLflow** for experiment tracking
- Log model parameters and evaluation metrics
- Create a deployment-ready ML project

The dataset used in this project is the **Red Wine Quality** dataset.

---

## ✨ Features

- Interactive Streamlit web app 🌐
- Train and compare multiple regression models
- Adjustable hyperparameters using sliders
- Real-time evaluation results
- MLflow experiment logging
- Clean UI for user-friendly interaction

---

## 🧠 Models Used

This project compares the following regression models:

- ElasticNet
- Linear Regression
- Ridge Regression
- Lasso Regression

---

## 📊 Metrics Used

The app evaluates models using:

- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **R² Score** — Coefficient of Determination

These metrics help measure model performance for wine quality prediction.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- MLflow
- Scikit-learn
- Pandas
- NumPy

---
## screenshots
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/38bed20c-bd9f-4d50-b46f-6991237e4734" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f9572c53-2aa1-46db-ab3b-fd20020f7074" />



## 📂 Project Structure

```bash
mlflowexperiments/
│
├── app.py                # Streamlit app for training and comparing models
├── train_models.py       # Separate MLflow experiment script
├── requirements.txt      # Project dependencies
├── runtime.txt           # Python runtime version for deployment
├── dockerfile            # Docker support
├── .gitignore            # Ignored files/folders
└── README.md             # Project documentation
=======
##MLflow experiment
MLFLOW_TRACKING_URI=https://dagshub.com/divyasreevemula918/mlflowexperiments.mlflow\
MLFLOW_TRACKING_USERNAME=divyasreevemula918\
MLFLOW_TRACKING_PASSWORD=5e6d9237c72e8605e5a73a914b27c7a28d617386\
python script.py
>>>>>>> 7d21fe7 (Added MLflow with DagsHub integration)
