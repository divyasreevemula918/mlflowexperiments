# 🚀 MLflow Experiment Tracking with DagsHub, Streamlit & Docker

## 📌 Project Overview

This project demonstrates a complete **Machine Learning pipeline** using **ElasticNet Regression**, integrated with **MLflow for experiment tracking**, **DagsHub for visualization**, **Streamlit for UI**, and **Docker for containerized deployment**.

---

## ⚙️ Tech Stack

* 🐍 Python
* 📊 Scikit-learn
* 📈 MLflow
* 🌐 DagsHub
* 🎨 Streamlit
* 🐳 Docker
* 🧮 NumPy, Pandas

---

## 📂 Project Structure

```
MLflow/
│── train_models.py        # Training + MLflow logging
│── app.py                 # Streamlit UI
│── Dockerfile             # Docker configuration
│── requirements.txt       # Dependencies
│── .dockerignore
│── .gitignore
│── README.md
```

---

## 🧠 Model Used

* **ElasticNet Regression**

  * Combines L1 (Lasso) and L2 (Ridge) regularization

---

## 📊 Metrics Logged

The following metrics are tracked using MLflow:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R2 Score

---

## 🔗 Project Links

* 💻 GitHub Repository:
  https://github.com/divyasreevemula918/mlflowexperiments

* 📊 DagsHub Experiments:
  https://dagshub.com/divyasreevemula918/mlflowexperiments

* 🌍 Streamlit App:
  https://mlflowexperiments-i6fssqfvpnqkdx3gxxkbue.streamlit.app/

---

## ▶️ Run Locally (Without Docker)

### 1️⃣ Clone repository

```
git clone https://github.com/divyasreevemula918/mlflowexperiments.git
cd mlflowexperiments
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run training

```
python train_models.py
```

### 4️⃣ Run Streamlit app

```
streamlit run app.py
```

---

## 🐳 Run Using Docker

### 1️⃣ Build Docker image

```
docker build -t mlflow-streamlit-app .
```

### 2️⃣ Run Docker container

```
docker run -p 8501:8501 \
-e MLFLOW_TRACKING_USERNAME=your_username \
-e MLFLOW_TRACKING_PASSWORD=your_token \
mlflow-streamlit-app
```

### 3️⃣ Open in browser

```
http://localhost:8501
```

---

## 🔐 Environment Variables

For secure authentication, use environment variables:

```
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_token
```

⚠️ Do not commit `.env` file to GitHub.

---

## 🔥 MLflow + DagsHub Integration

This project uses:

```python
import dagshub
dagshub.init(repo_owner="divyasreevemula918", repo_name="mlflowexperiments", mlflow=True)
```

This enables automatic logging of experiments to DagsHub.

---

## 📈 Output Example

```
RMSE: 0.7628
MAE: 0.6189
R2: 0.1096
```

---

## 📸 Screenshot

👉 Add screenshot of DagsHub experiments or Streamlit UI here

```
![App Screenshot](your_image.png)
```

---

## 🚀 Features

* Experiment tracking using MLflow
* Remote logging with DagsHub
* Interactive UI using Streamlit
* Containerized deployment using Docker
* Secure environment variable handling

---

## 🎯 Key Learnings

* MLflow experiment tracking
* DagsHub integration
* Docker containerization
* Streamlit app development
* End-to-end ML pipeline

---

## 🚀 Future Improvements

* Add multiple models (Random Forest, XGBoost)
* Hyperparameter tuning
* CI/CD pipeline integration
* Cloud deployment (AWS / Render / Railway)

---

## 🙌 Author

**Divya Sree Vemula**

* GitHub: https://github.com/divyasreevemula918
* LinkedIn: https://linkedin.com/in/divya-sree-vemula-4024982a0

---


