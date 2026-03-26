# 🚀 MLflow Experiment Tracking with DagsHub

## 📌 Project Overview

This project demonstrates how to build a Machine Learning pipeline using **ElasticNet Regression** and track experiments using **MLflow integrated with DagsHub**.

It helps in logging:

* Model parameters ⚙️
* Performance metrics 📊
* Trained models 💾

---

## ⚙️ Tech Stack

* 🐍 Python
* 📊 Scikit-learn
* 📈 MLflow
* 🌐 DagsHub
* 🧮 NumPy, Pandas

---

## 📂 Project Structure

```
MLflow/
│── train_models.py       # Model training + MLflow logging
│── app.py                # (Optional Streamlit app)
│── requirements.txt      # Dependencies
│── runtime.txt           # Python version
│── .gitignore
│── README.md
```

---

## 🧠 Model Used

* **ElasticNet Regression**

  * Combines L1 (Lasso) + L2 (Ridge) regularization

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
https://dagshub.com/divyasreevemula918/mlflowexperiments/experiments
* streamlit app:
  

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/divyasreevemula918/mlflowexperiments.git
cd mlflowexperiments
```

### 2️⃣ Create virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run training script

```bash
python train_models.py
```

---

## 🔥 MLflow + DagsHub Integration

This project uses:

```python
import dagshub
dagshub.init(repo_owner="divyasreevemula918", repo_name="mlflowexperiments", mlflow=True)
```

This automatically connects MLflow with DagsHub and logs experiments online.

---

## 📈 Output Example

After running the script, you will see:

```
RMSE: 0.7628  
MAE: 0.6189  
R2: 0.1096  
```

And all results will be available on DagsHub Experiments page.

---

## 📸 Screenshot (Optional)

👉 Add your DagsHub experiment screenshot here

```
![DagsHub Experiments](your_screenshot.png)
```

---

## 🚀 Future Improvements

* Add multiple models (Linear Regression, Random Forest)
* Hyperparameter
