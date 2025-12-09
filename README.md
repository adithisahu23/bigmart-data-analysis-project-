# 🛒 BigMart Sales Prediction – Machine Learning Project

Predicting product-level sales across retail outlets using data science and machine learning.

---

## 📌 Overview

This project aims to forecast the sales of various products across different BigMart outlets based on historical sales data and metadata. By analyzing product attributes, outlet characteristics, and pricing information, the model can predict future sales—helping businesses make informed decisions on inventory, supply chain, and demand planning.

This is a complete end-to-end ML project including **data preprocessing, EDA, model building, hyperparameter tuning, evaluation, and deployment**.

---

## 📂 Project Structure

```
BigMart-Sales-Prediction/
│── dataset/
│   ├── Train.csv
│   └── Test.csv
│── notebooks/
│   └── EDA.ipynb
│   └── Model_Training.ipynb
│── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── utils.py
│── deployment/
│   ├── app.py (Streamlit/Flask)
│   └── model.pkl
│── README.md
```

---

## 🎯 Project Objectives

* Understand key factors influencing retail sales
* Build a regression model to predict product sales
* Improve accuracy with feature engineering & tuning
* Deploy the model for real-world usage

---

## 📊 Dataset Details

The dataset contains product-level and outlet-level information such as:

### **Product Features**

* Item_Identifier
* Item_Weight
* Item_Fat_Content
* Item_Visibility
* Item_Type
* Item_MRP

### **Outlet Features**

* Outlet_Identifier
* Outlet_Establishment_Year
* Outlet_Size
* Outlet_Location_Type
* Outlet_Type

### **Target Variable**

* **Item_Outlet_Sales** (sales value to be predicted)

---

## 🔍 Exploratory Data Analysis (EDA)

Key EDA steps included:

* Handling missing values
* Feature correlation study
* Understanding outlet-wise sales distribution
* Identifying which product categories perform best
* Visualizing relationships (MRP vs Sales, Fat Content vs Sales, etc.)

---

## 🛠 Machine Learning Workflow

### **1️⃣ Data Preprocessing**

* Missing value handling
* Categorical encoding (Label Encoding / One-Hot Encoding)
* Feature scaling
* Transformations to improve model performance

### **2️⃣ Model Training**

Multiple models were tested:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor
* Gradient Boosting
* Decision Tree Regressor

### **3️⃣ Model Evaluation**

Evaluated using metrics such as:

* RMSE (Root Mean Squared Error)
* MAE
* R² Score

### **4️⃣ Best Model Selection**

The model with the best RMSE score was saved as `model.pkl` for deployment.

---

## 🚀 Deployment

A simple **Streamlit/Flask web app** was created where users can input product/outlet details and receive predicted sales instantly.

### To run the app:

```bash
streamlit run app.py
```

or

```bash
python app.py
```

---

## 📈 Results

* Achieved strong prediction accuracy with tuned ensemble models
* Identified key factors affecting sales (MRP, Outlet Type, Item Type, Visibility, etc.)
* Improved model performance using feature engineering and hyperparameter tuning

---

## 🔮 Future Enhancements

* Add a dashboard for live analytics
* Include time-series forecasting
* Improve model using deep learning
* Deploy using Docker or cloud platforms (AWS/GCP/Azure)

---

## 🧠 Key Learnings

* Hands-on experience with real retail datasets
* ML regression techniques and model optimization
* Deployment of ML models to production environments
* Understanding of retail analytics and business logic

