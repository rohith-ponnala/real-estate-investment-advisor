Real Estate Investment Advisor

An end-to-end Machine Learning project that predicts:

Whether a property is a Good Investment (Classification)

Estimated property price after 5 years (Regression)

Built using Python, Scikit-learn, EDA, Feature Engineering, MLflow, and Streamlit.

🔍 Project Overview

This project helps real estate investors make data-driven decisions by analyzing property features such as:

Size, BHK, Location

Amenities & Furnishing

Schools / Hospitals Nearby

Public Transport Accessibility

Property Age & Price per SqFt

The system performs:

1️⃣ Investment Classification

Predicts whether a property is a Good Investment using engineered features and ML models.

2️⃣ Price Forecasting

Predicts future price after 5 years based on appreciation models and regression algorithms.

🛠️ Tech Stack

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn (Logistic Regression, Random Forest, Linear Regression)

MLflow for experiment tracking

Streamlit for deployment UI

Joblib for model saving/loading

📁 Project Structure
real_estate_project/
│── app.py
│── requirements.txt
│── README.md
│
├── data/
│   └── india_housing_prices.csv
│
├── saved_models/
│   ├── best_classifier.pkl
│   └── best_regressor.pkl
│
└── real_estate.ipynb

🚀 How to Run Locally
pip install -r requirements.txt
streamlit run app.py


The app will launch at:

http://localhost:8501

📊 Features

Cleaned & preprocessed real estate dataset

Feature engineering (Price per SqFt, Age, Investment Score)

EDA visualizations

MLflow experiment logs

Deployed Streamlit app with:

Property input form

Investment prediction

5-year price forecast

Visual insights (heatmaps, trends, feature importance)

🧠 Business Insight

The system helps investors identify high-return properties and forecast future prices, improving decision-making and boosting transparency for real estate platforms.
