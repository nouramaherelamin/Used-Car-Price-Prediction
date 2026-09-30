# 🚗 Used Car Price Intelligence

### AI-Powered Used Car Price Prediction & Market Analytics

An interactive machine learning and analytics application for exploring
used-car market data, understanding pricing patterns, and predicting
estimated selling prices based on vehicle characteristics.

Built with Python, Pandas, Scikit-learn, Plotly, and Streamlit.

------------------------------------------------------------------------

## ✨ Project Overview

Buying or selling a used car involves many factors that can influence
its market price, including:

-   Manufacturing year
-   Present market price
-   Driven kilometers
-   Fuel type
-   Selling type
-   Transmission
-   Number of previous owners
-   Vehicle age

This project transforms used-car data into an interactive analytics
platform that combines:

📊 Exploratory Data Analysis\
📈 Market Insights\
🤖 Machine Learning\
💰 Price Prediction\
📉 Model Evaluation\
🔎 Data Exploration\
✅ Data Quality Monitoring

The application allows users to explore the dataset visually and
estimate a vehicle's selling price using a trained machine learning
model.

------------------------------------------------------------------------

## 🎯 Objectives

-   Understand the characteristics of the used-car market.
-   Explore relationships between vehicle features and selling price.
-   Identify important factors associated with vehicle pricing.
-   Build a machine learning model for price prediction.
-   Evaluate model performance using multiple metrics.
-   Provide an interactive interface for predictions.
-   Present analytical results through a modern dashboard.

------------------------------------------------------------------------

## 🧠 Machine Learning

The project uses a trained machine learning model to estimate the
selling price of a used vehicle.

### Model

**Gradient Boosting Regressor**

### Input Features

-   Manufacturing Year
-   Present Price
-   Driven Kilometers
-   Fuel Type
-   Selling Type
-   Transmission
-   Previous Owners
-   Car Age

### Model Performance

  Metric                Score
  ------------------ --------
  MAE                    1.09
  RMSE                   2.26
  R²                     0.80
  Cross-Validation     5-Fold

The trained model is saved as a reusable `.joblib` artifact and loaded
by the Streamlit application for predictions.

------------------------------------------------------------------------

## 📊 Dashboard

The application contains seven interactive sections:

### 1. Overview

A high-level view of the used-car dataset, including:

-   Total vehicles
-   Average price
-   Median price
-   Average kilometers driven
-   Average vehicle age
-   Key market insights

### 2. Price Analysis

Explore how vehicle prices vary according to:

-   Manufacturing year
-   Vehicle age
-   Kilometers driven
-   Fuel type
-   Transmission
-   Selling type

### 3. Market Insights

Interactive visualizations for discovering pricing patterns and
relationships between vehicle characteristics and selling price.

### 4. Price Predictor

Enter vehicle information and receive an estimated selling price from
the trained machine learning model.

### 5. Model Performance

Review:

-   MAE
-   RMSE
-   R²
-   Model information
-   Evaluation visualizations

### 6. Data Explorer

Interactively explore the processed dataset through tables and filters.

### 7. Data Quality

Review dataset quality and validation information.

------------------------------------------------------------------------

## 🎨 UI / UX

The application uses a custom visual system based on:

-   **Charcoal**
-   **Emerald Green**
-   **Soft White**

### Design Features

-   Responsive dashboard layout
-   Custom Streamlit theme
-   Animated Hero section
-   Interactive KPI cards
-   Hover animations
-   Animated gradients
-   Emerald glow effects
-   Interactive Plotly charts
-   Custom car favicon
-   Modern prediction interface
-   Responsive sections
-   Custom buttons and cards

The goal is to provide a modern analytics-product experience rather than
the default Streamlit appearance.

------------------------------------------------------------------------

## 🛠️ Technologies

### Programming

-   Python

### Data Analysis

-   Pandas
-   NumPy

### Visualization

-   Plotly
-   Matplotlib

### Machine Learning

-   Scikit-learn
-   Joblib

### Application

-   Streamlit

### Documentation

-   Jupyter Notebook
-   Markdown

------------------------------------------------------------------------

## 📁 Project Structure

``` text
Used_Car_Price_Intelligence/
│
├── app/
│   ├── app.py
│   └── assets/
│       ├── used_cars_hero.png
│       ├── car_icon.png
│       └── car_favicon.png
│
├── data/
│   ├── raw/
│   │   └── car data.csv
│   │
│   └── processed/
│       └── car_data_clean.csv
│
├── models/
│   └── best_car_price_model.joblib
│
├── notebooks/
│   └── ...
│
├── reports/
│   └── ...
│
├── src/
│   └── ...
│
├── tests/
│   └── ...
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
├── README.md
└── .gitignore
```

------------------------------------------------------------------------

## ⚙️ Installation

### 1. Clone the repository

``` bash
git clone YOUR_REPOSITORY_URL
cd Used_Car_Price_Intelligence
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
```

### 3. Activate the environment

#### Windows

``` bash
.venv\Scripts\activate
```

#### macOS / Linux

``` bash
source .venv/bin/activate
```

### 4. Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Run the Application

From the project root:

``` bash
streamlit run app/app.py
```

Or:

``` bash
python -m streamlit run app/app.py
```

The application will open in your browser.

------------------------------------------------------------------------

## 🤖 Prediction Workflow

``` text
User Vehicle Information
          │
          ▼
Input Validation
          │
          ▼
Feature Preparation
          │
          ▼
Car Age Calculation
          │
          ▼
Saved ML Model
          │
          ▼
Predicted Selling Price
          │
          ▼
Interactive Result
```

The application loads the saved model and does not retrain the model
during normal dashboard usage.

------------------------------------------------------------------------

## 📈 Analytics Workflow

``` text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Data Validation
     │
     ▼
Exploratory Analysis
     │
     ▼
Feature Preparation
     │
     ▼
Machine Learning
     │
     ▼
Model Evaluation
     │
     ▼
Interactive Dashboard
     │
     ▼
Price Prediction & Insights
```

------------------------------------------------------------------------

## 🔍 Key Features

-   📊 Interactive data analysis
-   🚗 Used-car market exploration
-   💰 Machine learning price prediction
-   📈 Interactive Plotly visualizations
-   🎯 Vehicle-specific predictions
-   🔎 Dataset filtering
-   📋 Data exploration
-   🤖 Saved machine learning model
-   🎨 Custom dashboard design
-   ✨ Animated user interface
-   📱 Responsive layout
-   🧪 Model evaluation
-   ✅ Data quality checks

------------------------------------------------------------------------

## 🚀 Future Improvements

-   Add more vehicle datasets.
-   Compare additional regression algorithms.
-   Apply hyperparameter optimization.
-   Add explainable AI for individual predictions.
-   Add prediction confidence intervals.
-   Add model comparison.
-   Add market-price recommendations.
-   Add batch prediction through CSV upload.
-   Deploy the application online.

------------------------------------------------------------------------

## ⚠️ Limitations

The predicted price is an estimated model output and should not be
considered a guaranteed market price.

Actual used-car prices can also depend on factors that may not be fully
represented in the dataset, such as:

-   Vehicle condition
-   Location
-   Accident history
-   Maintenance history
-   Additional features
-   Market demand
-   Seller/buyer negotiation

------------------------------------------------------------------------

## 🌐 Deployment

The application can be deployed using Streamlit Community Cloud after
placing the project in a GitHub repository with the required
dependencies.

For deployment, keep `requirements.txt` in the repository and make sure
the Python version used for deployment is compatible with the
development environment.

------------------------------------------------------------------------

## 📌 Disclaimer

This project is developed for educational, analytical, and machine
learning demonstration purposes.

Predicted prices are estimates generated by a machine learning model and
should not be considered professional vehicle valuation or financial
advice.

------------------------------------------------------------------------

## 👩‍💻 Author

**Noura Maher**

Data Analyst \| Machine Learning Enthusiast \| AI & Data Science

------------------------------------------------------------------------

## ⭐ Project Highlight

> **From raw used-car data to an interactive AI-powered pricing
> experience.**

Built with Python, Machine Learning, Data Analytics, and Streamlit.

------------------------------------------------------------------------

## 📄 License

This project is intended for educational and portfolio purposes.
