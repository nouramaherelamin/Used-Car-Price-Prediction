# Used Car Price Prediction — Final Business & Technical Report

## Executive Summary
This project builds an end-to-end machine-learning solution for estimating used-car selling prices from vehicle attributes.

## Dataset
- Raw rows: 301
- Columns: 9
- Duplicate rows: 2
- Missing cells: 0
- Modeling rows after duplicate removal: 299
- Target: `Selling_Price`

## Data Preparation
Two exact duplicate rows were removed. `Car_Name` remains in the source data for reference but is excluded from the baseline model because it is a high-cardinality name field. `Car_Age` was engineered from the maximum observed year.

## Modeling
Compared models:
- Linear Regression
- Random Forest
- Extra Trees
- Gradient Boosting

Five-fold shuffled cross-validation was used alongside a fixed 80/20 holdout split.

## Selected Model
**Gradient Boosting Regressor**

5-fold CV mean RMSE: 1.460

Held-out test performance:
- MAE: 1.091
- RMSE: 2.265
- R²: 0.801

## Business Interpretation
The model provides a repeatable estimate that can support used-car listing and pricing workflows. The estimate should be interpreted together with vehicle condition, service history, location, and current market conditions because these variables are not represented in the dataset.

## Deployment
A Streamlit dashboard is included in `app/app.py`. The serialized preprocessing + model pipeline is stored in `models/best_car_price_model.joblib`.

## Limitations
The dataset is small and does not document the valuation date, geography, vehicle condition, or complete market context. Therefore, this is a portfolio/analytical model and should not be treated as a live market pricing engine without additional validation.

## Reproducibility
Run `python src/train_model.py` to rebuild the model, then run:
`streamlit run app/app.py`
