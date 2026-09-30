# Used Car Price Prediction

An end-to-end machine-learning project that analyzes used-car attributes and predicts `Selling_Price` through a reproducible preprocessing and regression pipeline.

## Project Goal
Estimate used-car selling prices from:
- Year
- Present Price
- Driven Kilometers
- Fuel Type
- Selling Type
- Transmission
- Previous Owners
- Engineered Car Age

## Dataset
The supplied dataset contains **301 rows and 9 columns**. It has no missing values and contains **2 exact duplicate rows**, which are removed before modeling.

## Modeling
Four regression models were compared:
1. Linear Regression
2. Random Forest
3. Extra Trees
4. Gradient Boosting

Model selection uses a fixed 80/20 holdout plus 5-fold cross-validation. The final selected model is **Gradient Boosting Regressor**.

### Final Holdout Metrics
| Metric | Value |
|---|---:|
| MAE | 1.091 |
| RMSE | 2.265 |
| R² | 0.801 |

## Folder Structure
```text
Used_Car_Price_Prediction_FINAL_SUBMISSION/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   │   └── car data.csv
│   └── processed/
│       └── car_data_clean.csv
├── models/
│   └── best_car_price_model.joblib
├── notebooks/
│   └── Used_Car_Price_Prediction_Final.ipynb
├── reports/
│   ├── final_report.md
│   ├── data_quality_summary.csv
│   ├── model_comparison.csv
│   ├── cross_validation_results.csv
│   ├── test_predictions.csv
│   └── figures/
├── src/
│   ├── train_model.py
│   └── predict.py
├── tests/
├── requirements.txt
└── README.md
```

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Rebuild Model
```bash
python src/train_model.py
```

## Notes
`Car_Name` is preserved for reference but excluded from the baseline model because its high cardinality can encourage memorization. The model is a decision-support estimator and is not a guarantee of current market value.

## Methodology
The project is organized around the CRISP-DM lifecycle: business understanding, data understanding, data preparation, modeling, evaluation, and deployment. IBM describes CRISP-DM as an iterative data-mining lifecycle rather than a strictly linear sequence.
