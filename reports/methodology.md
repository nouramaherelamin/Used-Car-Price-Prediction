# Methodology

The project is organized around CRISP-DM: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment. IBM describes these as an iterative data-mining lifecycle.

The implementation uses:
- duplicate removal
- engineered `Car_Age`
- one-hot encoding for categorical features
- median imputation for numeric inputs as a pipeline safeguard
- 80/20 fixed holdout split
- 5-fold shuffled cross-validation
- MAE, RMSE, and R²
- serialized preprocessing + model pipeline
