from pathlib import Path
import pandas as pd
import joblib

BASE=Path(__file__).resolve().parents[1]

def test_dataset_exists():
    df=pd.read_csv(BASE/"data/raw/car data.csv")
    assert df.shape[0] == 301
    assert "Selling_Price" in df.columns

def test_model_loads():
    model=joblib.load(BASE/"models/best_car_price_model.joblib")
    assert hasattr(model,"predict")

def test_clean_data():
    df=pd.read_csv(BASE/"data/processed/car_data_clean.csv")
    assert df.duplicated().sum()==0
    assert df.isna().sum().sum()==0
