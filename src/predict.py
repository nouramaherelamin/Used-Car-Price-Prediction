import joblib, pandas as pd
MODEL="models/best_car_price_model.joblib"
model=joblib.load(MODEL)
def predict_price(year,present_price,driven_kms,fuel_type,selling_type,transmission,owner):
    car_age=max(0, pd.Timestamp.now().year-year)
    row=pd.DataFrame([{"Year":year,"Present_Price":present_price,"Driven_kms":driven_kms,
                       "Fuel_Type":fuel_type,"Selling_type":selling_type,
                       "Transmission":transmission,"Owner":owner,"Car_Age":car_age}])
    return float(model.predict(row)[0])
