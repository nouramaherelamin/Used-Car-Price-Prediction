import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
df=pd.read_csv("data/raw/car data.csv").drop_duplicates().copy()
df["Car_Age"]=df["Year"].max()-df["Year"]
features=["Year","Present_Price","Driven_kms","Fuel_Type","Selling_type","Transmission","Owner","Car_Age"]
X,y=df[features],df["Selling_Price"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
num=["Year","Present_Price","Driven_kms","Owner","Car_Age"]; cat=["Fuel_Type","Selling_type","Transmission"]
pre=ColumnTransformer([("num",SimpleImputer(strategy="median"),num),("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),cat)])
pipe=Pipeline([("preprocess",pre),("model",GradientBoostingRegressor(random_state=42,n_estimators=300,max_depth=2,learning_rate=.04,loss="huber"))])
pipe.fit(X_train,y_train)
joblib.dump(pipe,"models/best_car_price_model.joblib")
print("Saved models/best_car_price_model.joblib")
