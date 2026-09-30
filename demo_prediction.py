# Optional: quick prediction demo

import joblib
import pandas as pd

model = joblib.load("results/final_random_forest.joblib")

sample = pd.DataFrame([{
    "category": "shopping_net",
    "amt": 250000.0,
    "gender": "M",
    "state":"NY",
    "lat": 40.7128,
    "long": -74.0060,
    "city_pop": 100000,
    "merch_lat": 40.7306,
    "merch_long": -73.9352,
    "transaction_hour": 23,
    "transaction_day": 30,
    "transaction_month": 9,
    "transaction_dayofweek": 2
}])

prediction = model.predict(sample)[0]

if prediction == 1:
    print("Prediction: FRAUDULENT TRANSACTION")
else:
    print("Prediction: LEGITIMATE TRANSACTION")
