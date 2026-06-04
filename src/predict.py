import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

new_data = pd.DataFrame([{
    "study_hours": 6,
    "attendance": 78,
    "previous_score": 62
}])

prediction = model.predict(new_data)[0]
print("Pass" if prediction == 1 else "Fail")
