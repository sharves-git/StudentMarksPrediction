import pandas as pd
import joblib

model = joblib.load("student_marks_model.pkl")

hours = pd.DataFrame({'Hours': [8]})

prediction = model.predict(hours)

print("Predicted Marks:", prediction[0])