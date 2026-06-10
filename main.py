import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.DataFrame({
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Marks": [20,30,40,50,60,70,75,85,90,95]
})

# Features and target
X = data[['Hours']]
y = data['Marks']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)

# Predict new value
hours = np.array([[2.5]])
result = model.predict(hours)

print(f"Predicted Marks: {result[0]:.2f}")

# Plot
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.show()
# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
import joblib
joblib.dump(model, "student_marks_model.pkl")

print("Model saved successfully!")

# Make predictions
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)