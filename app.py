from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("linear_regression_model.pkl")

@app.get("/")
def home():
    return {"message": "Linear Regression Model API"}

@app.get("/predict")
def predict(experience: float):
    prediction = model.predict(np.array([[experience]]))
    return {
        "Years of Experience": experience,
        "Predicted Salary": round(float(prediction[0]), 2)
    }