from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
import os

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

model = joblib.load(os.path.join(BASE_DIR, "linear_regression_model.pkl"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"prediction": None, "experience": None},
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, experience: float = Form(...)):
    pred = model.predict(np.array([[experience]]))[0]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": round(float(pred), 2),
            "experience": experience,
        },
    )
