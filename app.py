import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import uvicorn

import src.predict as predictor

app = FastAPI(title="AI Phishing Detector")
templates = Jinja2Templates(directory="templates")


class PredictRequest(BaseModel):
    text: str = Field(min_length=20)
    threshold: float = Field(default=0.4, gt=0, lt=1)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/predict")
async def predict(body: PredictRequest):
    try:
        result = predictor.predict_email(body.text, threshold=body.threshold)
        return result
    except FileNotFoundError:
        return JSONResponse(
            {"error": "Model not found. Train the model first."}, status_code=503
        )
    except Exception as e:
        return JSONResponse({"error": f"Internal error: {str(e)}"}, status_code=500)


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=debug)