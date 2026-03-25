from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI(title="싱크홀 위험도 예측")

templates = Jinja2Templates(directory="templates")

# ================================
# 1) 모델 로드
# ================================
bundle = joblib.load("sinkhole_logi_model.pkl")

model = bundle["model"]
features = bundle["features"]
threshold = bundle["threshold"]
model_name = bundle["model_name"]

# ================================
# 2) 첫 화면
# ================================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "features": features,
            "result": None,
            "prob": None,
            "threshold": threshold,
            "model_name": model_name
        }
    )

# ================================
# 3) 예측 처리
# ================================
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()

    input_dict = {}
    for col in features:
        val = form.get(col, "")
        if val == "" or val is None:
            input_dict[col] = None
        else:
            input_dict[col] = float(val)

    X_input = pd.DataFrame([input_dict], columns=features)

    y_prob = model.predict_proba(X_input)[0, 1]
    y_pred = int(y_prob >= threshold)

    if y_prob >= 0.9:
        risk_label = "매우 위험"
    elif y_prob >= threshold:
        risk_label = "위험"
    elif y_prob >= 0.5:
        risk_label = "주의"
    else:
        risk_label = "낮음"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "features": features,
            "result": y_pred,
            "prob": round(float(y_prob), 4),
            "risk_label": risk_label,
            "threshold": threshold,
            "model_name": model_name,
            "form_data": input_dict
        }
    )