
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd

app = FastAPI(title="싱크홀 종합 위험도 대시보드")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# =========================
# 기본 설정
# =========================
MODEL_FILE = "sinkhole_logi_model.pkl"
BASE_GEOJSON_FILE = "final_df_4326.geojson"
EVENT_FILE = "싱크홀발생건수_위도경도_최종.xlsx"
DEFAULT_DF_FILE = "full_df_1210-1.xlsx"

RISK_LABELS = {
    1: "매우 낮음",
    2: "낮음",
    3: "보통",
    4: "높음",
    5: "매우 높음"
}

RISK_COLORS = {
    1: "#2ecc71",
    2: "#9be15d",
    3: "#f1c40f",
    4: "#f39c12",
    5: "#e74c3c"
}

# =========================
# 전역 데이터
# =========================
bundle = None
model = None
features = []
threshold = 0.5
model_name = ""
base_gdf = None
event_df = None
default_df = None

# =========================
# 앱 시작 시 1회 로드
# =========================
@app.on_event("startup")
async def startup_event():
    global bundle, model, features, threshold, model_name, base_gdf, event_df, default_df

    if os.path.exists(MODEL_FILE):
        bundle = joblib.load(MODEL_FILE)
        model = bundle["model"]
        features = bundle["features"]
        threshold = float(bundle.get("threshold", 0.5))
        model_name = bundle.get("model_name", "LogisticRegression_sinkhole_risk")
    else:
        model = None
        features = []
        threshold = 0.5
        model_name = "모델 파일 없음"

    if os.path.exists(BASE_GEOJSON_FILE):
        base_gdf = gpd.read_file(BASE_GEOJSON_FILE)
        if "id" in base_gdf.columns:
            base_gdf["id"] = base_gdf["id"].astype(str)
    else:
        base_gdf = None

    if os.path.exists(EVENT_FILE):
        event_df = pd.read_excel(EVENT_FILE)
    else:
        event_df = pd.DataFrame()

    if os.path.exists(DEFAULT_DF_FILE):
        default_df = pd.read_excel(DEFAULT_DF_FILE)
        if "id" in default_df.columns:
            default_df["id"] = default_df["id"].astype(str)
    else:
        default_df = pd.DataFrame()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_name": model_name,
            "threshold": threshold,
            "features": features
        }
    )

@app.get("/api/initial-data")
async def initial_data():
    if base_gdf is None:
        return JSONResponse(
            {
                "ok": False,
                "message": "공간 틀 로드 실패: final_df_4326.geojson 파일을 프로젝트 폴더에 넣어야 합니다.",
                "geojson": None,
                "stats": {
                    "total_grid": 0,
                    "high_risk_grid": 0,
                    "avg_risk": 0,
                    "total_events": 0,
                    "recent_3m": 0,
                    "recent_6m": 0
                },
                "risk_distribution": [],
                "sgg_top15": [],
                "monthly_trend": [],
                "feature_list": features,
                "current_layer": "종합 위험도"
            }
        )

    work_df = default_df.copy()

    if work_df.empty:
        merged = base_gdf.copy()
        merged["risk_prob"] = 0.0
        merged["risk_level"] = 1
        merged["risk_label"] = "매우 낮음"
        merged["fill_color"] = RISK_COLORS[1]
    else:
        work_df["id"] = work_df["id"].astype(str)
        merged = base_gdf.merge(work_df[["id"] + [c for c in features if c in work_df.columns]], on="id", how="left")

        if model is not None and len(features) > 0:
            X_pred = merged[features].apply(pd.to_numeric, errors="coerce")
            pred_prob = model.predict_proba(X_pred)[:, 1]
            merged["risk_prob"] = pred_prob

            q20 = float(np.nanquantile(pred_prob, 0.2))
            q40 = float(np.nanquantile(pred_prob, 0.4))
            q60 = float(np.nanquantile(pred_prob, 0.6))
            q80 = float(np.nanquantile(pred_prob, 0.8))

            merged["risk_level"] = np.select(
                [
                    merged["risk_prob"] <= q20,
                    merged["risk_prob"] <= q40,
                    merged["risk_prob"] <= q60,
                    merged["risk_prob"] <= q80
                ],
                [1, 2, 3, 4],
                default=5
            ).astype(int)
            merged["risk_label"] = merged["risk_level"].map(RISK_LABELS)
            merged["fill_color"] = merged["risk_level"].map(RISK_COLORS)
        else:
            merged["risk_prob"] = 0.0
            merged["risk_level"] = 1
            merged["risk_label"] = "매우 낮음"
            merged["fill_color"] = RISK_COLORS[1]

    total_events = 0
    recent_3m = 0
    recent_6m = 0
    sgg_top15 = []
    monthly_trend = []

    if not event_df.empty:
        ev = event_df.copy()

        if "발생일자" in ev.columns:
            ev["발생일자"] = pd.to_datetime(ev["발생일자"], errors="coerce")
        elif "발생일" in ev.columns:
            ev["발생일자"] = pd.to_datetime(ev["발생일"], errors="coerce")

        total_events = int(len(ev))

        if "발생일자" in ev.columns and ev["발생일자"].notna().any():
            max_dt = ev["발생일자"].max()
            recent_3m = int((ev["발생일자"] >= max_dt - pd.DateOffset(months=3)).sum())
            recent_6m = int((ev["발생일자"] >= max_dt - pd.DateOffset(months=6)).sum())

            month_series = (
                ev.dropna(subset=["발생일자"])
                  .assign(month=lambda x: x["발생일자"].dt.to_period("M").astype(str))
                  .groupby("month")
                  .size()
                  .reset_index(name="count")
                  .sort_values("month")
            )
            monthly_trend = month_series.tail(12).to_dict(orient="records")

        sgg_col = None
        for c in ["시군구", "SGG_NM", "시군구명", "주소"]:
            if c in ev.columns:
                sgg_col = c
                break

        if sgg_col is not None:
            if sgg_col == "주소":
                ev["시군구추출"] = ev["주소"].astype(str).str.extract(r'([가-힣]+시\s?[가-힣]+구|[가-힣]+시|[가-힣]+군)')
                sgg_group = ev.groupby("시군구추출").size().reset_index(name="count")
                sgg_group.columns = ["name", "count"]
            else:
                sgg_group = ev.groupby(sgg_col).size().reset_index(name="count")
                sgg_group.columns = ["name", "count"]

            sgg_group = sgg_group.sort_values("count", ascending=False).head(15)
            sgg_top15 = sgg_group.to_dict(orient="records")

    risk_distribution = (
        merged.groupby("risk_level").size().reindex([1, 2, 3, 4, 5], fill_value=0).reset_index(name="count")
    )
    risk_distribution["label"] = risk_distribution["risk_level"].map(RISK_LABELS)
    risk_distribution["color"] = risk_distribution["risk_level"].map(RISK_COLORS)

    merged["tooltip_html"] = (
        "<b>id:</b> " + merged["id"].astype(str)
        + "<br><b>시군구:</b> " + merged.get("SGG_NM", pd.Series([""] * len(merged))).astype(str)
        + "<br><b>행정동:</b> " + merged.get("DONG", pd.Series([""] * len(merged))).astype(str)
        + "<br><b>위험확률:</b> " + merged["risk_prob"].round(4).astype(str)
        + "<br><b>위험등급:</b> " + merged["risk_label"].astype(str)
    )

    geojson = json.loads(merged.to_json())

    return JSONResponse(
        {
            "ok": True,
            "message": "초기 데이터 로드 완료",
            "geojson": geojson,
            "stats": {
                "total_grid": int(len(merged)),
                "high_risk_grid": int((merged["risk_level"] >= 4).sum()),
                "avg_risk": round(float(merged["risk_prob"].mean()), 4),
                "total_events": total_events,
                "recent_3m": recent_3m,
                "recent_6m": recent_6m
            },
            "risk_distribution": risk_distribution.to_dict(orient="records"),
            "sgg_top15": sgg_top15,
            "monthly_trend": monthly_trend,
            "feature_list": features,
            "current_layer": "종합 위험도"
        }
    )

@app.post("/api/upload-latest-df")
async def upload_latest_df(file: UploadFile = File(...)):
    if base_gdf is None:
        return JSONResponse({"ok": False, "message": "final_df_4326.geojson 공간 틀이 없습니다."}, status_code=400)

    if model is None:
        return JSONResponse({"ok": False, "message": "sinkhole_logi_model.pkl 모델 파일이 없습니다."}, status_code=400)

    filename = file.filename.lower()

    if filename.endswith(".xlsx"):
        uploaded_df = pd.read_excel(io.BytesIO(await file.read()))
    elif filename.endswith(".csv"):
        uploaded_df = pd.read_csv(io.BytesIO(await file.read()))
    else:
        return JSONResponse({"ok": False, "message": "xlsx 또는 csv 파일만 업로드 가능합니다."}, status_code=400)

    if "id" not in uploaded_df.columns:
        return JSONResponse({"ok": False, "message": "업로드 파일에 id 컬럼이 없습니다."}, status_code=400)

    uploaded_df["id"] = uploaded_df["id"].astype(str)

    missing_cols = [c for c in features if c not in uploaded_df.columns]
    if missing_cols:
        return JSONResponse(
            {
                "ok": False,
                "message": f"모델 입력 컬럼이 부족합니다. 누락 컬럼 예시: {missing_cols[:10]}"
            },
            status_code=400
        )

    merged = base_gdf.merge(uploaded_df[["id"] + features], on="id", how="left")

    X_pred = merged[features].apply(pd.to_numeric, errors="coerce")
    pred_prob = model.predict_proba(X_pred)[:, 1]
    merged["risk_prob"] = pred_prob

    q20 = float(np.nanquantile(pred_prob, 0.2))
    q40 = float(np.nanquantile(pred_prob, 0.4))
    q60 = float(np.nanquantile(pred_prob, 0.6))
    q80 = float(np.nanquantile(pred_prob, 0.8))

    merged["risk_level"] = np.select(
        [
            merged["risk_prob"] <= q20,
            merged["risk_prob"] <= q40,
            merged["risk_prob"] <= q60,
            merged["risk_prob"] <= q80
        ],
        [1, 2, 3, 4],
        default=5
    ).astype(int)
    merged["risk_label"] = merged["risk_level"].map(RISK_LABELS)
    merged["fill_color"] = merged["risk_level"].map(RISK_COLORS)

    layer_col = "종합 위험도"

    if not event_df.empty:
        ev = event_df.copy()

        if "발생일자" in ev.columns:
            ev["발생일자"] = pd.to_datetime(ev["발생일자"], errors="coerce")
        elif "발생일" in ev.columns:
            ev["발생일자"] = pd.to_datetime(ev["발생일"], errors="coerce")

        total_events = int(len(ev))

        if "발생일자" in ev.columns and ev["발생일자"].notna().any():
            max_dt = ev["발생일자"].max()
            recent_3m = int((ev["발생일자"] >= max_dt - pd.DateOffset(months=3)).sum())
            recent_6m = int((ev["발생일자"] >= max_dt - pd.DateOffset(months=6)).sum())

            month_series = (
                ev.dropna(subset=["발생일자"])
                  .assign(month=lambda x: x["발생일자"].dt.to_period("M").astype(str))
                  .groupby("month")
                  .size()
                  .reset_index(name="count")
                  .sort_values("month")
            )
            monthly_trend = month_series.tail(12).to_dict(orient="records")
        else:
            recent_3m = 0
            recent_6m = 0
            monthly_trend = []

        sgg_col = None
        for c in ["시군구", "SGG_NM", "시군구명", "주소"]:
            if c in ev.columns:
                sgg_col = c
                break

        if sgg_col is not None:
            if sgg_col == "주소":
                ev["시군구추출"] = ev["주소"].astype(str).str.extract(r'([가-힣]+시\s?[가-힣]+구|[가-힣]+시|[가-힣]+군)')
                sgg_group = ev.groupby("시군구추출").size().reset_index(name="count")
                sgg_group.columns = ["name", "count"]
            else:
                sgg_group = ev.groupby(sgg_col).size().reset_index(name="count")
                sgg_group.columns = ["name", "count"]

            sgg_group = sgg_group.sort_values("count", ascending=False).head(15)
            sgg_top15 = sgg_group.to_dict(orient="records")
        else:
            sgg_top15 = []
    else:
        total_events = 0
        recent_3m = 0
        recent_6m = 0
        sgg_top15 = []
        monthly_trend = []

    risk_distribution = (
        merged.groupby("risk_level").size().reindex([1, 2, 3, 4, 5], fill_value=0).reset_index(name="count")
    )
    risk_distribution["label"] = risk_distribution["risk_level"].map(RISK_LABELS)
    risk_distribution["color"] = risk_distribution["risk_level"].map(RISK_COLORS)

    merged["tooltip_html"] = (
        "<b>id:</b> " + merged["id"].astype(str)
        + "<br><b>시군구:</b> " + merged.get("SGG_NM", pd.Series([""] * len(merged))).astype(str)
        + "<br><b>행정동:</b> " + merged.get("DONG", pd.Series([""] * len(merged))).astype(str)
        + "<br><b>위험확률:</b> " + merged["risk_prob"].round(4).astype(str)
        + "<br><b>위험등급:</b> " + merged["risk_label"].astype(str)
    )

    geojson = json.loads(merged.to_json())

    return JSONResponse(
        {
            "ok": True,
            "message": f"{file.filename} 업로드 반영 완료",
            "geojson": geojson,
            "stats": {
                "total_grid": int(len(merged)),
                "high_risk_grid": int((merged["risk_level"] >= 4).sum()),
                "avg_risk": round(float(merged["risk_prob"].mean()), 4),
                "total_events": total_events,
                "recent_3m": recent_3m,
                "recent_6m": recent_6m
            },
            "risk_distribution": risk_distribution.to_dict(orient="records"),
            "sgg_top15": sgg_top15,
            "monthly_trend": monthly_trend,
            "feature_list": features,
            "current_layer": layer_col
        }
    )
