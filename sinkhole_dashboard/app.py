from pathlib import Path
import json
import math
import joblib
import pandas as pd

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="싱크홀 종합 위험도 대시보드")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

print("### dashboard app.py 실행됨 ###")

# -----------------------------
# 1. 모델 로드
# -----------------------------
bundle = joblib.load(BASE_DIR / "sinkhole_logi_model.pkl")
model = bundle["model"]
model_features = bundle["features"]
threshold = float(bundle.get("threshold", 0.9))
model_name = bundle.get("model_name", "LogisticRegression_sinkhole_risk")

# -----------------------------
# 2. 공간틀(geojson) 로드
# -----------------------------
with open(BASE_DIR / "final_df_4326.geojson", "r", encoding="utf-8") as f:
    base_geojson = json.load(f)

base_rows = []
for feature in base_geojson["features"]:
    props = feature.get("properties", {}).copy()
    props["id"] = str(props.get("id"))
    base_rows.append(props)

base_df = pd.DataFrame(base_rows)

# -----------------------------
# 3. full_df 보조 병합 (있으면)
# -----------------------------
if (BASE_DIR / "full_df_1210-1.xlsx").exists():
    full_df = pd.read_excel(BASE_DIR / "full_df_1210-1.xlsx")
    if "id" in full_df.columns:
        full_df["id"] = full_df["id"].astype(str)
        temp_df = base_df.merge(full_df, on="id", how="left", suffixes=("", "__full"))
        for col in full_df.columns:
            if col == "id":
                continue
            full_col = f"{col}__full"
            if full_col in temp_df.columns:
                if col in temp_df.columns:
                    temp_df[col] = temp_df[full_col].combine_first(temp_df[col])
                else:
                    temp_df[col] = temp_df[full_col]
                temp_df.drop(columns=[full_col], inplace=True)
        base_df = temp_df.copy()

# -----------------------------
# 4. 숫자형 컬럼 정리
# -----------------------------
for col in model_features:
    if col in base_df.columns:
        base_df[col] = pd.to_numeric(base_df[col], errors="coerce")

# -----------------------------
# 5. 초기 예측
# -----------------------------
missing_model_cols = [col for col in model_features if col not in base_df.columns]
if missing_model_cols:
    raise RuntimeError(f"공간틀/기본데이터에 모델 컬럼이 없습니다: {missing_model_cols}")

current_df = base_df.copy()
current_prob = model.predict_proba(current_df[model_features])[:, 1]
current_df["risk_prob"] = current_prob

try:
    current_df["risk_level"] = pd.qcut(
        current_df["risk_prob"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)
except Exception:
    bins = [-0.0001, 0.2, 0.4, 0.6, 0.8, 1.0]
    current_df["risk_level"] = pd.cut(
        current_df["risk_prob"],
        bins=bins,
        labels=[1, 2, 3, 4, 5],
        include_lowest=True
    ).astype(int)

current_df["id"] = current_df["id"].astype(str)
current_df_indexed = current_df.set_index("id", drop=False)

# -----------------------------
# 6. 사건 파일 로드
# -----------------------------
event_df = pd.DataFrame()
event_total = 0
event_recent_3m = 0
event_recent_6m = 0
city_top15_labels = []
city_top15_values = []
month_labels = []
month_values = []

event_file = BASE_DIR / "싱크홀발생건수_위도경도_최종.xlsx"

if event_file.exists():
    event_df = pd.read_excel(event_file)
    event_df.columns = [str(c).strip() for c in event_df.columns]

    city_col = None
    date_col = None

    for c in event_df.columns:
        if c in ["SGG_NM", "시군구", "시군구명", "시군", "구", "지역"]:
            city_col = c
            break

    if city_col is None:
        for c in event_df.columns:
            if "주소" in c:
                city_col = c
                break

    for c in event_df.columns:
        if c in ["발생일자", "발생일", "date", "DATE", "일자"]:
            date_col = c
            break

    if city_col is not None:
        if "주소" in city_col:
            parsed_city = []
            for val in event_df[city_col].fillna("").astype(str):
                parts = val.split()
                city_name = ""
                for part in parts:
                    if part.endswith("시") or part.endswith("군") or part.endswith("구"):
                        city_name = part
                        break
                parsed_city.append(city_name)
            event_df["__city__"] = parsed_city
            city_col = "__city__"

        city_counts = event_df[city_col].fillna("").astype(str).value_counts()
        city_counts = city_counts[city_counts.index != ""]
        city_top15 = city_counts.head(15)
        city_top15_labels = city_top15.index.tolist()
        city_top15_values = [int(v) for v in city_top15.values.tolist()]
        event_total = int(len(event_df))

    if date_col is not None:
        event_df["__date__"] = pd.to_datetime(event_df[date_col], errors="coerce")
        valid_dates = event_df["__date__"].dropna()
        if len(valid_dates) > 0:
            max_date = valid_dates.max()
            event_recent_3m = int((event_df["__date__"] >= (max_date - pd.DateOffset(months=3))).sum())
            event_recent_6m = int((event_df["__date__"] >= (max_date - pd.DateOffset(months=6))).sum())

            month_group = (
                event_df.dropna(subset=["__date__"])
                .assign(month=event_df["__date__"].dt.to_period("M").astype(str))
                .groupby("month")
                .size()
                .reset_index(name="count")
                .sort_values("month")
            )

            month_labels = month_group["month"].tolist()
            month_values = [int(v) for v in month_group["count"].tolist()]

# -----------------------------
# 7. 상태값 계산
# -----------------------------
risk_counts = (
    current_df["risk_level"]
    .value_counts()
    .reindex([1, 2, 3, 4, 5], fill_value=0)
)

summary_cards = {
    "total_grid": int(len(current_df)),
    "high_risk_grid": int((current_df["risk_prob"] >= threshold).sum()),
    "avg_risk": round(float(current_df["risk_prob"].mean()), 4),
    "event_total": int(event_total),
    "event_recent_3m": int(event_recent_3m),
    "event_recent_6m": int(event_recent_6m),
}

# -----------------------------
# 8. 화면 라벨
# -----------------------------
layer_labels = {
    "risk": "종합 위험도",
    "population": "인구",
    "building_count": "건물 수",
    "slope_deg": "경사도",
    "rain_sum": "누적 강수",
    "sw_old_rt": "노후 하수도 비율",
}

# -----------------------------
# 9. 메인 페이지
# -----------------------------
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_name": model_name,
            "threshold": threshold,
            "model_features": model_features,
            "layer_labels": layer_labels,
        },
    )

# -----------------------------
# 10. 현재 상태 API
# -----------------------------
@app.get("/api/state")
async def api_state():
    features_out = []

    for feature in base_geojson["features"]:
        gid = str(feature.get("properties", {}).get("id"))
        if gid not in current_df_indexed.index:
            continue

        row = current_df_indexed.loc[gid]
        props = {
            "id": gid,
            "SGG_NM": row["SGG_NM"] if "SGG_NM" in row and pd.notna(row["SGG_NM"]) else "",
            "DONG": row["DONG"] if "DONG" in row and pd.notna(row["DONG"]) else "",
            "risk_prob": None if pd.isna(row["risk_prob"]) else round(float(row["risk_prob"]), 6),
            "risk_level": None if pd.isna(row["risk_level"]) else int(row["risk_level"]),
        }

        for col in model_features:
            if col in row.index:
                val = row[col]
                if pd.isna(val):
                    props[col] = None
                elif isinstance(val, (int, float, bool)) and not isinstance(val, bool):
                    props[col] = float(val)
                else:
                    props[col] = val

        features_out.append(
            {
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": props,
            }
        )

    return JSONResponse(
        {
            "model_name": model_name,
            "threshold": threshold,
            "layer_labels": layer_labels,
            "model_features": model_features,
            "summary_cards": summary_cards,
            "risk_distribution": {
                "labels": ["매우 낮음", "낮음", "보통", "높음", "매우 높음"],
                "values": [int(risk_counts.get(i, 0)) for i in [1, 2, 3, 4, 5]],
            },
            "city_top15": {
                "labels": city_top15_labels,
                "values": city_top15_values,
            },
            "month_series": {
                "labels": month_labels,
                "values": month_values,
            },
            "geojson": {
                "type": "FeatureCollection",
                "features": features_out,
            },
        }
    )

# -----------------------------
# 11. 최신 df 업로드 후 위험도 갱신
# -----------------------------
@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    global current_df, current_df_indexed, summary_cards

    suffix = Path(file.filename).suffix.lower()
    contents = await file.read()

    upload_df = None
    if suffix in [".xlsx", ".xls"]:
        upload_df = pd.read_excel(pd.io.common.BytesIO(contents))
    elif suffix == ".csv":
        upload_df = pd.read_csv(pd.io.common.BytesIO(contents))
    else:
        return JSONResponse(
            {"ok": False, "message": "xlsx, xls, csv 파일만 업로드 가능합니다."},
            status_code=400
        )

    upload_df.columns = [str(c).strip() for c in upload_df.columns]
    if "id" not in upload_df.columns:
        return JSONResponse(
            {"ok": False, "message": "업로드 파일에 id 컬럼이 없습니다."},
            status_code=400
        )

    upload_df["id"] = upload_df["id"].astype(str)

    merged_df = base_df.copy()
    merged_df = merged_df.merge(upload_df, on="id", how="left", suffixes=("", "__new"))

    for col in upload_df.columns:
        if col == "id":
            continue

        new_col = f"{col}__new"
        if new_col in merged_df.columns:
            if col in merged_df.columns:
                merged_df[col] = merged_df[new_col].combine_first(merged_df[col])
            else:
                merged_df[col] = merged_df[new_col]
            merged_df.drop(columns=[new_col], inplace=True)

    for col in model_features:
        if col not in merged_df.columns:
            return JSONResponse(
                {"ok": False, "message": f"모델 컬럼 누락: {col}"},
                status_code=400
            )
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

    pred_prob = model.predict_proba(merged_df[model_features])[:, 1]
    merged_df["risk_prob"] = pred_prob

    try:
        merged_df["risk_level"] = pd.qcut(
            merged_df["risk_prob"].rank(method="first"),
            5,
            labels=[1, 2, 3, 4, 5]
        ).astype(int)
    except Exception:
        bins = [-0.0001, 0.2, 0.4, 0.6, 0.8, 1.0]
        merged_df["risk_level"] = pd.cut(
            merged_df["risk_prob"],
            bins=bins,
            labels=[1, 2, 3, 4, 5],
            include_lowest=True
        ).astype(int)

    merged_df["id"] = merged_df["id"].astype(str)

    current_df = merged_df.copy()
    current_df_indexed = current_df.set_index("id", drop=False)

    summary_cards = {
        "total_grid": int(len(current_df)),
        "high_risk_grid": int((current_df["risk_prob"] >= threshold).sum()),
        "avg_risk": round(float(current_df["risk_prob"].mean()), 4),
        "event_total": int(event_total),
        "event_recent_3m": int(event_recent_3m),
        "event_recent_6m": int(event_recent_6m),
    }

    return JSONResponse(
        {
            "ok": True,
            "message": f"{file.filename} 업로드 반영 완료",
        }
    )