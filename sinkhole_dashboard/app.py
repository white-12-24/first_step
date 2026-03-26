from pathlib import Path
import json
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

# =========================================================
# [백엔드 조절 포인트 1]
# 고위험 판정 기준 threshold
# 고위험 Grid 개수 계산에 사용됨
# =========================================================
bundle = joblib.load(BASE_DIR / "sinkhole_logi_model.pkl")
model = bundle["model"]
model_features = bundle["features"]
threshold = float(bundle.get("threshold", 0.9))
model_name = bundle.get("model_name", "LogisticRegression_sinkhole_risk")

# =========================================================
# [백엔드 조절 포인트 2]
# 지역 컬럼명
# 현재는 공간틀/테이블에 SGG_NM 컬럼을 지역명으로 사용
# 다른 컬럼을 쓰고 싶으면 여기만 바꾸면 됨
# =========================================================
REGION_COL = "SGG_NM"

# =========================================================
# [백엔드 조절 포인트 3]
# 왼쪽 표시 모드 목록
# key는 프론트에서 사용하는 값
# value는 화면에 표시되는 이름
# =========================================================
layer_labels = {
    "risk": "종합 위험도",
    "population": "인구",
    "building_count": "건물 수",
    "slope_deg": "경사도",
    "rain_sum": "누적 강수",
    "sw_old_rt": "노후 하수도 비율",
}

# =========================================================
# 공간틀(geojson) 로드
# =========================================================
with open(BASE_DIR / "final_df_4326.geojson", "r", encoding="utf-8") as f:
    base_geojson = json.load(f)

base_rows = []
for feature in base_geojson["features"]:
    props = feature.get("properties", {}).copy()
    props["id"] = str(props.get("id"))
    base_rows.append(props)

base_df = pd.DataFrame(base_rows)

# =========================================================
# full_df 병합
# 공간틀 속성 + full_df 속성을 id 기준으로 합침
# =========================================================
if (BASE_DIR / "full_df_1210-1.xlsx").exists():
    full_df = pd.read_excel(BASE_DIR / "full_df_1210-1.xlsx")

    if "id" in full_df.columns:
        full_df["id"] = full_df["id"].astype(str)

        merged_temp = base_df.merge(full_df, on="id", how="left", suffixes=("", "__full"))

        for col in full_df.columns:
            if col == "id":
                continue

            full_col = f"{col}__full"
            if full_col in merged_temp.columns:
                if col in merged_temp.columns:
                    merged_temp[col] = merged_temp[full_col].combine_first(merged_temp[col])
                else:
                    merged_temp[col] = merged_temp[full_col]
                merged_temp.drop(columns=[full_col], inplace=True)

        base_df = merged_temp.copy()

# =========================================================
# 숫자형 컬럼 정리
# =========================================================
for col in model_features:
    if col in base_df.columns:
        base_df[col] = pd.to_numeric(base_df[col], errors="coerce")

missing_model_cols = [col for col in model_features if col not in base_df.columns]
if missing_model_cols:
    raise RuntimeError(f"공간틀/기본데이터에 모델 컬럼이 없습니다: {missing_model_cols}")

# =========================================================
# 초기 예측
# =========================================================
current_df = base_df.copy()
current_prob = model.predict_proba(current_df[model_features])[:, 1]
current_df["risk_prob"] = current_prob

# =========================================================
# [백엔드 조절 포인트 4]
# 위험도 5단계 생성 방식
# 기본: qcut 5등분
# 실패 시 고정구간(0.2 단위) 사용
# =========================================================
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

# =========================================================
# 사건 파일 로드
# =========================================================
event_total = 0
event_recent_3m = 0
event_recent_6m = 0

city_top10_labels = []
city_top10_values = []

# =========================================================
# [백엔드 조절 포인트 5]
# 월별 차트 x축 라벨
# 현재 1월 ~ 12월
# =========================================================
month_labels = [f"{i}월" for i in range(1, 13)]
month_values = [0] * 12

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

        # =====================================================
        # [백엔드 조절 포인트 6]
        # 지역별 발생건수 차트 개수
        # 현재 TOP 10
        # 5, 8, 15 등으로 변경 가능
        # =====================================================
        city_top10 = city_counts.head(10)

        city_top10_labels = city_top10.index.tolist()
        city_top10_values = [int(v) for v in city_top10.values.tolist()]
        event_total = int(len(event_df))

    if date_col is not None:
        # =====================================================
        # [백엔드 조절 포인트 7]
        # 발생일자 처리 방식
        # 예: 20180126 -> 가운데 2자리 "01"만 추출해서 월별 집계
        # =====================================================
        raw_date_str = event_df[date_col].fillna("").astype(str).str.replace(".0", "", regex=False).str.strip()
        raw_date_str = raw_date_str.str.replace(r"[^0-9]", "", regex=True)
        raw_date_str = raw_date_str.str.zfill(8)

        event_df["__date__"] = pd.to_datetime(raw_date_str, format="%Y%m%d", errors="coerce")

        valid_dates = event_df["__date__"].dropna()
        if len(valid_dates) > 0:
            max_date = valid_dates.max()
            event_recent_3m = int((event_df["__date__"] >= (max_date - pd.DateOffset(months=3))).sum())
            event_recent_6m = int((event_df["__date__"] >= (max_date - pd.DateOffset(months=6))).sum())

        month_str = raw_date_str.str.slice(4, 6)
        month_num = pd.to_numeric(month_str, errors="coerce")

        month_values = []
        for m in range(1, 13):
            month_values.append(int((month_num == m).sum()))

# =========================================================
# 요약값 계산
# =========================================================
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

# =========================================================
# [백엔드 조절 포인트 8]
# 왼쪽 지역 목록 생성 기준
# 현재는 current_df[SGG_NM] 기준 고유값 정렬
# =========================================================
region_labels = []
if REGION_COL in current_df.columns:
    region_labels = sorted(
        [str(v) for v in current_df[REGION_COL].dropna().astype(str).unique().tolist() if str(v).strip() != ""]
    )

# =========================================================
# 메인 페이지
# =========================================================
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_name": model_name,
            "threshold": threshold,
            "layer_labels": layer_labels,
        },
    )

# =========================================================
# 현재 상태 API
# =========================================================
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
            "SGG_NM": row[REGION_COL] if REGION_COL in row and pd.notna(row[REGION_COL]) else "",
            "DONG": row["DONG"] if "DONG" in row and pd.notna(row["DONG"]) else "",
            "risk_prob": None if pd.isna(row["risk_prob"]) else round(float(row["risk_prob"]), 6),
            "risk_level": None if pd.isna(row["risk_level"]) else int(row["risk_level"]),
        }

        for col in model_features:
            if col in row.index:
                val = row[col]
                if pd.isna(val):
                    props[col] = None
                elif isinstance(val, (int, float)) and not isinstance(val, bool):
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
            "regions": region_labels,
            "summary_cards": summary_cards,
            "risk_distribution": {
                "labels": ["매우 낮음", "낮음", "보통", "높음", "매우 높음"],
                "values": [int(risk_counts.get(i, 0)) for i in [1, 2, 3, 4, 5]],
                "levels": [1, 2, 3, 4, 5],
            },
            "city_top10": {
                "labels": city_top10_labels,
                "values": city_top10_values,
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

# =========================================================
# 최신 df 업로드 후 갱신
# =========================================================
@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    global current_df, current_df_indexed, summary_cards, region_labels

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

    if REGION_COL in current_df.columns:
        region_labels = sorted(
            [str(v) for v in current_df[REGION_COL].dropna().astype(str).unique().tolist() if str(v).strip() != ""]
        )

    return JSONResponse(
        {
            "ok": True,
            "message": f"{file.filename} 업로드 반영 완료",
        }
    )