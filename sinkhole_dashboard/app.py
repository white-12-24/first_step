from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import geopandas as gpd
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

MODEL_PATH = BASE_DIR / "sinkhole_logi_model.pkl"
DEFAULT_GRID_INPUT_PATH = BASE_DIR / "full_df_1210-1.xlsx"
DEFAULT_EVENT_PATH = BASE_DIR / "싱크홀발생건수_위도경도_최종.xlsx"
SPATIAL_TEMPLATE_GEOJSON = BASE_DIR / "final_df_4326.geojson"
SPATIAL_TEMPLATE_GPKG = BASE_DIR / "final_df.gpkg"
SPATIAL_LAYER_NAME = "final_df"

app = FastAPI(title="싱크홀 종합 위험도 대시보드")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]
THRESHOLD = float(bundle["threshold"])
MODEL_NAME = bundle.get("model_name", "LogisticRegression")
MODEL_CREATED_AT = bundle.get("created_at", "")

state = {
    "spatial_template": None,
    "current_input_df": None,
    "risk_gdf": None,
    "event_df": None,
    "message": "",
}


# ------------------------------
# 유틸
# ------------------------------
def to_str_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def normalize_sgg_from_address(address: str) -> Optional[str]:
    if pd.isna(address):
        return None
    address = str(address).strip()
    for token in address.split():
        if token.endswith(("시", "군", "구")):
            return token
    return None


def parse_event_date(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(".0", "", regex=False).str.strip()
    s = s.replace({"nan": None, "None": None, "": None})
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")


def risk_level_from_prob(prob: pd.Series) -> pd.Series:
    bins = [-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf]
    labels = [1, 2, 3, 4, 5]
    return pd.cut(prob, bins=bins, labels=labels).astype("Int64")


def risk_label_from_level(level: pd.Series) -> pd.Series:
    mapper = {1: "매우 낮음", 2: "낮음", 3: "보통", 4: "높음", 5: "매우 높음"}
    return level.map(mapper)


def color_from_level(level: int) -> str:
    return {
        1: "#2E8B57",
        2: "#7BC96F",
        3: "#F2D64B",
        4: "#F39C34",
        5: "#D64541",
    }.get(int(level or 1), "#2E8B57")


def load_dataframe(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)


def load_spatial_template() -> gpd.GeoDataFrame:
    if SPATIAL_TEMPLATE_GEOJSON.exists():
        gdf = gpd.read_file(SPATIAL_TEMPLATE_GEOJSON)
    elif SPATIAL_TEMPLATE_GPKG.exists():
        gdf = gpd.read_file(SPATIAL_TEMPLATE_GPKG, layer=SPATIAL_LAYER_NAME)
        if gdf.crs is not None and str(gdf.crs).lower() != "epsg:4326":
            gdf = gdf.to_crs(epsg=4326)
    else:
        raise FileNotFoundError(
            "final_df_4326.geojson 또는 final_df.gpkg 파일을 프로젝트 폴더에 넣어야 합니다."
        )

    gdf.columns = [str(c) for c in gdf.columns]
    if "id" not in gdf.columns:
        raise ValueError("공간 틀 파일에 id 컬럼이 없습니다.")
    gdf["id"] = to_str_id(gdf["id"])
    return gdf


def validate_input_columns(df: pd.DataFrame):
    missing = [col for col in FEATURES if col not in df.columns]
    if "id" not in df.columns:
        missing = ["id"] + missing
    return missing


def build_predictions(input_df: pd.DataFrame, spatial_template: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = input_df.copy()
    df.columns = [str(c) for c in df.columns]
    df["id"] = to_str_id(df["id"])

    work = df[["id"] + [c for c in FEATURES]].copy()
    for col in FEATURES:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    probs = model.predict_proba(work[FEATURES])[:, 1]
    work["risk_prob"] = probs
    work["risk_level"] = risk_level_from_prob(work["risk_prob"])
    work["risk_label"] = risk_label_from_level(work["risk_level"])

    base_cols = [c for c in ["id", "SGG_NM", "DONG"] if c in spatial_template.columns]
    base = spatial_template[base_cols + ["geometry"]].copy()
    merged = base.merge(work, on="id", how="left")
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=spatial_template.crs)
    merged["display_name"] = (
        merged.get("SGG_NM", pd.Series(index=merged.index, dtype=object)).fillna("")
        + " "
        + merged.get("DONG", pd.Series(index=merged.index, dtype=object)).fillna("")
    ).str.strip()
    return merged


def summarize_grid(risk_gdf: gpd.GeoDataFrame):
    if risk_gdf is None or len(risk_gdf) == 0:
        return {
            "total_grid_count": 0,
            "high_risk_grid_count": 0,
            "avg_risk_prob": 0,
            "risk_distribution": [],
            "sgg_summary": [],
        }

    valid = risk_gdf.dropna(subset=["risk_prob"]).copy()
    total = int(len(valid))
    high = int((valid["risk_level"] >= 4).sum())
    avg_prob = round(float(valid["risk_prob"].mean()), 4) if total else 0

    dist = (
        valid["risk_level"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("risk_level")
        .reset_index(name="count")
    )
    dist["label"] = dist["risk_level"].map({1: "매우 낮음", 2: "낮음", 3: "보통", 4: "높음", 5: "매우 높음"})

    top_sgg = []
    if "SGG_NM" in valid.columns:
        tmp = (
            valid.groupby("SGG_NM", dropna=False)
            .agg(
                grid_count=("id", "size"),
                avg_risk_prob=("risk_prob", "mean"),
                high_risk_count=("risk_level", lambda s: int((s >= 4).sum())),
            )
            .reset_index()
            .sort_values(["avg_risk_prob", "high_risk_count"], ascending=False)
        )
        tmp["avg_risk_prob"] = tmp["avg_risk_prob"].round(4)
        top_sgg = tmp.head(15).to_dict(orient="records")

    return {
        "total_grid_count": total,
        "high_risk_grid_count": high,
        "avg_risk_prob": avg_prob,
        "risk_distribution": dist.to_dict(orient="records"),
        "sgg_summary": top_sgg,
    }


def summarize_events(event_df: Optional[pd.DataFrame]):
    if event_df is None or len(event_df) == 0:
        return {
            "total_events": 0,
            "recent_3m_events": 0,
            "recent_6m_events": 0,
            "sgg_event_counts": [],
            "monthly_counts": [],
            "event_points": [],
            "max_date": None,
        }

    work = event_df.copy()
    if "발생일자" in work.columns:
        work["event_date"] = parse_event_date(work["발생일자"])
    else:
        work["event_date"] = pd.NaT

    if "SGG_NM" not in work.columns:
        if "주소" in work.columns:
            work["SGG_NM"] = work["주소"].apply(normalize_sgg_from_address)
        else:
            work["SGG_NM"] = None

    if "위도" in work.columns:
        work["위도"] = pd.to_numeric(work["위도"], errors="coerce")
    if "경도" in work.columns:
        work["경도"] = pd.to_numeric(work["경도"], errors="coerce")

    max_date = work["event_date"].max()
    recent_3m = 0
    recent_6m = 0
    monthly_counts = []
    if pd.notna(max_date):
        recent_3m = int((work["event_date"] >= (max_date - pd.DateOffset(months=3))).sum())
        recent_6m = int((work["event_date"] >= (max_date - pd.DateOffset(months=6))).sum())
        monthly = (
            work.dropna(subset=["event_date"])
            .assign(period=lambda d: d["event_date"].dt.to_period("M").astype(str))
            .groupby("period")
            .size()
            .reset_index(name="count")
            .sort_values("period")
        )
        monthly_counts = monthly.to_dict(orient="records")

    sgg_counts = (
        work.groupby("SGG_NM", dropna=False)
        .size()
        .reset_index(name="event_count")
        .sort_values("event_count", ascending=False)
    )

    point_cols = [c for c in ["주소", "위도", "경도", "event_date", "최초발생원인", "SGG_NM"] if c in work.columns]
    points = work.dropna(subset=[c for c in ["위도", "경도"] if c in work.columns])[point_cols].copy()
    if "event_date" in points.columns:
        points["event_date"] = points["event_date"].astype(str)

    return {
        "total_events": int(len(work)),
        "recent_3m_events": recent_3m,
        "recent_6m_events": recent_6m,
        "sgg_event_counts": sgg_counts.head(15).to_dict(orient="records"),
        "monthly_counts": monthly_counts,
        "event_points": points.head(3000).to_dict(orient="records"),
        "max_date": str(max_date.date()) if pd.notna(max_date) else None,
    }


def build_map_payload(risk_gdf: Optional[gpd.GeoDataFrame], selected_metric: str = "risk_prob"):
    if risk_gdf is None or len(risk_gdf) == 0:
        return {"map_ready": False, "geojson": None, "message": "표시할 grid 데이터가 없습니다."}

    gdf = risk_gdf.copy()
    gdf["display_value"] = pd.to_numeric(gdf.get(selected_metric), errors="coerce") if selected_metric in gdf.columns else np.nan
    gdf["color"] = gdf["risk_level"].apply(lambda x: color_from_level(int(x) if pd.notna(x) else 1))

    keep_cols = [c for c in ["id", "SGG_NM", "DONG", "display_name", "risk_prob", "risk_level", "risk_label", "display_value", "color", selected_metric] if c in gdf.columns]
    geojson = json.loads(gdf[keep_cols + ["geometry"]].to_json())
    return {"map_ready": True, "geojson": geojson, "message": None}


def load_initial_state():
    messages = []
    try:
        state["spatial_template"] = load_spatial_template()
        messages.append("공간 틀 로드 완료")
    except Exception as e:
        state["spatial_template"] = None
        messages.append(f"공간 틀 로드 실패: {e}")

    if DEFAULT_GRID_INPUT_PATH.exists() and state["spatial_template"] is not None:
        try:
            default_df = load_dataframe(DEFAULT_GRID_INPUT_PATH)
            missing = validate_input_columns(default_df)
            if missing:
                messages.append(f"기본 df 컬럼 누락: {', '.join(missing)}")
            else:
                state["current_input_df"] = default_df
                state["risk_gdf"] = build_predictions(default_df, state["spatial_template"])
                messages.append("기본 위험도 계산 완료")
        except Exception as e:
            messages.append(f"기본 위험도 계산 실패: {e}")

    if DEFAULT_EVENT_PATH.exists():
        try:
            state["event_df"] = load_dataframe(DEFAULT_EVENT_PATH)
            messages.append("기본 발생 이력 로드 완료")
        except Exception as e:
            messages.append(f"기본 발생 이력 로드 실패: {e}")

    state["message"] = " / ".join(messages)


load_initial_state()


# ------------------------------
# 라우트
# ------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_name": MODEL_NAME,
            "threshold": THRESHOLD,
            "created_at": MODEL_CREATED_AT,
        },
    )


@app.get("/api/bootstrap")
def bootstrap(selected_metric: str = "risk_prob"):
    return {
        "model_name": MODEL_NAME,
        "threshold": THRESHOLD,
        "features": FEATURES,
        "selected_metric": selected_metric,
        "grid_summary": summarize_grid(state["risk_gdf"]),
        "event_summary": summarize_events(state["event_df"]),
        "map_payload": build_map_payload(state["risk_gdf"], selected_metric),
        "message": state["message"],
    }


@app.post("/api/upload-grid")
async def upload_grid(file: UploadFile = File(...)):
    if state["spatial_template"] is None:
        return JSONResponse({"ok": False, "message": "공간 틀 파일(final_df_4326.geojson 또는 final_df.gpkg)이 먼저 필요합니다."}, status_code=400)

    target = DATA_DIR / file.filename
    target.write_bytes(await file.read())
    try:
        input_df = load_dataframe(target)
        missing = validate_input_columns(input_df)
        if missing:
            return JSONResponse({"ok": False, "message": f"업로드 df에 필요한 컬럼이 없습니다: {', '.join(missing)}"}, status_code=400)

        input_df["id"] = to_str_id(input_df["id"])
        state["current_input_df"] = input_df
        state["risk_gdf"] = build_predictions(input_df, state["spatial_template"])
        matched = int(state["risk_gdf"]["risk_prob"].notna().sum())
        total_template = int(len(state["spatial_template"]))
        state["message"] = f"최신 df 업로드 완료 / id 기준 merge 완료 / 예측 반영 grid: {matched:,}개 / 공간 틀 전체 grid: {total_template:,}개"
        return {"ok": True, "message": state["message"]}
    except Exception as e:
        return JSONResponse({"ok": False, "message": f"grid 업로드 실패: {e}"}, status_code=500)


@app.post("/api/upload-events")
async def upload_events(file: UploadFile = File(...)):
    target = DATA_DIR / file.filename
    target.write_bytes(await file.read())
    try:
        state["event_df"] = load_dataframe(target)
        state["message"] = "발생 이력 파일 업로드 완료"
        return {"ok": True, "message": state["message"]}
    except Exception as e:
        return JSONResponse({"ok": False, "message": f"이벤트 파일 업로드 실패: {e}"}, status_code=500)


@app.get("/api/map-data")
def map_data(metric: str = "risk_prob"):
    return build_map_payload(state["risk_gdf"], metric)
