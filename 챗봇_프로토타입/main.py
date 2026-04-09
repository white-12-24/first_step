import os
import json
import math
import re
import traceback
import uuid
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter
from difflib import SequenceMatcher

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# =========================
# (선택) OpenAI LLM 연결
# =========================
USE_LLM = True  # 키 없으면 자동 fallback
LLM_MODEL = "gpt-4o-mini"

try:
    from openai import OpenAI
    _openai_client = OpenAI()
except Exception:
    _openai_client = None


# =========================================================
# 0) 파일 경로
# =========================================================
SPOTS_XLS   = r"C:\py_temp\2차프로젝트\챗봇_프로토타입\spot.xlsx"
MENU_XLSX   = r"C:\py_temp\2차프로젝트\챗봇_프로토타입\FINAL\store_menu_TRANSLATED_FINAL_20260202_202906.xlsx"
REVIEW_XLSX = r"C:\py_temp\2차프로젝트\챗봇_프로토타입\FINAL\review_preprocessed_FINAL_20260202_202906.xlsx"
STORE_JSON  = r"C:\py_temp\2차프로젝트\챗봇_프로토타입\store_profile_v1.json"
RULES_JSON  = r"C:\py_temp\2차프로젝트\챗봇_프로토타입\menu_pair_rules_v1_TUNED.json"

# ✅ 추가: 가게 정보 엑셀(휴무일/예약 등)
STORE_XLSX  = r"C:\py_temp\2차프로젝트\챗봇_프로토타입\tabelog-store_TRANSLATED_ALL.xlsx"

RADIUS_M_DEFAULT = 5000
LIMIT_DEFAULT = 5
RETURN_CARDS_DEFAULT = False

MAKE_CARD_VERSION = "v4.0.0-hybrid-intent-llm-no-jump-storeinfo"


# =========================================================
# 1) 로드
# =========================================================
spots_df = pd.read_excel(SPOTS_XLS)

try:
    menu_df = pd.read_excel(MENU_XLSX)
except Exception:
    menu_df = pd.DataFrame()

review_df = pd.read_excel(REVIEW_XLSX)

with open(STORE_JSON, "r", encoding="utf-8") as f:
    store_profiles = json.load(f)

with open(RULES_JSON, "r", encoding="utf-8") as f:
    store_rules = json.load(f)

# ✅ 추가: store info 엑셀 로드 (없으면 빈 DF)
try:
    if os.path.exists(STORE_XLSX):
        store_info_df = pd.read_excel(STORE_XLSX)
    else:
        store_info_df = pd.DataFrame()
except Exception:
    store_info_df = pd.DataFrame()

store_by_id: Dict[int, Dict[str, Any]] = {
    int(p["store_id"]): p
    for p in store_profiles
    if p.get("store_id") is not None
}

rules_by_store: Dict[int, List[Dict[str, Any]]] = {}
for x in store_rules:
    sid = x.get("store_id")
    if sid is None:
        continue
    try:
        sid = int(sid)
    except Exception:
        continue
    rules_by_store[sid] = x.get("rules", []) or []


# =========================================================
# 2) 유틸
# =========================================================
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def _norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "", s)
    return s.lower()

def _split_items(x: Any) -> List[str]:
    if x is None:
        return []
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []
    s = s.replace("｜", "|")
    parts = re.split(r"[,\|;/]+", s)
    out = []
    for p in parts:
        t = p.strip()
        if t:
            out.append(t)
    return out

def _safe_mean(series: pd.Series) -> Optional[float]:
    try:
        if series is None or len(series) == 0:
            return None
        v = float(series.mean())
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

# ✅ “없음/결측” 텍스트 통일
MISSING_TOKENS = set([
    "", "nan", "none", "null",
    "입력값이없습니다.", "입력값이 없습니다.", "정보없음", "없음", "-"
])

def is_missing_value(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    if not s:
        return True
    if _norm(s) in { _norm(x) for x in MISSING_TOKENS }:
        return True
    return False


# =========================================================
# ✅ (추가) store_info_df 컬럼 자동 정의(설명/예시)
#     - “학습”이라기보다: 컬럼명+샘플값으로 의미/용도 정의
# =========================================================
def infer_storeinfo_column_defs(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    defs: Dict[str, Dict[str, Any]] = {}
    if df is None or len(df) == 0:
        return defs

    for c in df.columns:
        c_low = c.lower()
        sample = None
        try:
            s = df[c].dropna().astype(str)
            s = s[s.str.strip() != ""]
            if len(s) > 0:
                sample = s.iloc[0][:60]
        except Exception:
            sample = None

        desc = "가게 정보 컬럼"
        if "dayoff" in c_low or "휴무" in c:
            desc = "휴무/정기휴무 정보"
        elif "reservation" in c_low or "예약" in c:
            desc = "예약 가능 여부/예약 방식"
        elif "address" in c_low or "주소" in c:
            desc = "주소"
        elif "station" in c_low or "역" in c:
            desc = "가까운 역/교통 정보"
        elif "rating" in c_low or "평점" in c:
            desc = "평점"
        elif "review_count" in c_low:
            desc = "리뷰 수"
        elif "bookmark" in c_low:
            desc = "북마크/저장 수"
        elif "lunch_avg" in c_low:
            desc = "점심 평균 가격대"
        elif "dinner_avg" in c_low:
            desc = "저녁 평균 가격대"
        elif c_low.endswith("_ko"):
            # 번역본
            desc = f"{desc} (한국어 번역본)"

        defs[c] = {"desc": desc, "example": sample}
    return defs

STOREINFO_COL_DEFS = infer_storeinfo_column_defs(store_info_df)
if STOREINFO_COL_DEFS:
    print("[store_info_df 컬럼 정의]")
    for k, v in STOREINFO_COL_DEFS.items():
        print(f"- {k}: {v.get('desc')} | ex={v.get('example')}")


# =========================================================
# ✅ (추가) store_info_df에서 store_id로 값 조회
# =========================================================
def get_store_info_value(store_id: int, col: str) -> Optional[str]:
    if store_info_df is None or len(store_info_df) == 0:
        return None
    if "store_id" not in store_info_df.columns:
        return None
    if col not in store_info_df.columns:
        return None
    sub = store_info_df[store_info_df["store_id"].astype(str) == str(store_id)].copy()
    if len(sub) == 0:
        return None
    v = sub.iloc[0][col]
    if is_missing_value(v):
        return None
    return str(v).strip()


# =========================================================
# 3) spots 컬럼 찾기
# =========================================================
ANIME_COL_CANDS = ["name(kr)", "name_kr", "anime", "anime_kr"]
ANIME_COL = next((c for c in ANIME_COL_CANDS if c in spots_df.columns), None)
if ANIME_COL is None:
    raise KeyError("spot 파일에서 애니 이름 컬럼(name(kr) 등)을 찾지 못했습니다.")

SPOT_COL_CANDS = ["spot", "spot_name", "spot_kr", "spot_name_kr"]
LAT_COL_CANDS  = ["lat", "latitude"]
LON_COL_CANDS  = ["lon", "longitude"]

SPOT_COL = next((c for c in SPOT_COL_CANDS if c in spots_df.columns), None)
LAT_COL  = next((c for c in LAT_COL_CANDS  if c in spots_df.columns), None)
LON_COL  = next((c for c in LON_COL_CANDS  if c in spots_df.columns), None)

if SPOT_COL is None or LAT_COL is None or LON_COL is None:
    raise KeyError("spot 파일에서 spot/lat/lon 컬럼을 찾지 못했습니다.")

ANIME_LIST = sorted(
    spots_df[ANIME_COL].dropna().astype(str).unique().tolist(),
    key=len,
    reverse=True
)

def extract_anime_simple(text: str) -> Optional[str]:
    t = (text or "").strip()
    for a in ANIME_LIST:
        if a and a in t:
            return a
    return None


# =========================================================
# 4) 스팟 resolve
# =========================================================
STOPWORDS = [
    "성지순례", "성지", "성지스팟", "스팟", "근처", "주변", "가까운", "맛집", "라멘", "추천", "추천해줘",
    "찾아줘", "알려줘", "어디", "있어", "해줘", "좀", "나", "난", "지금", "왔는데", "와있는데", "있는데",
    "부근", "인근", "에서", "으로", "까지", "같이", "좀요", "카페", "식당", "밥", "먹을", "먹고", "싶어",
]

def clean_spot_query(message: str, anime: Optional[str]) -> str:
    t = (message or "").strip()
    if anime:
        t = t.replace(anime, " ")
    for w in STOPWORDS:
        t = t.replace(w, " ")
    t = re.sub(r"[^\w가-힣ㄱ-ㅎㅏ-ㅣ\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def resolve_spot(anime: str, spot_text: str, topk: int = 3) -> Dict[str, Any]:
    sub = spots_df[spots_df[ANIME_COL].astype(str) == str(anime)].copy()
    if len(sub) == 0:
        return {"status": "not_found", "candidates": [], "anime": anime, "spot_query": spot_text}

    q_raw = (spot_text or "").strip()
    q = _norm(q_raw)

    names = sub[SPOT_COL].astype(str).fillna("").tolist()
    scored = []
    for i, nm in enumerate(names):
        nm_raw = nm
        nm_n = _norm(nm_raw)

        if nm_n and nm_n in q:
            sc = 100
        elif q and q in nm_n:
            sc = 95
        else:
            sc = int(SequenceMatcher(None, q, nm_n).ratio() * 100)

        scored.append((sc, i, nm_raw))

    scored.sort(reverse=True, key=lambda x: x[0])
    scored = scored[:topk]

    candidates = []
    for sc, idx_pos, nm_raw in scored:
        row = sub.iloc[idx_pos]
        spot_id = str(row.name)
        candidates.append({
            "spot_id": spot_id,
            "spot_name": row[SPOT_COL],
            "score": int(sc),
            "lat": float(row[LAT_COL]),
            "lon": float(row[LON_COL]),
        })

    if candidates:
        top1 = candidates[0]["score"]
        top2 = candidates[1]["score"] if len(candidates) > 1 else 0
        if top1 >= 90 or (top1 >= 70 and (top1 - top2) >= 10):
            return {"status": "resolved", "candidates": candidates, "anime": anime, "spot_query": q_raw}

    return {"status": "ambiguous", "candidates": candidates, "anime": anime, "spot_query": q_raw}

def guess_anime_by_best_match(spot_text: str, topk: int = 3) -> Dict[str, Any]:
    best = None
    for a in ANIME_LIST:
        r = resolve_spot(a, spot_text, topk=topk)
        cands = r.get("candidates") or []
        if not cands:
            continue
        c0 = cands[0]
        key = (int(c0["score"]), a)
        if (best is None) or (key > best["key"]):
            best = {"key": key, "anime": a, "resolve": r}
    if best is None:
        return {"status": "not_found"}
    return {"status": "ok", "anime": best["anime"], "resolve": best["resolve"]}


# =========================================================
# 5) 데이터 기반(리뷰/메뉴) 함수들
# =========================================================
ALIAS_MAP = {
    "반공기밥": "밥",
    "반 공기 밥": "밥",
    "반공기": "밥",
    "밥소": "밥",
    "밥중": "밥",
    "밥대": "밥",
    "밥대사이즈": "밥",
    "밥세트": "밥",
    "고기밥": "고기 덮밥",
    "TKG": "계란밥",
    "TKG(계란밥)": "계란밥",
    "계란덮밥": "계란밥",
    "반숙조란": "맛달걀",
    "반숙달걀": "맛달걀",
    "맛달걀토핑": "맛달걀",
    "면추가": "면 추가",
}

def normalize_item(s: str) -> str:
    if not s:
        return ""
    raw = str(s).strip()
    key = raw.replace(" ", "")
    if key in ALIAS_MAP:
        return ALIAS_MAP[key]
    if "밥" in raw:
        return "밥"
    if "달걀" in raw or "계란" in raw or "조란" in raw:
        return "맛달걀"
    return raw

def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def topn_from_series(series: pd.Series, top_n: int) -> List[str]:
    cnt = Counter()
    if series is None or len(series) == 0:
        return []
    for v in series.dropna():
        for x in _split_items(v):
            nx = normalize_item(x)
            if nx:
                cnt[nx] += 1
    return [k for k, _ in cnt.most_common(top_n)]

def get_menu_top1_from_reviews(store_id: int) -> Optional[str]:
    if review_df is None or len(review_df) == 0:
        return None
    if "store_id" not in review_df.columns or "menu_name_main" not in review_df.columns:
        return None
    sub = review_df[review_df["store_id"] == store_id].copy()
    if len(sub) == 0:
        return None
    s = sub["menu_name_main"].astype(str).fillna("").str.strip()
    s = s[s != ""]
    s = s[~s.str.lower().isin(["nan", "none", "null"])]
    if len(s) == 0:
        return None
    top1 = s.value_counts().head(1).index.tolist()
    return str(top1[0]).strip() if top1 else None

def get_menu_topn_from_reviews(store_id: int, topn: int = 5) -> List[Tuple[str, int]]:
    if review_df is None or len(review_df) == 0:
        return []
    if "store_id" not in review_df.columns or "menu_name_main" not in review_df.columns:
        return []
    sub = review_df[review_df["store_id"] == store_id].copy()
    if len(sub) == 0:
        return []
    s = sub["menu_name_main"].astype(str).fillna("").str.strip()
    s = s[s != ""]
    s = s[~s.str.lower().isin(["nan", "none", "null"])]
    if len(s) == 0:
        return []
    vc = s.value_counts().head(topn)
    return [(idx, int(cnt)) for idx, cnt in vc.items()]

def _detect_menu_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    if df is None or len(df) == 0:
        return {"store_id": None, "menu_name": None, "price_yen": None, "category": None}

    store_cands = ["store_id", "storeId", "sid"]
    name_cands  = ["menu_name_ko", "menu_name", "menu", "name_ko", "name"]
    price_cands = ["menu_price_yen", "price_yen", "price", "price_jpy", "yen"]
    cat_cands   = ["menu_category", "category", "type", "menu_type"]

    store_col = next((c for c in store_cands if c in df.columns), None)
    name_col  = next((c for c in name_cands  if c in df.columns), None)
    price_col = next((c for c in price_cands if c in df.columns), None)
    cat_col   = next((c for c in cat_cands if c in df.columns), None)

    return {"store_id": store_col, "menu_name": name_col, "price_yen": price_col, "category": cat_col}

MENU_COLS = _detect_menu_cols(menu_df)

def get_price_for_menu(store_id: int, menu_name: str) -> Optional[float]:
    if menu_df is None or len(menu_df) == 0:
        return None
    store_col = MENU_COLS.get("store_id")
    name_col  = MENU_COLS.get("menu_name")
    price_col = MENU_COLS.get("price_yen")
    if not store_col or not name_col or not price_col:
        return None

    try:
        sub = menu_df[menu_df[store_col].astype(str) == str(store_id)].copy()
    except Exception:
        return None
    if len(sub) == 0:
        return None

    sub[name_col] = sub[name_col].astype(str).fillna("")
    sub[price_col] = pd.to_numeric(sub[price_col], errors="coerce")
    sub = sub.dropna(subset=[price_col])

    q = _norm(menu_name)
    if not q:
        return None

    m = sub[sub[name_col].apply(lambda x: (_norm(x) == q or q in _norm(x) or _norm(x) in q))].copy()
    if len(m) == 0:
        return None
    return float(m[price_col].median())

def get_store_menu_rows(store_id: int) -> pd.DataFrame:
    if menu_df is None or len(menu_df) == 0:
        return pd.DataFrame()
    store_col = MENU_COLS.get("store_id")
    if not store_col:
        return pd.DataFrame()
    try:
        sub = menu_df[menu_df[store_col].astype(str) == str(store_id)].copy()
    except Exception:
        return pd.DataFrame()
    return sub


# =========================================================
# 6) 추천 로직
# =========================================================
def recommend_by_spot_index(spot_index: int, radius_m: int, limit: int) -> Dict[str, Any]:
    if spot_index not in spots_df.index:
        return {"status": "error", "message": "invalid spot_id(index)"}

    spot = spots_df.loc[spot_index]
    spot_lat = float(spot[LAT_COL])
    spot_lon = float(spot[LON_COL])
    spot_name = str(spot[SPOT_COL])

    candidates = []
    for sid, p in store_by_id.items():
        loc = p.get("location", {})
        lat = loc.get("lat") if isinstance(loc, dict) else None
        lon = loc.get("lon") if isinstance(loc, dict) else None
        if lat is None or lon is None:
            continue

        d = haversine_m(spot_lat, spot_lon, lat, lon)
        if d <= radius_m:
            candidates.append((sid, d))

    candidates.sort(key=lambda x: x[1])
    candidates = candidates[:limit]

    stores = []
    for sid, d in candidates:
        stores.append({
            "store_id": int(sid),
            "store_name": store_by_id[sid].get("store_name_ko", ""),
            "distance_m": float(d),
        })

    return {
        "status": "ok",
        "spot": {"spot_id": str(spot_index), "spot_name": spot_name, "lat": spot_lat, "lon": spot_lon},
        "radius_m": int(radius_m),
        "limit": int(limit),
        "stores": stores,
    }


# =========================================================
# 7) 세션(대화 상태)
# =========================================================
SESSIONS: Dict[str, Dict[str, Any]] = {}

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "history": [],
            "last_spot": None,
            "last_reco": [],
            "last_selected_store": None,
        }
    return SESSIONS[session_id]

def push_history(sess: Dict[str, Any], role: str, text: str):
    sess["history"].append({"role": role, "text": text})
    if len(sess["history"]) > 20:
        sess["history"] = sess["history"][-20:]


# =========================================================
# 8) Intent 분류 (규칙 우선 → unknown일 때만 LLM)
#    ✅ recommend 점프는 “명시 요청”일 때만
# =========================================================
INTENTS = [
    "recommend",
    "followup_spiciest",
    "followup_top_rated",
    "store_side_list",
    "store_topping_list",
    "store_price",
    "store_best_menu",
    "store_popular_menus",
    # ✅ 추가: store info(엑셀 컬럼 기반)
    "store_dayoff",
    "store_reservation",
    "reset",
    "unknown"
]

FOLLOWUP_HINT_WORDS = [
    "이 중", "그중", "방금", "추천", "목록", "1번", "2번", "3번", "그 가게", "이 가게", "거기",
    "토핑", "사이드", "가격", "얼마", "비싼", "대표", "잘나가", "인기", "메뉴",
    "가장 매운", "제일 매운", "맵", "매운", "평점", "별점", "1등", "저중", "저중에",
    # ✅ 추가
    "휴무", "휴무일", "정기휴무", "예약", "예약제"
]

SPOT_REQUEST_WORDS = ["근처", "주변", "가까운", "인근", "부근", "라멘", "추천", "맛집", "알려줘", "찾아줘"]

def has_followup_signal(msg: str) -> bool:
    m = msg or ""
    return any(s in m for s in FOLLOWUP_HINT_WORDS)

def looks_like_spot_request(msg: str) -> bool:
    m = (msg or "").strip()
    return any(w in m for w in SPOT_REQUEST_WORDS)

def rule_intent_guess(message: str, sess: Dict[str, Any]) -> Dict[str, Any]:
    m = (message or "").strip()

    if any(x in m for x in ["리셋", "초기화", "대화 리셋"]):
        return {"intent": "reset"}

    # ✅ last_reco가 있으면: 기본은 follow-up
    if sess.get("last_reco"):
        if any(x in m for x in ["가장 매운", "제일 매운", "맵", "매운"]):
            return {"intent": "followup_spiciest"}
        if any(x in m for x in ["평점", "별점", "제일 높은", "가장 높은", "1등"]):
            return {"intent": "followup_top_rated"}
        if any(x in m for x in ["사이드", "사이드메뉴", "사이드 메뉴"]):
            return {"intent": "store_side_list"}
        if any(x in m for x in ["토핑", "토핑메뉴", "토핑 메뉴"]):
            return {"intent": "store_topping_list"}
        if any(x in m for x in ["가격", "얼마", "엔", "원", "비싼", "비싸"]):
            return {"intent": "store_price"}
        if any(x in m for x in ["휴무", "휴무일", "정기휴무", "쉬는날", "쉬는 날"]):
            return {"intent": "store_dayoff"}
        if any(x in m for x in ["예약", "예약제", "예약 가능", "예약불가", "예약 돼"]):
            return {"intent": "store_reservation"}

        if any(x in m for x in ["대표", "시그니처", "베스트", "잘나가", "인기", "추천메뉴", "추천 메뉴"]):
            if any(x in m for x in ["대표", "시그니처", "베스트"]):
                return {"intent": "store_best_menu"}
            return {"intent": "store_popular_menus"}

        # ✅ “새 추천/다시/다른 스팟/리롤” 명시한 경우만 recommend 허용
        if looks_like_spot_request(m) and any(x in m for x in ["새로", "다시", "다른 스팟", "다른 장소", "리롤"]):
            return {"intent": "recommend"}

        return {"intent": "unknown"}

    # last_reco 없으면: recommend
    return {"intent": "recommend"}

def llm_classify_intent(message: str, sess: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    ✅ 룰에서 unknown일 때만 호출할 것.
    ✅ 결과는 intent 분류(JSON)만. (절대 답 생성 X)
    """
    if not USE_LLM:
        return None
    if _openai_client is None:
        return None
    if not os.getenv("OPENAI_API_KEY"):
        return None

    reco_names = [x.get("store_name","") for x in (sess.get("last_reco") or [])][:10]
    spot_name = (sess.get("last_spot") or {}).get("spot_name")

    schema = {
        "name": "ramen_intent",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "intent": {"type": "string", "enum": INTENTS},
                "spot_query": {"type": ["string", "null"]},
                "radius_m": {"type": ["integer", "null"]},
                "limit": {"type": ["integer", "null"]},
                "target_store": {"type": ["string", "null"]},
            },
            "required": ["intent", "spot_query", "radius_m", "limit", "target_store"]
        }
    }

    sys = (
        "너는 라멘 추천 챗봇의 '의도 분류기'다.\n"
        "절대 규칙:\n"
        "1) 답변을 만들지 말고 intent/필드만 JSON으로 출력.\n"
        "2) last_reco_names가 있으면 대부분 후속질문이다.\n"
        "3) 후속질문에서 recommend로 보내지 마라.\n"
        "   단, 사용자가 '새로 추천/다시 추천/다른 스팟/리롤'을 명확히 말한 경우만 recommend 허용.\n"
        "4) 휴무/예약 질문은 store_dayoff/store_reservation으로 분류해라.\n"
        f"\n[컨텍스트]\n- last_spot: {spot_name}\n- last_reco_names: {reco_names}\n"
    )

    try:
        resp = _openai_client.responses.create(
            model=LLM_MODEL,
            input=[
                {"role":"system","content":sys},
                {"role":"user","content":message},
            ],
            text={"format":{"type":"json_schema","json_schema":schema}},
            temperature=0.1,
        )
        return json.loads(resp.output_text)
    except Exception:
        return None


# =========================================================
# 9) last_reco에서 가게 선택 해결
# =========================================================
def resolve_target_store_from_last_reco(target_store: str, sess: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    last = sess.get("last_reco") or []
    if not last:
        return None, {"why": "no_last_reco"}

    t = (target_store or "").strip()
    if not t:
        return None, {"why": "empty_target_store"}

    m = re.search(r"(\d+)\s*번", t)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(last):
            return last[idx], {"why": "picked_by_index", "index": idx+1}
        return None, {"why": "index_out_of_range", "index": idx+1, "len": len(last)}

    best = None
    for item in last:
        nm = item.get("store_name","")
        sc = similar(t, nm)
        if (best is None) or (sc > best["score"]):
            best = {"score": sc, "item": item, "name": nm}

    if best and best["score"] >= 0.62:
        return best["item"], {"why": "picked_by_fuzzy", "score": best["score"], "matched": best["name"]}

    return None, {"why": "fuzzy_too_low", "best_score": (best["score"] if best else None)}

def resolve_store_from_message_using_last_reco(message: str, sess: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    last = sess.get("last_reco") or []
    if not last:
        return None, {"why": "no_last_reco"}

    m = (message or "").strip()

    idx_hit = re.search(r"(\d+)\s*번", m)
    if idx_hit:
        return resolve_target_store_from_last_reco(idx_hit.group(0), sess)

    best = None
    for item in last:
        nm = item.get("store_name","")
        if nm and _norm(nm) in _norm(m):
            return item, {"why": "name_in_message", "matched": nm}
        sc = similar(m, nm)
        if (best is None) or (sc > best["score"]):
            best = {"score": sc, "item": item, "name": nm}

    if best and best["score"] >= 0.60:
        return best["item"], {"why": "picked_by_fuzzy_message", "score": best["score"], "matched": best["name"]}

    return None, {"why": "no_store_match"}


# =========================================================
# 10) 후속에 필요한 데이터 계산
# =========================================================
def get_store_review_means(store_id: int) -> Dict[str, Optional[float]]:
    if review_df is None or len(review_df) == 0:
        return {"salt": None, "rich": None, "spicy": None}
    if "store_id" not in review_df.columns:
        return {"salt": None, "rich": None, "spicy": None}
    sub = review_df[review_df["store_id"] == store_id].copy()
    if len(sub) == 0:
        return {"salt": None, "rich": None, "spicy": None}

    salt = _safe_mean(sub["saltiness_1to5"]) if "saltiness_1to5" in sub.columns else None
    rich = _safe_mean(sub["richness_1to5"]) if "richness_1to5" in sub.columns else None
    spicy = _safe_mean(sub["spiciness_1to5"]) if "spiciness_1to5" in sub.columns else None
    return {"salt": salt, "rich": rich, "spicy": spicy}

def followup_pick_spiciest(sess: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    last = sess.get("last_reco") or []
    if not last:
        return None, []
    scored = []
    for it in last:
        sid = int(it["store_id"])
        means = get_store_review_means(sid)
        if means.get("spicy") is not None:
            scored.append((means.get("spicy"), it, means))
    scored.sort(reverse=True, key=lambda x: x[0])
    if not scored:
        return None, []
    ranking = [{"store": x[1], "spicy": x[0]} for x in scored]
    return scored[0][1], ranking

def followup_pick_top_rated(sess: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    last = sess.get("last_reco") or []
    if not last:
        return None, []
    scored = []
    for it in last:
        sid = int(it["store_id"])
        sub = review_df[review_df["store_id"] == sid].copy() if "store_id" in review_df.columns else pd.DataFrame()
        rating = None
        for cand in ["rating_1to5", "star_1to5", "score_1to5", "overall_1to5"]:
            if cand in sub.columns:
                rating = _safe_mean(sub[cand])
                break
        if rating is None:
            means = get_store_review_means(sid)
            vals = [v for v in [means.get("salt"), means.get("rich"), means.get("spicy")] if v is not None]
            rating = (sum(vals)/len(vals)) if vals else None
        if rating is not None:
            scored.append((rating, it))
    scored.sort(reverse=True, key=lambda x: x[0])
    if not scored:
        return None, []
    ranking = [{"store": x[1], "rating": x[0]} for x in scored]
    return scored[0][1], ranking

def store_side_list(store_id: int) -> List[str]:
    if "store_id" not in review_df.columns or "side_names" not in review_df.columns:
        return []
    sub = review_df[review_df["store_id"] == store_id].copy()
    if len(sub) == 0:
        return []
    return dedup_keep_order(topn_from_series(sub["side_names"], 30))

def store_topping_list(store_id: int) -> List[str]:
    if "store_id" not in review_df.columns or "topping_names" not in review_df.columns:
        return []
    sub = review_df[review_df["store_id"] == store_id].copy()
    if len(sub) == 0:
        return []
    return dedup_keep_order(topn_from_series(sub["topping_names"], 30))


# =========================================================
# 11) 말투 생성(LLM) - ✅ 환각 방지 강화
# =========================================================
def llm_say(message: str, sess: Dict[str, Any], data_payload: Dict[str, Any]) -> Optional[str]:
    if not USE_LLM:
        return None
    if _openai_client is None:
        return None
    if not os.getenv("OPENAI_API_KEY"):
        return None

    hist = sess.get("history") or []
    hist_txt = "\n".join([f"{h['role']}: {h['text']}" for h in hist[-8:]])

    sys = (
        "너는 라멘 추천 챗봇의 '말투 생성기'다.\n"
        "절대 규칙:\n"
        "1) data_payload 안의 사실만 사용.\n"
        "2) data_payload에 없는 가게/메뉴/가격/거리/수치를 절대 만들어내지 마.\n"
        "3) 모르면 '데이터가 없어'라고 말해.\n"
        "4) 사용자는 한국어.\n"
        "5) '다른 가게/다른 스팟 추천'으로 튀지 마. 사용자가 명시적으로 요청한 경우에만.\n"
    )

    user = (
        f"[최근 대화]\n{hist_txt}\n\n"
        f"[사용자 최신 질문]\n{message}\n\n"
        f"[data_payload(JSON)]\n{json.dumps(data_payload, ensure_ascii=False)}"
    )

    try:
        resp = _openai_client.responses.create(
            model=LLM_MODEL,
            input=[
                {"role":"system","content":sys},
                {"role":"user","content":user},
            ],
            temperature=0.2,
        )
        return (resp.output_text or "").strip()
    except Exception:
        return None


# =========================================================
# 12) reply 템플릿
# =========================================================
def build_reply_recommend_list(spot_name: str, radius_m: int, stores: List[Dict[str, Any]], score: int) -> str:
    lines = []
    lines.append(f"{spot_name} 근처 라멘 맛집은 다음과 같아! (반경 {radius_m}m / 매칭 {score})")
    lines.append("")
    for i, s in enumerate(stores, start=1):
        lines.append(f"{i}. **{s['store_name']}** - 약 {int(s['distance_m'])}m")
    lines.append("")
    lines.append("원하면 이어서 이렇게 물어봐도 돼:")
    lines.append("- “이 중 제일 매운 곳?”")
    lines.append("- “2번 가게 토핑/사이드/가격/대표메뉴/인기메뉴/휴무일/예약 알려줘”")
    return "\n".join(lines)

def build_reply_need_store() -> str:
    return (
        "어느 가게를 말하는지 확인이 필요해 😅\n"
        "예) **'2번 가게'**, **가게 이름**, 또는 **'그 가게'**(바로 직전에 언급한 가게)"
    )

def build_reply_unknown_keep_context() -> str:
    return (
        "그 질문은 지금 내가 가진 데이터로는 확인이 어려워 😭\n"
        "가능한 질문 예시:\n"
        "- 가격 / 토핑 / 사이드\n"
        "- 대표메뉴 / 인기메뉴\n"
        "- 휴무일 / 예약(가게정보 엑셀에 있으면)\n"
        "\n원하는 가게를 **2번 가게**처럼 지정해서 다시 물어봐줘!"
    )

def user_explicit_new_reco(message: str) -> bool:
    m = (message or "").strip()
    return any(x in m for x in ["새로", "다시", "다른 스팟", "다른 장소", "리롤", "새 추천"])


# =========================================================
# 13) FastAPI
# =========================================================
app = FastAPI()

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass

@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "<h3>static/index.html이 없습니다.</h3>"

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    radius_m: int = RADIUS_M_DEFAULT
    limit: int = LIMIT_DEFAULT
    topk: int = 3
    return_cards: bool = RETURN_CARDS_DEFAULT

@app.post("/reset")
def reset(session_id: str):
    if session_id in SESSIONS:
        del SESSIONS[session_id]
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        session_id = (req.session_id or "").strip() or str(uuid.uuid4())
        sess = get_session(session_id)

        message = (req.message or "").strip()
        if not message:
            return {"session_id": session_id, "reply": "메시지가 비어있어. 예: '롯폰기 힐즈 근처 라멘 추천해줘'", "cards": [], "intent": "unknown"}

        push_history(sess, "user", message)

        # =========================================================
        # ✅ 1) Intent 분류: 룰 우선 → unknown일 때만 LLM intent JSON
        # =========================================================
        rule_guess = rule_intent_guess(message, sess)
        llm_intent = {"intent": rule_guess.get("intent", "unknown"), "spot_query": None, "radius_m": None, "limit": None, "target_store": None}
        intent = llm_intent["intent"]

        # 룰이 unknown이면 LLM intent로만 보강
        if intent == "unknown":
            cand = llm_classify_intent(message, sess)
            if cand:
                llm_intent = cand
                intent = (llm_intent.get("intent") or "unknown")

        # =========================================================
        # ✅ 절대 규칙: last_reco가 있으면 recommend 금지 (명시 요청만 허용)
        # =========================================================
        if sess.get("last_reco") and intent == "recommend" and not user_explicit_new_reco(message):
            # 무조건 차단 → unknown으로 후속 처리 흐름으로 보냄
            intent = "unknown"
            llm_intent["intent"] = "unknown"

        # 2) reset
        if intent == "reset":
            SESSIONS[session_id] = {"history": [], "last_spot": None, "last_reco": [], "last_selected_store": None}
            reply = "오케이! 대화 리셋했어. 다시 스팟/장소로 추천 받아볼래?"
            push_history(SESSIONS[session_id], "assistant", reply)
            return {"session_id": session_id, "reply": reply, "cards": [], "intent": "reset"}

        # =========================================================
        # 3) follow-up 처리 (절대 추천 점프 X)
        # =========================================================
        if sess.get("last_reco") and intent in [
            "followup_spiciest", "followup_top_rated",
            "store_side_list", "store_topping_list",
            "store_price", "store_best_menu", "store_popular_menus",
            "store_dayoff", "store_reservation",
            "unknown"
        ]:
            # unknown이면 LLM intent를 이미 한 번 시도했으니,
            # 여기서는 “정해진 기능”으로만 처리하고, 못하면 “없음/확인불가” + 문맥 유지
            if intent == "unknown":
                reply = build_reply_unknown_keep_context()
                push_history(sess, "assistant", reply)
                return {"session_id": session_id, "reply": reply, "cards": [], "intent": "unknown"}

            # ---- followup: spiciest (가게 지정 필요 없음)
            if intent == "followup_spiciest":
                best, ranking = followup_pick_spiciest(sess)
                if best is None:
                    reply = "추천 목록에서 '맵기' 데이터가 충분한 가게를 찾지 못했어. 다른 기준(평점/토핑/가격)으로 볼까?"
                    push_history(sess, "assistant", reply)
                    return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

                sid = int(best["store_id"])
                means = get_store_review_means(sid)
                data_payload = {
                    "intent": "followup_spiciest",
                    "picked": {"store_name": best["store_name"], "spicy_avg": means.get("spicy")},
                    "ranking_top5": [{"store_name": x["store"]["store_name"], "spicy_avg": x["spicy"]} for x in ranking[:5]],
                }
                reply = llm_say(message, sess, data_payload) or (
                    f"추천 목록 중에서는 **{best['store_name']}** 이(가) 제일 매운 편이야. "
                    f"(맵기 평균 {means.get('spicy'):.2f}/5)"
                )
                sess["last_selected_store"] = {"store_id": int(best["store_id"]), "store_name": best["store_name"]}
                push_history(sess, "assistant", reply)
                return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

            # ---- followup: top rated
            if intent == "followup_top_rated":
                best, ranking = followup_pick_top_rated(sess)
                if best is None:
                    reply = "추천 목록에서 '평점'을 계산할 데이터가 부족해. 다른 기준(맵기/토핑/가격)으로 볼까?"
                    push_history(sess, "assistant", reply)
                    return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

                data_payload = {
                    "intent": "followup_top_rated",
                    "picked": {"store_name": best["store_name"]},
                    "ranking_top5": [{"store_name": x["store"]["store_name"], "rating": x["rating"]} for x in ranking[:5]],
                }
                reply = llm_say(message, sess, data_payload) or f"추천 목록 중 평점 기준으로는 **{best['store_name']}** 쪽이 가장 좋아 보여!"
                sess["last_selected_store"] = {"store_id": int(best["store_id"]), "store_name": best["store_name"]}
                push_history(sess, "assistant", reply)
                return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

            # ---- 아래는 가게 지정 필요
            need_store = intent in [
                "store_side_list", "store_topping_list", "store_price",
                "store_best_menu", "store_popular_menus",
                "store_dayoff", "store_reservation"
            ]

            picked_store = None

            if need_store:
                if any(x in message for x in ["그 가게", "이 가게", "거기", "방금", "아까"]) and sess.get("last_selected_store"):
                    picked_store = sess["last_selected_store"]

                if picked_store is None:
                    ts = (llm_intent.get("target_store") or "").strip()
                    if ts:
                        picked_store, _ = resolve_target_store_from_last_reco(ts, sess)

                if picked_store is None:
                    picked_store, _ = resolve_store_from_message_using_last_reco(message, sess)

                if picked_store is None:
                    reply = build_reply_need_store()
                    push_history(sess, "assistant", reply)
                    return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

                sess["last_selected_store"] = {"store_id": int(picked_store["store_id"]), "store_name": picked_store.get("store_name") or ""}

            sid = int(sess["last_selected_store"]["store_id"])
            store_name = sess["last_selected_store"]["store_name"]

            # ---- store_dayoff (엑셀: dayoff_raw_ko)
            if intent == "store_dayoff":
                dayoff = get_store_info_value(sid, "dayoff_raw_ko")
                if dayoff is None:
                    reply = f"**{store_name}** 휴무일 정보는 현재 데이터에 없어(확인 불가) 😭"
                    push_history(sess, "assistant", reply)
                    return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

                data_payload = {"intent":"store_dayoff", "store": {"store_name": store_name}, "dayoff_raw_ko": dayoff}
                reply = llm_say(message, sess, data_payload) or f"**{store_name}** 휴무일(데이터 기반): **{dayoff}**"
                push_history(sess, "assistant", reply)
                return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

            # ---- store_reservation (엑셀: reservation_ko)
            if intent == "store_reservation":
                rv = get_store_info_value(sid, "reservation_ko")
                if rv is None:
                    reply = f"**{store_name}** 예약 정보는 현재 데이터에 없어(확인 불가) 😭"
                    push_history(sess, "assistant", reply)
                    return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

                data_payload = {"intent":"store_reservation", "store": {"store_name": store_name}, "reservation_ko": rv}
                reply = llm_say(message, sess, data_payload) or f"**{store_name}** 예약(데이터 기반): **{rv}**"
                push_history(sess, "assistant", reply)
                return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

            # ---- store_side_list
            if intent == "store_side_list":
                sides = store_side_list(sid)
                data_payload = {"intent":"store_side_list", "store": {"store_name": store_name}, "side_list": sides[:30]}
                reply = llm_say(message, sess, data_payload) or (
                    f"**{store_name}** 사이드 메뉴(데이터 기반)야:\n- " +
                    ("\n- ".join(sides[:20]) if sides else "(데이터가 없어)")
                )
                push_history(sess, "assistant", reply)
                return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

            # ---- store_topping_list
            if intent == "store_topping_list":
                tops = store_topping_list(sid)
                data_payload = {"intent":"store_topping_list", "store": {"store_name": store_name}, "topping_list": tops[:30]}
                reply = llm_say(message, sess, data_payload) or (
                    f"**{store_name}** 토핑(데이터 기반) 목록이야:\n- " +
                    ("\n- ".join(tops[:20]) if tops else "(데이터가 없어)")
                )
                push_history(sess, "assistant", reply)
                return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

            # ---- store_best_menu
            if intent == "store_best_menu":
                best_menu = get_menu_top1_from_reviews(sid)
                topn = get_menu_topn_from_reviews(sid, topn=5)

                if not best_menu and topn:
                    best_menu = topn[0][0]

                if not best_menu and not topn:
                    reply = f"**{store_name}** 는 리뷰에서 대표 메뉴를 판단할 데이터가 부족해. (menu_name_main이 비어있음)"
                    push_history(sess, "assistant", reply)
                    return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

                price = get_price_for_menu(sid, best_menu) if best_menu else None

                data_payload = {
                    "intent":"store_best_menu",
                    "store": {"store_name": store_name},
                    "best_menu": best_menu,
                    "best_menu_price_yen": price,
                    "review_top5": [{"menu": m, "count": c} for m, c in topn],
                    "note": "가격은 메뉴파일에 있을 때만 제공. 없으면 '없음' 처리.",
                }

                reply = llm_say(message, sess, data_payload)
                if not reply:
                    reply_lines = [f"**{store_name}** 대표 메뉴(리뷰 기반)는 **{best_menu}** 쪽이야."]
                    if price is not None:
                        reply_lines.append(f"- 메뉴파일 기준 가격: 약 **{int(price)}엔**")
                    else:
                        reply_lines.append("- 가격 정보: (메뉴파일에 없음)")
                    if topn:
                        reply_lines.append("")
                        reply_lines.append("리뷰 언급 TOP5:")
                        for m, c in topn:
                            reply_lines.append(f"- {m} ({c})")
                    reply = "\n".join(reply_lines)

                push_history(sess, "assistant", reply)
                return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

            # ---- store_popular_menus
            if intent == "store_popular_menus":
                topn = get_menu_topn_from_reviews(sid, topn=5)
                if not topn:
                    reply = f"**{store_name}** 는 리뷰에서 인기 메뉴를 판단할 데이터가 부족해. (menu_name_main이 비어있음)"
                    push_history(sess, "assistant", reply)
                    return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

                enriched = []
                for m, c in topn:
                    p = get_price_for_menu(sid, m)
                    enriched.append({"menu": m, "count": c, "price_yen": p})

                data_payload = {
                    "intent":"store_popular_menus",
                    "store": {"store_name": store_name},
                    "popular_menus_top5": enriched,
                    "note": "가격은 메뉴파일에 있을 때만 제공. 없으면 None.",
                }

                reply = llm_say(message, sess, data_payload)
                if not reply:
                    lines = [f"**{store_name}** 인기 메뉴 TOP5(리뷰 언급 기반)야:"]
                    for x in enriched:
                        if x["price_yen"] is not None:
                            lines.append(f"- {x['menu']} ({x['count']}) / 약 {int(x['price_yen'])}엔")
                        else:
                            lines.append(f"- {x['menu']} ({x['count']}) / 가격 정보 없음")
                    reply = "\n".join(lines)

                push_history(sess, "assistant", reply)
                return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

            # ---- store_price
            if intent == "store_price":
                is_most_expensive = any(x in message for x in ["제일 비싼", "가장 비싼", "비싼 메뉴"])

                menu_rows = get_store_menu_rows(sid)
                name_col = MENU_COLS.get("menu_name")
                price_col = MENU_COLS.get("price_yen")

                if is_most_expensive:
                    if len(menu_rows) > 0 and name_col and price_col and (name_col in menu_rows.columns) and (price_col in menu_rows.columns):
                        tmp = menu_rows.copy()
                        tmp[price_col] = pd.to_numeric(tmp[price_col], errors="coerce")
                        tmp = tmp.dropna(subset=[price_col])
                        tmp[name_col] = tmp[name_col].astype(str).fillna("").str.strip()
                        tmp = tmp[tmp[name_col] != ""]
                        if len(tmp) > 0:
                            row = tmp.sort_values(price_col, ascending=False).iloc[0]
                            menu_name = str(row[name_col]).strip()
                            price = float(row[price_col])
                            data_payload = {"intent":"store_price_most_expensive", "store": {"store_name": store_name}, "menu": menu_name, "price_yen": price}
                            reply = llm_say(message, sess, data_payload) or f"**{store_name}** 에서 메뉴파일 기준 제일 비싼 메뉴는 **{menu_name}** (약 **{int(price)}엔**) 이야."
                            push_history(sess, "assistant", reply)
                            return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

                    reply = (
                        f"**{store_name}** 는 메뉴파일에 **가격 데이터가 없어서** ‘제일 비싼 메뉴’를 계산할 수 없어.\n"
                        f"대신 리뷰 기반으로 **인기 메뉴 TOP**은 알려줄까? (예: “그 가게 인기메뉴 알려줘”)"
                    )
                    push_history(sess, "assistant", reply)
                    return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

                best_menu = get_menu_top1_from_reviews(sid)
                if not best_menu:
                    top1 = get_menu_topn_from_reviews(sid, topn=1)
                    best_menu = top1[0][0] if top1 else None

                if not best_menu:
                    reply = f"**{store_name}** 는 리뷰에서 대표 메뉴를 잡을 데이터가 부족해서 가격도 매칭하기 어려워. (menu_name_main 없음)"
                    push_history(sess, "assistant", reply)
                    return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

                price = get_price_for_menu(sid, best_menu)
                data_payload = {"intent":"store_price", "store": {"store_name": store_name}, "best_menu": best_menu, "price_yen": price}

                reply = llm_say(message, sess, data_payload)
                if not reply:
                    if price is not None:
                        reply = f"**{store_name}** 대표 메뉴(리뷰 기반) **{best_menu}** 가격은 메뉴파일 기준으로 약 **{int(price)}엔** 정도야."
                    else:
                        reply = (
                            f"**{store_name}** 는 메뉴파일에 **가격 정보가 없어** 정확한 가격은 말하기 어려워.\n"
                            f"대신 리뷰에서 가장 많이 언급된 메뉴는 **{best_menu}** 야."
                        )

                push_history(sess, "assistant", reply)
                return {"session_id": session_id, "reply": reply, "cards": [], "intent": intent}

        # =========================================================
        # 4) 새 추천 처리 (recommend)
        #    ✅ last_reco가 없거나, 사용자 명시 요청일 때만 도달
        # =========================================================
        radius_m = req.radius_m
        limit = req.limit

        if isinstance(llm_intent.get("radius_m"), int) and llm_intent["radius_m"] and llm_intent["radius_m"] > 0:
            radius_m = int(llm_intent["radius_m"])
        if isinstance(llm_intent.get("limit"), int) and llm_intent["limit"] and llm_intent["limit"] > 0:
            limit = int(llm_intent["limit"])

        spot_text = (llm_intent.get("spot_query") or "").strip()
        if not spot_text:
            anime_simple = extract_anime_simple(message)
            spot_text = clean_spot_query(message, anime_simple)
            if not spot_text:
                spot_text = message

        anime = extract_anime_simple(message)
        if not anime:
            g = guess_anime_by_best_match(spot_text, topk=req.topk)
            if g.get("status") == "ok":
                anime = g["anime"]
                resolved = g["resolve"]
            else:
                reply = "스팟 후보를 찾지 못했어. 장소 이름을 조금만 더 정확히 써줘!"
                push_history(sess, "assistant", reply)
                return {"session_id": session_id, "reply": reply, "cards": [], "intent": "recommend"}
        else:
            resolved = resolve_spot(anime, spot_text, topk=req.topk)

        if not (resolved.get("candidates") or []):
            reply = "스팟 후보를 찾지 못했어. 스팟 이름을 조금만 더 정확히 써줘!"
            push_history(sess, "assistant", reply)
            return {"session_id": session_id, "reply": reply, "cards": [], "intent": "recommend"}

        best = resolved["candidates"][0]
        spot_id = best["spot_id"]
        spot_name = best["spot_name"]
        score = best["score"]

        rec = recommend_by_spot_index(int(spot_id), radius_m, limit)
        stores = rec.get("stores", []) if isinstance(rec, dict) else []

        if not stores:
            reply = f"'{spot_name}' 기준 {radius_m}m 안에서 추천 가게를 찾지 못했어. 반경을 늘려볼까?"
            push_history(sess, "assistant", reply)
            return {"session_id": session_id, "reply": reply, "cards": [], "intent": "recommend"}

        sess["last_spot"] = {
            "spot_id": str(spot_id),
            "spot_name": spot_name,
            "anime": anime,
            "score": score,
            "lat": best.get("lat"),
            "lon": best.get("lon"),
            "radius_m": radius_m,
            "limit": limit,
        }
        sess["last_reco"] = [{"store_id": s["store_id"], "store_name": s["store_name"], "distance_m": s["distance_m"]} for s in stores]
        sess["last_selected_store"] = None

        cards = []
        if bool(req.return_cards):
            for s in stores:
                cards.append({
                    "title": s["store_name"],
                    "body": f"거리: {int(s['distance_m'])}m",
                    "footer": "",
                    "tags": ["recommend"],
                })

        data_payload = {
            "intent": "recommend",
            "spot": {"spot_name": spot_name, "radius_m": radius_m, "limit": limit, "match_score": score},
            "reco_summary": [{"rank": i+1, "store_name": s["store_name"], "distance_m": int(s["distance_m"])} for i, s in enumerate(stores)],
        }

        reply = llm_say(message, sess, data_payload) or build_reply_recommend_list(spot_name, radius_m, stores, score)
        push_history(sess, "assistant", reply)

        return {
            "session_id": session_id,
            "reply": reply,
            "cards": cards,
            "intent": "recommend",
            "spot": {"spot_name": spot_name},
            "debug": {
                "intent_raw": llm_intent,
                "intent_final": "recommend",
                "version": MAKE_CARD_VERSION,
                "menu_cols_detected": MENU_COLS,
                "storeinfo_cols": list(store_info_df.columns) if (store_info_df is not None and len(store_info_df) > 0) else [],
                "session": {"has_last_reco": True, "reco_count": len(sess["last_reco"])},
            }
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return JSONResponse(
            status_code=200,
            content={
                "session_id": (req.session_id or None),
                "reply": "서버 내부 에러가 발생했어. debug.traceback을 확인해줘.",
                "cards": [],
                "intent": "error",
                "debug": {"error": str(e), "traceback": tb},
            },
        )