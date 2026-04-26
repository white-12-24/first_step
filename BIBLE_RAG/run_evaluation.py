# run_evaluation.py
# --------------------------------------------------
# Bible RAG local evaluation runner
# 실행 전:
# 1) FastAPI 서버 실행: uvicorn app:app --port 8000
# 2) 이 파일과 bible_rag_eval_questions.csv를 BIBLE_RAG 폴더에 둔다.
# 3) python run_evaluation.py
# --------------------------------------------------

import csv
import json
import re
import time
import uuid
from pathlib import Path
from urllib import request, error

BASE_URL = "http://127.0.0.1:8000"
QUESTION_CSV = Path("bible_rag_eval_questions_v2_diverse.csv")
OUTPUT_CSV = Path("evaluation_results_V2.csv")
SLEEP_SEC = 0.2

session_id = "eval_" + str(uuid.uuid4())

def post_json(path, payload, timeout=120):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, {"error": body}
    except Exception as e:
        return 0, {"error": str(e)}

def reset_memory():
    post_json("/reset_memory", {"session_id": session_id}, timeout=30)

def split_answer_evidence(text):
    text = str(text)
    markers = ["\n---\n[검색 근거]\n", "\n---\n[이전 검색 근거]\n"]
    for m in markers:
        if m in text:
            left, right = text.split(m, 1)
            return left.strip(), right.strip(), m.strip()
    return text.strip(), "", ""

def contains_any(text, pipe_keywords):
    kws = [x.strip() for x in str(pipe_keywords).split("|") if x.strip()]
    if not kws:
        return ""
    hits = []
    for kw in kws:
        if kw in text:
            hits.append(kw)
    return "|".join(hits)

def rough_score(answer_text, evidence_text, expected_refs, expected_topics):
    combined = (answer_text or "") + "\n" + (evidence_text or "")
    ref_hits = contains_any(combined, expected_refs)
    topic_hits = contains_any(combined, expected_topics)

    score = 0
    if ref_hits:
        score += 2
    if topic_hits:
        score += 1
    if len(answer_text.strip()) >= 100:
        score += 1
    if "제공된 구절만으로는 정확히 답하기 어렵" in answer_text:
        score -= 1
    if "요청 처리 중 오류" in answer_text or "LLM 답변 생성 중 오류" in answer_text:
        score -= 2

    if score >= 3:
        verdict = "PASS"
    elif score == 2:
        verdict = "CHECK"
    else:
        verdict = "FAIL"

    return score, verdict, ref_hits, topic_hits

rows = []
with QUESTION_CSV.open("r", encoding="utf-utf-8-sig".replace("utf-utf", "utf")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

results = []

for idx, row in enumerate(rows, 1):
    if row.get("reset_before", "").upper() == "Y":
        reset_memory()

    q = row["question"]
    status, data = post_json("/chat", {"question": q, "session_id": session_id})

    question_type = str(data.get("question_type", ""))
    answer_text_raw = str(data.get("answer_text", data.get("error", "")))
    answer_main, evidence, evidence_marker = split_answer_evidence(answer_text_raw)

    combined_for_hits = answer_text_raw + "\n" + evidence
    ref_hits = contains_any(combined_for_hits, row.get("expected_reference_keywords", ""))
    topic_hits = contains_any(combined_for_hits, row.get("expected_topics", ""))

    score, verdict, _, _ = rough_score(
        answer_main,
        evidence,
        row.get("expected_reference_keywords", ""),
        row.get("expected_topics", ""),
    )

    route_expected = row.get("expected_route", "")
    route_ok = ""
    if route_expected:
        allowed = [x.strip() for x in route_expected.split("|")]
        route_ok = "Y" if question_type in allowed or any(a in question_type for a in allowed) else "N"

    results.append({
        "case_id": row["case_id"],
        "category": row["category"],
        "question": q,
        "http_status": status,
        "question_type": question_type,
        "expected_route": route_expected,
        "route_ok": route_ok,
        "expected_topics": row.get("expected_topics", ""),
        "topic_hits": topic_hits,
        "expected_reference_keywords": row.get("expected_reference_keywords", ""),
        "reference_hits": ref_hits,
        "score": score,
        "verdict": verdict,
        "answer_preview": re.sub(r"\s+", " ", answer_main)[:500],
        "evidence_preview": re.sub(r"\s+", " ", evidence)[:500],
        "manual_notes": "",
    })

    print(f"[{idx:02d}/{len(rows)}] {row['case_id']} {verdict} type={question_type} refs={ref_hits}")
    time.sleep(SLEEP_SEC)

with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
    fieldnames = [
        "case_id", "category", "question", "http_status", "question_type",
        "expected_route", "route_ok", "expected_topics", "topic_hits",
        "expected_reference_keywords", "reference_hits", "score", "verdict",
        "answer_preview", "evidence_preview", "manual_notes"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

summary = {}
for r in results:
    summary[r["verdict"]] = summary.get(r["verdict"], 0) + 1

print("\n=== Evaluation Summary ===")
print(summary)
print(f"saved: {OUTPUT_CSV.resolve()}")
