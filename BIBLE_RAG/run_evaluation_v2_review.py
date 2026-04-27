# run_evaluation_v2_review.py
# --------------------------------------------------
# 개선점:
# - expected_topics / expected_reference_keywords가 없는 질문은 FAIL이 아니라 REVIEW_UNLABELED로 분류
# - 라벨 있는 질문만 PASS/CHECK/FAIL로 자동평가
# - guardrail 오탐, 장절 포함 해석 질문 직접조회 문제를 issue_type으로 표시
#
# 실행:
# 1) uvicorn app:app --port 8000
# 2) QUESTION_CSV 값을 평가할 CSV로 설정
# 3) python run_evaluation_v2_review.py
# --------------------------------------------------

import csv
import json
import re
import time
import uuid
from pathlib import Path
from urllib import request, error

BASE_URL = "http://127.0.0.1:8000"
QUESTION_CSV = Path("bible_rag_eval_questions_from_gotquestions.csv")
OUTPUT_CSV = Path("evaluation_results_reviewed.csv")
SLEEP_SEC = 0.2

session_id = "eval_" + str(uuid.uuid4())

def has_value(x):
    return x is not None and str(x).strip() != ""

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
    kws = [x.strip() for x in str(pipe_keywords).split("|") if x.strip() and x.strip().lower() != "nan"]
    if not kws:
        return ""
    hits = []
    for kw in kws:
        if kw in text:
            hits.append(kw)
    return "|".join(hits)

def score_labeled(answer_text, evidence_text, expected_refs, expected_topics):
    combined = (answer_text or "") + "\n" + (evidence_text or "")
    ref_hits = contains_any(combined, expected_refs)
    topic_hits = contains_any(combined, expected_topics)

    score = 0
    if ref_hits:
        score += 2
    if topic_hits:
        score += 1
    if len(str(answer_text).strip()) >= 100:
        score += 1
    if "LLM 답변 생성 중 오류" in str(answer_text) or "요청 처리 중 오류" in str(answer_text):
        score -= 2

    if score >= 3:
        verdict = "PASS"
    elif score == 2:
        verdict = "CHECK"
    else:
        verdict = "FAIL"

    return score, verdict, ref_hits, topic_hits

def detect_issue_type(question, question_type, final_verdict, expected_topics):
    q = str(question)

    if final_verdict == "REVIEW_UNLABELED":
        return "정답 라벨 없음: 수동검토/라벨링 필요"

    if question_type == "guardrail" and has_value(expected_topics):
        return "guardrail 오탐 가능성"

    if question_type == "verse_lookup":
        interpret_words = ["가르치", "뜻", "의미", "필요", "참조", "무엇", "인가요", "인가?", "설명"]
        if any(w in q for w in interpret_words):
            return "장절 포함 해석 질문이 직접조회로 처리됨"

    if final_verdict == "FAIL":
        return "근거/주제 hit 실패"

    if final_verdict == "CHECK":
        return "부분 성공: 근거 보강 또는 라벨 점검"

    return "정상"

with QUESTION_CSV.open("r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

results = []

for idx, row in enumerate(rows, 1):
    if row.get("reset_before", "Y").upper() == "Y":
        reset_memory()

    q = row["question"]
    status, data = post_json("/chat", {"question": q, "session_id": session_id})

    question_type = str(data.get("question_type", ""))
    answer_text_raw = str(data.get("answer_text", data.get("error", "")))
    answer_main, evidence, evidence_marker = split_answer_evidence(answer_text_raw)

    expected_topics = row.get("expected_topics", "")
    expected_refs = row.get("expected_reference_keywords", "")
    labeled = has_value(expected_topics) or has_value(expected_refs)

    if labeled:
        score, verdict, ref_hits, topic_hits = score_labeled(answer_main, evidence, expected_refs, expected_topics)
        final_verdict = verdict
    else:
        score = ""
        verdict = ""
        ref_hits = ""
        topic_hits = ""
        final_verdict = "REVIEW_UNLABELED"

    route_expected = row.get("expected_route", "")
    route_ok = ""
    if has_value(route_expected):
        allowed = [x.strip() for x in route_expected.split("|") if x.strip()]
        route_ok = "Y" if question_type in allowed or any(a in question_type for a in allowed) else "N"

    issue_type = detect_issue_type(q, question_type, final_verdict, expected_topics)

    results.append({
        "case_id": row.get("case_id", f"CASE{idx:04d}"),
        "category": row.get("category", ""),
        "question": q,
        "http_status": status,
        "question_type": question_type,
        "expected_route": route_expected,
        "route_ok": route_ok,
        "expected_topics": expected_topics,
        "topic_hits": topic_hits,
        "expected_reference_keywords": expected_refs,
        "reference_hits": ref_hits,
        "score": score,
        "raw_verdict": verdict,
        "final_verdict": final_verdict,
        "issue_type": issue_type,
        "answer_preview": re.sub(r"\s+", " ", answer_main)[:500],
        "evidence_preview": re.sub(r"\s+", " ", evidence)[:500],
        "manual_notes": "",
    })

    print(f"[{idx:03d}/{len(rows)}] {row.get('case_id', '')} {final_verdict} type={question_type} issue={issue_type}")
    time.sleep(SLEEP_SEC)

with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
    fieldnames = [
        "case_id", "category", "question", "http_status", "question_type",
        "expected_route", "route_ok", "expected_topics", "topic_hits",
        "expected_reference_keywords", "reference_hits", "score",
        "raw_verdict", "final_verdict", "issue_type",
        "answer_preview", "evidence_preview", "manual_notes"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

summary = {}
for r in results:
    summary[r["final_verdict"]] = summary.get(r["final_verdict"], 0) + 1

print("\n=== Evaluation Summary ===")
print(summary)
print(f"saved: {OUTPUT_CSV.resolve()}")
