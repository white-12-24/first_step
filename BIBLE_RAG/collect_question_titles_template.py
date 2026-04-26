# collect_question_titles_template.py
# --------------------------------------------------
# 목적:
# - 공개 Q&A 사이트의 "질문 제목"만 수집해서 평가 질문 후보로 참고한다.
# - 답변 본문을 무단 복제하지 않는다.
# - 사이트 이용약관/robots.txt/저작권을 반드시 확인한다.
# - 요청 간 sleep을 두고, 개인 프로젝트 검토용으로만 사용한다.
# --------------------------------------------------

import time
import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

URLS = [
    # 예시: GotQuestions 한국어 카테고리/FAQ 페이지
    "https://www.gotquestions.org/Korean/Korean-FAQ.html",
    "https://www.gotquestions.org/Korean/Korean-crucial.html",
    "https://www.gotquestions.org/Korean/Korean-Q-theology.html",
]

OUT = "collected_question_titles.csv"

headers = {
    "User-Agent": "BibleRAGStudyBot/0.1 (personal study; contact: local)"
}

rows = []

for url in URLS:
    print("fetch:", url)
    resp = requests.get(url, headers=headers, timeout=20)
    print("status:", resp.status_code)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    for a in soup.find_all("a"):
        title = a.get_text(" ", strip=True)
        href = a.get("href", "")

        if not title:
            continue

        # 질문형 제목 후보만 대략 필터링
        if ("?" in title) or ("인가" in title) or ("무엇" in title) or ("어떻게" in title) or ("왜" in title):
            rows.append({
                "source_url": url,
                "question_title": title,
                "link": urljoin(url, href),
            })

    time.sleep(2)

# 중복 제거
seen = set()
dedup = []
for r in rows:
    key = r["question_title"]
    if key not in seen:
        seen.add(key)
        dedup.append(r)

with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["source_url", "question_title", "link"])
    writer.writeheader()
    writer.writerows(dedup)

print("saved:", OUT)
print("count:", len(dedup))
