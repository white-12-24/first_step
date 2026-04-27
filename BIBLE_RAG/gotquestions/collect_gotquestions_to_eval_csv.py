# collect_gotquestions_to_eval_csv.py
# --------------------------------------------------
# GotQuestions 한국어 페이지에서 '질문 제목'만 수집해 평가 CSV 형태로 변환하는 코드
# 주의:
# - 답변 본문은 수집하지 않음
# - 질문 제목/링크만 평가 질문 후보로 활용
# - 사이트 약관/robots.txt 확인 후 개인 프로젝트 검증용으로 사용
# - 요청 간 sleep을 두어 서버에 부담을 주지 않음
# --------------------------------------------------

import csv
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urljoin

# --------------------------------------------------
# 1. 수집 대상 페이지
# 필요하면 GotQuestions 한국어 카테고리 URL을 추가
# --------------------------------------------------
URLS = [
    "https://www.gotquestions.org/Korean/Korean-FAQ.html",
    "https://www.gotquestions.org/Korean/Korean-crucial.html",
    "https://www.gotquestions.org/Korean/Korean-Q-God.html",
    "https://www.gotquestions.org/Korean/Korean-Q-Jesus.html",
    "https://www.gotquestions.org/Korean/Korean-Q-Holy-Spirit.html",
    "https://www.gotquestions.org/Korean/Korean-Q-Salvation.html",
    "https://www.gotquestions.org/Korean/Korean-Q-Bible.html",
    "https://www.gotquestions.org/Korean/Korean-Q-Church.html",
    "https://www.gotquestions.org/Korean/Korean-Q-Christian.html",
    "https://www.gotquestions.org/Korean/Korean-Q-prayer.html",
    "https://www.gotquestions.org/Korean/Korean-Q-sin.html",
    "https://www.gotquestions.org/Korean/Korean-Q-marriage.html",
    "https://www.gotquestions.org/Korean/Korean-Q-relationships.html",
    "https://www.gotquestions.org/Korean/Korean-Q-family.html",
    "https://www.gotquestions.org/Korean/Korean-Q-theology.html",
    "https://www.gotquestions.org/Korean/Korean-Q-worldview.html",
    "https://www.gotquestions.org/Korean/Korean-Q-creation.html",
    "https://www.gotquestions.org/Korean/Korean-Q-end-times.html",
    "https://www.gotquestions.org/Korean/Korean-Q-miscellaneous.html",
]

OUT_TITLES = Path("gotquestions_question_titles.csv")
OUT_EVAL = Path("bible_rag_eval_questions_from_gotquestions.csv")
SLEEP_SEC = 2.0
MAX_QUESTIONS = 300

HEADERS = {
    "User-Agent": "BibleRAGStudyBot/0.1 (question-title collection for local evaluation)"
}

# --------------------------------------------------
# 2. HTML a 태그 파서
# --------------------------------------------------
class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_a = False
        self.current_href = ""
        self.current_text = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self.in_a = True
            self.current_href = ""
            self.current_text = []
            for k, v in attrs:
                if k.lower() == "href":
                    self.current_href = v or ""

    def handle_data(self, data):
        if self.in_a:
            self.current_text.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self.in_a:
            text = " ".join("".join(self.current_text).split())
            href = self.current_href
            if text:
                self.links.append((text, href))
            self.in_a = False
            self.current_href = ""
            self.current_text = []


def fetch_html(url):
    req = Request(url, headers=HEADERS)
    with urlopen(req, timeout=30) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="replace")


def is_question_title(text):
    t = str(text).strip()
    if len(t) < 8:
        return False
    bad_words = [
        "처음으로", "구독", "검색", "질문하기", "더 알아보기", "개인정보", "연락", "영생을", "용서를", "English",
    ]
    for b in bad_words:
        if b in t:
            return False
    question_markers = ["?", "무엇", "어떻게", "왜", "인가", "입니까", "합니까", "말하는가", "가르치는가", "뜻", "의미"]
    return any(m in t for m in question_markers)

# --------------------------------------------------
# 3. 간단 휴리스틱으로 평가 CSV 컬럼 채우기
# 완벽한 정답표가 아니라, 자동평가 초안 생성용
# 이후 FAIL/CHECK를 보고 수동 보정하면 됨
# --------------------------------------------------
TOPIC_RULES = {
    "삼위일체": ("삼위일체|하나님|예수|성령", "마태복음 28|요한복음 1|고린도후서 13"),
    "성령": ("성령|은사", "요한복음 14|사도행전 2|로마서 8|고린도전서 12"),
    "방언": ("방언|은사|성령", "사도행전 2|고린도전서 12|고린도전서 14|고린도전서 13"),
    "구원": ("구원|믿음|은혜", "요한복음 3|로마서 10|에베소서 2"),
    "믿음": ("믿음|구원|순종", "히브리서 11|에베소서 2|야고보서 2"),
    "십일조": ("재물|예배|순종", "말라기 3|고린도후서 9|마태복음 6"),
    "문신": ("몸|윤리|거룩", "레위기 19|고린도전서 6|고린도전서 10|로마서 14|베드로전서 3"),
    "피어싱": ("몸|윤리|거룩", "레위기 19|고린도전서 6|고린도전서 10|로마서 14|베드로전서 3"),
    "술": ("절제|윤리|거룩", "에베소서 5|고린도전서 6|잠언 20|로마서 14"),
    "도박": ("재물|지혜|죄", "디모데전서 6|마태복음 6|잠언"),
    "이혼": ("결혼|가족|관계", "마태복음 19|고린도전서 7|에베소서 5"),
    "재혼": ("결혼|가족|관계", "마태복음 19|고린도전서 7|로마서 7"),
    "기도": ("기도|믿음", "마태복음 6|빌립보서 4|야고보서 5"),
    "죄": ("죄|회개|은혜", "요한1서 1|시편 51|로마서 6"),
    "회개": ("회개|죄|은혜", "시편 51|누가복음 15|사도행전 2"),
    "천국": ("죽음|구원|소망", "요한복음 14|요한계시록 21|고린도전서 15"),
    "지옥": ("죽음|구원|심판", "마태복음 25|요한계시록 20|누가복음 16"),
    "교회": ("교회|관계|섬김", "에베소서 4|고린도전서 12|마태복음 18"),
    "정치": ("정치|지혜|순종", "로마서 13|디모데전서 2|마태복음 22"),
    "환경": ("환경|창조|섬김", "창세기 1|창세기 2|시편 24"),
    "창조": ("창조|말씀|지혜", "창세기 1|창세기 2|시편 19"),
    "요한계시록": ("종말|소망|말씀", "요한계시록 21|마태복음 24|데살로니가전서 4"),
}

STORY_RULES = {
    "탕자": ("탕자", "누가복음 15"),
    "선한 사마리아": ("선한사마리아인", "누가복음 10"),
    "십계명": ("십계명", "출애굽기 20"),
    "주기도문": ("주기도문", "마태복음 6"),
    "다윗": ("다윗과골리앗", "사무엘상 17"),
    "골리앗": ("다윗과골리앗", "사무엘상 17"),
    "오순절": ("성령강림", "사도행전 2"),
    "바울": ("바울회심", "사도행전 9"),
}


def build_eval_row(case_id, title, source_url, link):
    route = "topic_search|explanation"
    topics = ""
    story_key = ""
    refs = ""
    category = "gotquestions_title"

    for k, (sk, ref) in STORY_RULES.items():
        if k in title:
            route = "story|story_explanation|topic_search|explanation"
            story_key = sk
            refs = ref
            category = "gotquestions_story_or_doctrine"
            break

    if refs == "":
        for k, (t, r) in TOPIC_RULES.items():
            if k in title:
                topics = t
                refs = r
                category = "gotquestions_topic"
                break

    if refs == "":
        # 모르는 질문도 평가 대상에 넣되, 자동판정은 CHECK로 보게끔 넓게 둠
        topics = ""
        refs = ""
        category = "gotquestions_unmapped"

    return {
        "case_id": case_id,
        "category": category,
        "question": title,
        "purpose": "GotQuestions 질문 제목 기반 다양화 평가",
        "expected_route": route,
        "expected_topics": topics,
        "expected_story_key": story_key,
        "expected_reference_keywords": refs,
        "reset_before": "Y",
        "followup_group": "",
        "source_inspiration": source_url,
        "source_link": link,
    }

# --------------------------------------------------
# 4. 수집 실행
# --------------------------------------------------
all_rows = []
seen = set()

for url in URLS:
    print("fetch:", url)
    try:
        html = fetch_html(url)
    except Exception as e:
        print("  error:", e)
        continue

    parser = LinkParser()
    parser.feed(html)

    for text, href in parser.links:
        title = re.sub(r"\s+", " ", text).strip()
        if not is_question_title(title):
            continue
        if title in seen:
            continue
        seen.add(title)
        link = urljoin(url, href)
        all_rows.append({
            "source_url": url,
            "question_title": title,
            "link": link,
        })

        if len(all_rows) >= MAX_QUESTIONS:
            break

    time.sleep(SLEEP_SEC)
    if len(all_rows) >= MAX_QUESTIONS:
        break

# 질문 제목 원본 저장
with OUT_TITLES.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["source_url", "question_title", "link"])
    writer.writeheader()
    writer.writerows(all_rows)

# 평가 CSV 저장
fieldnames = [
    "case_id", "category", "question", "purpose", "expected_route", "expected_topics",
    "expected_story_key", "expected_reference_keywords", "reset_before", "followup_group",
    "source_inspiration", "source_link"
]

eval_rows = []
for idx, r in enumerate(all_rows, 1):
    eval_rows.append(build_eval_row(f"GQ{idx:04d}", r["question_title"], r["source_url"], r["link"]))

with OUT_EVAL.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(eval_rows)

print("saved:", OUT_TITLES.resolve())
print("saved:", OUT_EVAL.resolve())
print("count:", len(eval_rows))
print("주의: expected_topics/reference는 휴리스틱 초안이므로 CHECK/FAIL 결과를 보고 수동 보정하세요.")
