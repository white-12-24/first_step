# app.py
# --------------------------------------------------
# 이 코드는 성경공부 도우미 FastAPI 프로토타입 백엔드입니다.
# 역할:
# 1) CSV 로드
# 2) 질문 유형 분류
# 3) 절조회 / 장조회 / 주제검색 처리
# 4) 결과를 JSON으로 반환
# --------------------------------------------------

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
import re

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --------------------------------------------------
# 1. 데이터 경로 설정
# --------------------------------------------------
# 현재 프로젝트 루트: C:\py_temp\new_proj\BIBLE_RAG
# CSV 위치: data/processed/
# --------------------------------------------------
BASE_DIR = r"C:\py_temp\new_proj\BIBLE_RAG"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# --------------------------------------------------
# 2. CSV 로드
# --------------------------------------------------
bible_verses = pd.read_csv(os.path.join(PROCESSED_DIR, "bible_verses.csv"))
bible_chunks = pd.read_csv(os.path.join(PROCESSED_DIR, "bible_chunks.csv"))

# --------------------------------------------------
# 3. 책 이름 별칭표
# 역할:
# - 요, 롬, 계 같은 축약형을 정식 책 이름으로 변환
# --------------------------------------------------
book_alias_rows = [
    ["창세기", "창세기"], ["창", "창세기"], ["genesis", "창세기"], ["gen", "창세기"],
    ["출애굽기", "출애굽기"], ["출", "출애굽기"], ["exodus", "출애굽기"], ["exo", "출애굽기"],
    ["레위기", "레위기"], ["레", "레위기"], ["leviticus", "레위기"], ["lev", "레위기"],
    ["민수기", "민수기"], ["민", "민수기"], ["numbers", "민수기"], ["num", "민수기"],
    ["신명기", "신명기"], ["신", "신명기"], ["deuteronomy", "신명기"], ["deu", "신명기"],
    ["여호수아", "여호수아"], ["수", "여호수아"], ["joshua", "여호수아"], ["jos", "여호수아"],
    ["사사기", "사사기"], ["삿", "사사기"], ["judges", "사사기"], ["jdg", "사사기"],
    ["룻기", "룻기"], ["룻", "룻기"], ["ruth", "룻기"], ["rut", "룻기"],
    ["사무엘상", "사무엘상"], ["삼상", "사무엘상"], ["1samuel", "사무엘상"], ["1sa", "사무엘상"],
    ["사무엘하", "사무엘하"], ["삼하", "사무엘하"], ["2samuel", "사무엘하"], ["2sa", "사무엘하"],
    ["열왕기상", "열왕기상"], ["왕상", "열왕기상"], ["1kings", "열왕기상"], ["1ki", "열왕기상"],
    ["열왕기하", "열왕기하"], ["왕하", "열왕기하"], ["2kings", "열왕기하"], ["2ki", "열왕기하"],
    ["역대상", "역대상"], ["대상", "역대상"], ["1chronicles", "역대상"], ["1ch", "역대상"],
    ["역대하", "역대하"], ["대하", "역대하"], ["2chronicles", "역대하"], ["2ch", "역대하"],
    ["에스라", "에스라"], ["스", "에스라"], ["ezra", "에스라"], ["ezr", "에스라"],
    ["느헤미야", "느헤미야"], ["느", "느헤미야"], ["nehemiah", "느헤미야"], ["neh", "느헤미야"],
    ["에스더", "에스더"], ["에", "에스더"], ["esther", "에스더"], ["est", "에스더"],
    ["욥기", "욥기"], ["욥", "욥기"], ["job", "욥기"],
    ["시편", "시편"], ["시", "시편"], ["psalms", "시편"], ["psalm", "시편"], ["psa", "시편"],
    ["잠언", "잠언"], ["잠", "잠언"], ["proverbs", "잠언"], ["pro", "잠언"],
    ["전도서", "전도서"], ["전", "전도서"], ["ecclesiastes", "전도서"], ["ecc", "전도서"],
    ["아가", "아가"], ["songofsongs", "아가"], ["songofsolomon", "아가"], ["sng", "아가"],
    ["이사야", "이사야"], ["사", "이사야"], ["isaiah", "이사야"], ["isa", "이사야"],
    ["예레미야", "예레미야"], ["렘", "예레미야"], ["jeremiah", "예레미야"], ["jer", "예레미야"],
    ["예레미야애가", "예레미야 애가"], ["예레미야 애가", "예레미야 애가"], ["애", "예레미야 애가"], ["lamentations", "예레미야 애가"], ["lam", "예레미야 애가"],
    ["에스겔", "에스겔"], ["겔", "에스겔"], ["ezekiel", "에스겔"], ["ezk", "에스겔"],
    ["다니엘", "다니엘"], ["단", "다니엘"], ["daniel", "다니엘"], ["dan", "다니엘"],
    ["호세아", "호세아"], ["호", "호세아"], ["hosea", "호세아"], ["hos", "호세아"],
    ["요엘", "요엘"], ["욜", "요엘"], ["joel", "요엘"], ["jol", "요엘"],
    ["아모스", "아모스"], ["암", "아모스"], ["amos", "아모스"], ["amo", "아모스"],
    ["오바댜", "오바댜"], ["옵", "오바댜"], ["obadiah", "오바댜"], ["oba", "오바댜"],
    ["요나", "요나"], ["욘", "요나"], ["jonah", "요나"], ["jon", "요나"],
    ["미가", "미가"], ["미", "미가"], ["micah", "미가"], ["mic", "미가"],
    ["나훔", "나훔"], ["나", "나훔"], ["nahum", "나훔"], ["nam", "나훔"],
    ["하박국", "하박국"], ["합", "하박국"], ["habakkuk", "하박국"], ["hab", "하박국"],
    ["스바냐", "스바냐"], ["습", "스바냐"], ["zephaniah", "스바냐"], ["zep", "스바냐"],
    ["학개", "학개"], ["학", "학개"], ["haggai", "학개"], ["hag", "학개"],
    ["스가랴", "스가랴"], ["슥", "스가랴"], ["zechariah", "스가랴"], ["zec", "스가랴"],
    ["말라기", "말라기"], ["말", "말라기"], ["malachi", "말라기"], ["mal", "말라기"],
    ["마태복음", "마태복음"], ["마", "마태복음"], ["matthew", "마태복음"], ["mat", "마태복음"],
    ["마가복음", "마가복음"], ["막", "마가복음"], ["mark", "마가복음"], ["mrk", "마가복음"],
    ["누가복음", "누가복음"], ["눅", "누가복음"], ["luke", "누가복음"], ["luk", "누가복음"],
    ["요한복음", "요한복음"], ["요", "요한복음"], ["john", "요한복음"], ["jhn", "요한복음"],
    ["사도행전", "사도행전"], ["행", "사도행전"], ["acts", "사도행전"], ["act", "사도행전"],
    ["로마서", "로마서"], ["롬", "로마서"], ["romans", "로마서"], ["rom", "로마서"],
    ["고린도전서", "고린도전서"], ["고전", "고린도전서"], ["1corinthians", "고린도전서"], ["1co", "고린도전서"],
    ["고린도후서", "고린도후서"], ["고후", "고린도후서"], ["2corinthians", "고린도후서"], ["2co", "고린도후서"],
    ["갈라디아서", "갈라디아서"], ["갈", "갈라디아서"], ["galatians", "갈라디아서"], ["gal", "갈라디아서"],
    ["에베소서", "에베소서"], ["엡", "에베소서"], ["ephesians", "에베소서"], ["eph", "에베소서"],
    ["빌립보서", "빌립보서"], ["빌", "빌립보서"], ["philippians", "빌립보서"], ["php", "빌립보서"],
    ["골로새서", "골로새서"], ["골", "골로새서"], ["colossians", "골로새서"], ["col", "골로새서"],
    ["데살로니가전서", "데살로니가전서"], ["살전", "데살로니가전서"], ["1thessalonians", "데살로니가전서"], ["1th", "데살로니가전서"],
    ["데살로니가후서", "데살로니가후서"], ["살후", "데살로니가후서"], ["2thessalonians", "데살로니가후서"], ["2th", "데살로니가후서"],
    ["디모데전서", "디모데전서"], ["딤전", "디모데전서"], ["1timothy", "디모데전서"], ["1ti", "디모데전서"],
    ["디모데후서", "디모데후서"], ["딤후", "디모데후서"], ["2timothy", "디모데후서"], ["2ti", "디모데후서"],
    ["디도서", "디도서"], ["딛", "디도서"], ["titus", "디도서"], ["tit", "디도서"],
    ["빌레몬서", "빌레몬서"], ["몬", "빌레몬서"], ["philemon", "빌레몬서"], ["phm", "빌레몬서"],
    ["히브리서", "히브리서"], ["히", "히브리서"], ["hebrews", "히브리서"], ["heb", "히브리서"],
    ["야고보서", "야고보서"], ["약", "야고보서"], ["james", "야고보서"], ["jas", "야고보서"],
    ["베드로전서", "베드로전서"], ["벧전", "베드로전서"], ["1peter", "베드로전서"], ["1pe", "베드로전서"],
    ["베드로후서", "베드로후서"], ["벧후", "베드로후서"], ["2peter", "베드로후서"], ["2pe", "베드로후서"],
    ["요한1서", "요한1서"], ["요일", "요한1서"], ["1john", "요한1서"], ["1jn", "요한1서"],
    ["요한2서", "요한2서"], ["요이", "요한2서"], ["2john", "요한2서"], ["2jn", "요한2서"],
    ["요한3서", "요한3서"], ["요삼", "요한3서"], ["3john", "요한3서"], ["3jn", "요한3서"],
    ["유다서", "유다서"], ["유", "유다서"], ["jude", "유다서"], ["jud", "유다서"],
    ["요한계시록", "요한계시록"], ["계", "요한계시록"], ["revelation", "요한계시록"], ["rev", "요한계시록"],
]

book_alias_df = pd.DataFrame(book_alias_rows, columns=["alias", "book_kor"])
book_alias_df["alias_norm"] = (
    book_alias_df["alias"].astype(str).str.lower().str.replace(" ", "", regex=False)
)
book_alias_df = book_alias_df.drop_duplicates(subset=["alias_norm"]).reset_index(drop=True)

# --------------------------------------------------
# 4. 주제 사전
# 역할:
# - 주제검색 질문일 때 관련 청크를 찾기 위한 키워드 모음
# - 지금은 프로토타입용 1차 사전
# --------------------------------------------------
topic_dict = {
    "사랑": ["사랑", "사랑하사", "사랑하는", "인애", "자비", "긍휼"],
    "용서": ["용서", "사함", "죄 사함", "용납", "긍휼"],
    "믿음": ["믿음", "믿는", "신뢰", "의심치", "확신"],
    "소망": ["소망", "기대", "기다림", "영광", "약속"],
    "위로": ["위로", "평안", "안위", "두려워말라", "근심", "염려"],
    "기도": ["기도", "간구", "구하라", "부르짖", "기도하라"],
    "구원": ["구원", "영생", "멸망치", "구속", "복음"],
    "회개": ["회개", "돌이키", "죄인", "죄를", "회개하라"],
    "순종": ["순종", "지키", "계명", "듣고", "행하라"],
    "지혜": ["지혜", "명철", "훈계", "슬기", "깨닫"],
    "불안": ["두려워", "염려", "근심", "평안", "안심"],
    "고난": ["고난", "환난", "시험", "핍박", "눈물"],
}

# --------------------------------------------------
# 5. 메인 페이지 라우트
# 역할:
# - 브라우저에서 / 접속 시 index.html 화면 띄우기
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --------------------------------------------------
# 6. 질문 처리 API
# 역할:
# - 질문을 받아 유형 분류 후
# - 직접조회 / 장조회 / 주제검색 중 하나를 실행
# - 결과를 JSON으로 반환
# --------------------------------------------------
@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    q0 = str(body.get("question", "")).strip()
    q1 = q0.lower().replace(" ", "")

    # ------------------------------------------
    # 6-1. 질문 유형 분류
    # ------------------------------------------
    question_type = "unknown"

    if re.search(r"[0-9]+장[0-9]+절", q1) or re.search(r"[0-9]+:[0-9]+", q1):
        question_type = "verse_lookup"
    elif re.search(r"[0-9]+장", q1) or re.search(r"[0-9]+편", q1):
        question_type = "chapter_lookup"
    elif ("말씀" in q0) or ("구절" in q0) or ("추천" in q0):
        question_type = "topic_search"
    elif ("왜" in q0) or ("무엇" in q0) or ("설명" in q0) or ("의미" in q0):
        question_type = "explanation"

    # ------------------------------------------
    # 6-2. 책 이름 찾기
    # ------------------------------------------
    found_book = None
    found_alias = None

    for i in range(len(book_alias_df)):
        alias_norm = book_alias_df.loc[i, "alias_norm"]
        book_kor = book_alias_df.loc[i, "book_kor"]

        if alias_norm in q1:
            if (found_alias is None) or (len(alias_norm) > len(found_alias)):
                found_alias = alias_norm
                found_book = book_kor

    # ------------------------------------------
    # 6-3. 장/절 파싱
    # ------------------------------------------
    chapter = None
    verse = None

    m1 = re.search(r"([0-9]+)장([0-9]+)절", q1)
    if m1:
        chapter = int(m1.group(1))
        verse = int(m1.group(2))
    else:
        m2 = re.search(r"([0-9]+):([0-9]+)", q1)
        if m2:
            chapter = int(m2.group(1))
            verse = int(m2.group(2))
        else:
            m3 = re.search(r"([0-9]+)장", q1)
            if m3:
                chapter = int(m3.group(1))
            else:
                m4 = re.search(r"([0-9]+)편", q1)
                if m4:
                    chapter = int(m4.group(1))

    # ------------------------------------------
    # 6-4. 절 직접조회
    # ------------------------------------------
    if question_type == "verse_lookup":
        if found_book is None or chapter is None or verse is None:
            return JSONResponse({
                "question_type": question_type,
                "answer_text": "장절 정보를 정확히 인식하지 못했습니다."
            })

        x = bible_verses[
            (bible_verses["book_kor"] == found_book) &
            (bible_verses["chapter"] == chapter) &
            (bible_verses["verse"] == verse)
        ].copy()

        if len(x) == 0:
            return JSONResponse({
                "question_type": question_type,
                "answer_text": "해당 구절이 없습니다."
            })

        row = x.iloc[0]
        answer_text = f"[직접조회 결과]\n{row['book_kor']} {int(row['chapter'])}:{int(row['verse'])}\n{row['text']}"

        return JSONResponse({
            "question_type": question_type,
            "answer_text": answer_text
        })

    # ------------------------------------------
    # 6-5. 장 조회
    # ------------------------------------------
    if question_type == "chapter_lookup":
        if found_book is None or chapter is None:
            return JSONResponse({
                "question_type": question_type,
                "answer_text": "장 정보를 정확히 인식하지 못했습니다."
            })

        x = bible_verses[
            (bible_verses["book_kor"] == found_book) &
            (bible_verses["chapter"] == chapter)
        ].copy().sort_values("verse")

        if len(x) == 0:
            return JSONResponse({
                "question_type": question_type,
                "answer_text": "해당 장이 없습니다."
            })

        lines = []
        lines.append(f"[장조회 결과] {found_book} {chapter}장")
        lines.append("")

        for _, r in x.iterrows():
            lines.append(f"{int(r['verse'])}절 {r['text']}")

        lines.append("")
        lines.append(f"총 절 수: {len(x)}")

        return JSONResponse({
            "question_type": question_type,
            "answer_text": "\n".join(lines)
        })

    # ------------------------------------------
    # 6-6. 주제검색
    # ------------------------------------------
    if question_type == "topic_search":
        found_topics = []

        for topic_name, keywords in topic_dict.items():
            if topic_name in q0:
                found_topics.append(topic_name)
                continue

            for kw in keywords:
                if kw in q0:
                    found_topics.append(topic_name)
                    break

        found_topics = list(dict.fromkeys(found_topics))

        search_keywords = []
        for t in found_topics:
            search_keywords.extend(topic_dict[t])

        search_keywords = list(dict.fromkeys(search_keywords))

        temp = bible_chunks.copy()
        temp["score"] = 0

        for kw in search_keywords:
            temp["score"] += temp["text_chunk"].astype(str).str.count(re.escape(kw))

        temp = temp[temp["score"] > 0].copy()
        temp = temp.sort_values(["score", "book_kor", "chapter"], ascending=[False, True, True])

        if len(temp) == 0:
            return JSONResponse({
                "question_type": question_type,
                "answer_text": "관련 주제의 구절을 찾지 못했습니다."
            })

        lines = []
        lines.append("[주제검색 결과]")
        lines.append(f"인식 주제: {found_topics}")
        lines.append("")

        top_n = temp.head(5).copy()
        for _, r in top_n.iterrows():
            lines.append(f"- {r['book_kor']} {int(r['chapter'])}장 | score={int(r['score'])}")
            lines.append(f"  {r['text_chunk']}")
            lines.append("")

        return JSONResponse({
            "question_type": question_type,
            "answer_text": "\n".join(lines)
        })

    # ------------------------------------------
    # 6-7. 설명형 질문(임시)
    # ------------------------------------------
    if question_type == "explanation":
        return JSONResponse({
            "question_type": question_type,
            "answer_text": "설명형 질문은 다음 단계에서 검색 결과 + 설명 생성으로 연결할 예정입니다."
        })

    return JSONResponse({
        "question_type": question_type,
        "answer_text": "질문 유형을 아직 인식하지 못했습니다."
    })