# app.py
# --------------------------------------------------
# 성경 RAG 챗봇 FastAPI 백엔드
#
# 포함 기능:
# 1) 절 직접조회
# 2) 장 전체 조회
# 3) 장/절 설명 자동화
# 4) 주제 기반 하이브리드 검색
# 5) 벡터DB + reranker
# 6) LLM 근거 기반 답변
# 7) story_map 기반 비유/사건 직접 매핑
# 8) session_id별 대화형 기억
# 9) 최근 검색 근거 evidence 저장 및 후속 질문에서 재사용
# --------------------------------------------------

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import os
import re
import pandas as pd
import chromadb

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder


# --------------------------------------------------
# 0. 기본 설정
# --------------------------------------------------
app = FastAPI()

BASE_DIR = r"C:\py_temp\new_proj\BIBLE_RAG"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
ENV_PATH = os.path.join(BASE_DIR, ".env")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# --------------------------------------------------
# 1. 데이터 로드
# --------------------------------------------------
bible_verses = pd.read_csv(os.path.join(PROCESSED_DIR, "bible_verses.csv"))
bible_chunks = pd.read_csv(os.path.join(PROCESSED_DIR, "bible_chunks.csv"))

print("bible_verses:", len(bible_verses))
print("bible_chunks:", len(bible_chunks))


# --------------------------------------------------
# 2. OpenAI 설정
# --------------------------------------------------
# 중요:
# - ENV_PATH를 먼저 만든 뒤 load_dotenv를 실행해야 함
# - override=True로 Windows 환경변수보다 현재 프로젝트 .env 값을 우선 사용
# - 키 앞뒤 공백/따옴표 제거
# --------------------------------------------------
load_dotenv(dotenv_path=ENV_PATH, override=True)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None or OPENAI_API_KEY.strip() == "":
    print("[경고] OPENAI_API_KEY를 찾지 못했습니다. .env 파일을 확인하세요.")
    openai_client = None
else:
    OPENAI_API_KEY = OPENAI_API_KEY.strip().replace('"', '').replace("'", "")
    print("[확인] OPENAI_API_KEY 로드됨:", OPENAI_API_KEY[:10] + "..." + OPENAI_API_KEY[-4:])
    print("[확인] OPENAI_MODEL:", OPENAI_MODEL)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)


# --------------------------------------------------
# 3. 벡터DB / 임베딩 / reranker 로드
# --------------------------------------------------
embedding_model_name = "intfloat/multilingual-e5-small"
embedding_model = SentenceTransformer(embedding_model_name)

chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
vector_collection = chroma_client.get_collection(name="bible_chunks")

reranker_model_name = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
reranker_model = CrossEncoder(reranker_model_name)

print("vector collection count:", vector_collection.count())


# --------------------------------------------------
# 4. session_id별 대화형 기억 저장소
# --------------------------------------------------
chat_memory_store = {}
MAX_MEMORY = 5


# --------------------------------------------------
# 5. 책 이름 별칭표
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
# 6. 주제 사전
# --------------------------------------------------
topic_dict = {
    "사랑": ["사랑", "사랑하사", "사랑하는", "인애", "자비", "긍휼"],
    "용서": ["용서", "사함", "죄 사함", "용납", "긍휼"],
    "믿음": ["믿음", "믿는", "신뢰", "의심치", "확신"],
    "소망": ["소망", "기대", "기다림", "영광", "약속"],
    "위로": ["위로", "평안", "안위", "두려워", "염려", "근심"],
    "기도": ["기도", "간구", "구하라", "부르짖", "기도하라"],
    "구원": ["구원", "영생", "멸망", "구속", "복음"],
    "회개": ["회개", "돌이키", "죄인", "죄를", "회개하라"],
    "순종": ["순종", "지키", "계명", "듣고", "행하라"],
    "지혜": ["지혜", "명철", "훈계", "슬기", "깨닫"],
    "불안": ["두려워", "염려", "근심", "평안", "안심"],
    "고난": ["고난", "환난", "시험", "핍박", "눈물", "위로"],

    # 확장 주제
    "비유": ["비유", "탕자", "사마리아", "씨 뿌리는", "달란트", "잃은 양"],
    "회복": ["회복", "돌아오", "찾았", "살아났", "잃었다가"],
    "죄": ["죄", "죄인", "범죄", "악", "불의"],
    "은혜": ["은혜", "긍휼", "자비", "은총"],
    "평안": ["평안", "평강", "안심", "두려워", "염려"],
    "인내": ["인내", "참음", "견디", "시험", "환난"],
    "감사": ["감사", "감사하", "찬송", "기뻐"],
    "겸손": ["겸손", "낮추", "교만", "온유"],
    "재물": ["재물", "돈", "부자", "가난", "소유"],
    "섬김": ["섬기", "종", "봉사", "이웃"],
}


# --------------------------------------------------
# 7. 주제별 질의 확장어
# --------------------------------------------------
query_expand_dict = {
    "불안": ["두려워 말라", "염려하지 말라", "평안을 너희에게 주노라", "강하고 담대하라", "근심하지 말라"],
    "위로": ["위로", "평안", "소망", "환난 중 위로", "두려워 말라"],
    "사랑": ["하나님의 사랑", "서로 사랑하라", "사랑은 하나님께 속한 것", "독생자를 주신 사랑"],
    "기도": ["기도하라", "간구", "부르짖음", "구하라", "의인의 간구"],
    "용서": ["용서하라", "죄 사함", "서로 용서", "긍휼", "사하여 주옵소서"],
    "고난": ["고난", "환난", "시험", "위로", "인내"],
    "비유": ["예수님의 비유", "탕자의 비유", "선한 사마리아인", "씨 뿌리는 비유"],
    "회복": ["잃었다가 찾음", "돌아옴", "회개", "용서", "회복"],
}


# --------------------------------------------------
# 8. 대표 구절 boost
# --------------------------------------------------
representative_boost = {
    "불안": [("마태복음", 6), ("빌립보서", 4), ("요한복음", 14), ("이사야", 41), ("시편", 23)],
    "사랑": [("요한복음", 3), ("요한1서", 4), ("고린도전서", 13), ("로마서", 5)],
    "기도": [("마태복음", 6), ("야고보서", 5), ("빌립보서", 4), ("역대하", 6), ("열왕기상", 8)],
    "용서": [("마태복음", 6), ("골로새서", 3), ("고린도후서", 2), ("누가복음", 23)],
    "위로": [("고린도후서", 1), ("이사야", 41), ("시편", 23), ("요한복음", 14)],
    "고난": [("고린도후서", 1), ("로마서", 5), ("야고보서", 1), ("베드로전서", 4)],
    "비유": [("누가복음", 15), ("누가복음", 10), ("마태복음", 13), ("마태복음", 25)],
}


# --------------------------------------------------
# 9. 비유/사건 직접 매핑
# --------------------------------------------------
story_map = {
    "탕자": {
        "title": "탕자의 비유",
        "book": "누가복음",
        "chapter": 15,
        "start_verse": 11,
        "end_verse": 32
    },
    "탕자의비유": {
        "title": "탕자의 비유",
        "book": "누가복음",
        "chapter": 15,
        "start_verse": 11,
        "end_verse": 32
    },
    "돌아온아들": {
        "title": "탕자의 비유",
        "book": "누가복음",
        "chapter": 15,
        "start_verse": 11,
        "end_verse": 32
    },
    "선한사마리아인": {
        "title": "선한 사마리아인의 비유",
        "book": "누가복음",
        "chapter": 10,
        "start_verse": 25,
        "end_verse": 37
    },
    "씨뿌리는비유": {
        "title": "씨 뿌리는 비유",
        "book": "마태복음",
        "chapter": 13,
        "start_verse": 1,
        "end_verse": 23
    },
    "잃은양": {
        "title": "잃은 양의 비유",
        "book": "누가복음",
        "chapter": 15,
        "start_verse": 1,
        "end_verse": 7
    },
    "달란트": {
        "title": "달란트 비유",
        "book": "마태복음",
        "chapter": 25,
        "start_verse": 14,
        "end_verse": 30
    },
    "다윗과골리앗": {
        "title": "다윗과 골리앗",
        "book": "사무엘상",
        "chapter": 17,
        "start_verse": 1,
        "end_verse": 58
    },
}


# --------------------------------------------------
# 10. 메인 페이지
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# --------------------------------------------------
# 11. 대화 기억 초기화 API
# --------------------------------------------------
@app.post("/reset_memory")
async def reset_memory(request: Request):
    global chat_memory_store

    body = await request.json()
    session_id = str(body.get("session_id", "default")).strip()

    if session_id in chat_memory_store:
        chat_memory_store[session_id] = []

    return JSONResponse({
        "status": "ok",
        "message": "대화 기억이 초기화되었습니다."
    })


# --------------------------------------------------
# 12. 질문 처리 API
# --------------------------------------------------
@app.post("/chat")
async def chat(request: Request):
    global chat_memory_store

    # --------------------------------------------------
    # 12-1. 사용자 질문 / session_id 받기
    # --------------------------------------------------
    body = await request.json()
    q0 = str(body.get("question", "")).strip()
    q1 = q0.lower().replace(" ", "")

    session_id = str(body.get("session_id", "default")).strip()
    if session_id == "":
        session_id = "default"

    if session_id not in chat_memory_store:
        chat_memory_store[session_id] = []

    chat_memory = chat_memory_store[session_id]

    # --------------------------------------------------
    # 12-2. 설명/후속 질문 감지
    # --------------------------------------------------
    has_explain_intent = (
        ("설명" in q0) or
        ("해석" in q0) or
        ("의미" in q0) or
        ("뜻" in q0) or
        ("요약" in q0) or
        ("풀어서" in q0) or
        ("자세히" in q0) or
        ("적용" in q0)
    )

    followup_words = [
        "더", "자세히", "그거", "그 부분", "이 내용", "그 내용",
        "이걸", "이것", "여기서", "위에서", "앞에서", "방금",
        "적용", "예시", "다시", "쉽게", "아버지", "아들은", "그 사람",
        "그럼", "그러면", "이 말씀", "이 구절"
    ]

    is_followup = any(w in q0 for w in followup_words) and len(chat_memory) > 0

    recent_memory_text = ""
    if len(chat_memory) > 0:
        recent_items = chat_memory[-3:]
        memory_lines = []

        for item in recent_items:
            memory_lines.append(f"이전 질문: {item.get('question', '')}")
            memory_lines.append(f"이전 주제: {item.get('topic', '')}")
            memory_lines.append(f"이전 요약: {item.get('summary', '')}")

        recent_memory_text = "\n".join(memory_lines)

    if is_followup:
        last_question = chat_memory[-1].get("question", "")
        last_topic = chat_memory[-1].get("topic", "")
        retrieval_query = f"{last_question} {last_topic} {q0}"
    else:
        retrieval_query = q0

    retrieval_query_norm = retrieval_query.lower().replace(" ", "")

    # --------------------------------------------------
    # 12-3. 질문 유형 분류
    # --------------------------------------------------
    question_type = "unknown"

    if re.search(r"[0-9]+장[0-9]+절", q1) or re.search(r"[0-9]+:[0-9]+", q1):
        question_type = "verse_lookup"
    elif re.search(r"[0-9]+장", q1) or re.search(r"[0-9]+편", q1):
        question_type = "chapter_lookup"
    elif ("말씀" in q0) or ("구절" in q0) or ("추천" in q0):
        question_type = "topic_search"
    elif has_explain_intent or ("왜" in q0) or ("무엇" in q0):
        question_type = "explanation"
    elif is_followup:
        question_type = "explanation"

    # --------------------------------------------------
    # 12-4. 책 이름 찾기
    # --------------------------------------------------
    found_book = None
    found_alias = None

    for i in range(len(book_alias_df)):
        alias_norm = book_alias_df.loc[i, "alias_norm"]
        book_kor = book_alias_df.loc[i, "book_kor"]

        if alias_norm in q1:
            if (found_alias is None) or (len(alias_norm) > len(found_alias)):
                found_alias = alias_norm
                found_book = book_kor

    # --------------------------------------------------
    # 12-5. 장/절 파싱
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 13. 절 직접조회 / 절 설명
    # --------------------------------------------------
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

        if not has_explain_intent:
            memory_evidence = f"{row['book_kor']} {int(row['chapter'])}:{int(row['verse'])} {row['text']}"

            chat_memory.append({
                "question": q0,
                "topic": f"{row['book_kor']} {int(row['chapter'])}:{int(row['verse'])}",
                "summary": str(row["text"])[:200],
                "evidence": memory_evidence,
                "evidence_debug": memory_evidence
            })

            chat_memory_store[session_id] = chat_memory[-MAX_MEMORY:]

            answer_text = (
                f"[직접조회 결과]\n"
                f"{row['book_kor']} {int(row['chapter'])}:{int(row['verse'])}\n"
                f"{row['text']}"
            )

            return JSONResponse({
                "question_type": question_type,
                "answer_text": answer_text
            })

        context_df = bible_verses[
            (bible_verses["book_kor"] == found_book) &
            (bible_verses["chapter"] == chapter) &
            (bible_verses["verse"] >= max(1, verse - 2)) &
            (bible_verses["verse"] <= verse + 2)
        ].copy().sort_values("verse")

        evidence_lines = []
        for _, r in context_df.iterrows():
            evidence_lines.append(
                f"{r['book_kor']} {int(r['chapter'])}:{int(r['verse'])} {r['text']}"
            )

        evidence_text = "\n".join(evidence_lines)

        system_prompt = """
너는 성경공부를 돕는 조심스러운 AI 도우미다.
반드시 제공된 성경 근거 안에서만 답변한다.
근거에 없는 내용은 단정하지 않는다.
특정 교단의 교리 논쟁은 단정하지 말고 본문 중심으로 설명한다.
오탈자 없이 자연스러운 한국어로 답변한다.
답변은 한국어로 한다.
"""

        user_prompt = f"""
최근 대화 맥락:
{recent_memory_text}

사용자 질문:
{q0}

성경 근거:
{evidence_text}

답변 형식:
1. 핵심 답변
- 질문에 대해 4~6문장으로 자연스럽게 답한다.
- 구절의 핵심 메시지를 먼저 설명하고, 사용자가 성경공부를 하는 상황이라고 생각하고 친절하게 풀어준다.

2. 관련 구절 설명
- 제공된 근거 중 핵심 구절 2~3개를 선택한다.
- 각 구절마다 2~3문장으로 왜 중요한지 설명한다.

3. 본문 설명
- 앞뒤 문맥을 바탕으로 3~4문단으로 설명한다.
- 본문의 흐름, 핵심 의미, 오늘날 적용을 나누어 설명한다.
- 근거에 없는 내용은 단정하지 않는다.

4. 적용 질문 2개
- 단순 질문이 아니라 실제 삶을 돌아볼 수 있는 질문으로 작성한다.
"""

        try:
            llm_response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            llm_answer = llm_response.choices[0].message.content

        except Exception as e:
            llm_answer = f"LLM 답변 생성 중 오류가 발생했습니다: {e}"

        chat_memory.append({
            "question": q0,
            "topic": f"{found_book} {chapter}:{verse}",
            "summary": llm_answer[:250],
            "evidence": evidence_text,
            "evidence_debug": evidence_text
        })

        chat_memory_store[session_id] = chat_memory[-MAX_MEMORY:]

        return JSONResponse({
            "question_type": "verse_explanation",
            "answer_text": llm_answer + "\n\n---\n[검색 근거]\n" + evidence_text
        })

    # --------------------------------------------------
    # 14. 장 조회 / 장 설명
    # --------------------------------------------------
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

        if not has_explain_intent:
            lines = []
            lines.append(f"[장조회 결과] {found_book} {chapter}장")
            lines.append("")

            for _, r in x.iterrows():
                lines.append(f"{int(r['verse'])}절 {r['text']}")

            lines.append("")
            lines.append(f"총 절 수: {len(x)}")

            memory_evidence = "\n".join(lines)

            chat_memory.append({
                "question": q0,
                "topic": f"{found_book} {chapter}장",
                "summary": "장 전체 조회",
                "evidence": memory_evidence,
                "evidence_debug": memory_evidence
            })

            chat_memory_store[session_id] = chat_memory[-MAX_MEMORY:]

            return JSONResponse({
                "question_type": question_type,
                "answer_text": "\n".join(lines)
            })

        evidence_lines = []
        for _, r in x.iterrows():
            evidence_lines.append(
                f"{r['book_kor']} {int(r['chapter'])}:{int(r['verse'])} {r['text']}"
            )

        evidence_text = "\n".join(evidence_lines)

        system_prompt = """
너는 성경공부를 돕는 조심스러운 AI 도우미다.
반드시 제공된 성경 본문 안에서만 요약하고 설명한다.
근거에 없는 내용은 단정하지 않는다.
오탈자 없이 자연스러운 한국어로 답변한다.
답변은 한국어로 한다.
"""

        user_prompt = f"""
최근 대화 맥락:
{recent_memory_text}

사용자 질문:
{q0}

성경 본문:
{evidence_text}

답변 형식:
1. 핵심 요약
- 이 장의 중심 내용을 4~6문장으로 요약한다.
- 너무 짧게 결론만 말하지 말고, 장 전체의 분위기와 메시지도 함께 설명한다.

2. 본문 흐름 설명
- 장 전체의 흐름을 3~4문단으로 나누어 설명한다.
- 사건, 시적 표현, 권면, 약속 등 본문 성격에 맞게 쉽게 풀어쓴다.

3. 중요한 구절 2~3개
- 중요한 구절을 2~3개 고르고, 각 구절이 왜 중요한지 2문장 이상 설명한다.

4. 적용 질문 2개
- 오늘의 삶에 연결할 수 있는 질문으로 작성한다.
"""

        try:
            llm_response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            llm_answer = llm_response.choices[0].message.content

        except Exception as e:
            llm_answer = f"LLM 답변 생성 중 오류가 발생했습니다: {e}"

        chat_memory.append({
            "question": q0,
            "topic": f"{found_book} {chapter}장",
            "summary": llm_answer[:250],
            "evidence": evidence_text,
            "evidence_debug": evidence_text
        })

        chat_memory_store[session_id] = chat_memory[-MAX_MEMORY:]

        return JSONResponse({
            "question_type": "chapter_explanation",
            "answer_text": llm_answer + "\n\n---\n[검색 근거]\n" + evidence_text
        })

    # --------------------------------------------------
    # 15. 후속 질문 처리
    # --------------------------------------------------
    if is_followup and len(chat_memory) > 0 and question_type == "explanation":
        last_item = chat_memory[-1]
        previous_evidence = str(last_item.get("evidence", "")).strip()
        previous_topic = str(last_item.get("topic", "")).strip()

        if previous_evidence != "":
            system_prompt = """
너는 성경공부를 돕는 조심스러운 AI 도우미다.
반드시 제공된 이전 성경 근거 안에서만 답변한다.
근거에 없는 내용은 단정하지 않는다.
사용자의 질문이 이전 대화의 후속 질문이면, 이전 주제와 본문을 기준으로 자연스럽게 이어서 설명한다.
특정 교단의 교리 논쟁은 단정하지 말고 본문 중심으로 설명한다.
오탈자 없이 자연스러운 한국어로 답변한다.
답변은 한국어로 한다.
"""

            user_prompt = f"""
최근 대화 맥락:
{recent_memory_text}

이전 주제:
{previous_topic}

사용자 후속 질문:
{q0}

이전 검색 근거:
{previous_evidence}

답변 형식:
1. 핵심 답변
- 이전 주제와 사용자의 후속 질문을 연결해서 4~6문장으로 답한다.
- 앞선 답변을 반복하기보다, 사용자가 추가로 궁금해한 부분을 중심으로 설명한다.

2. 본문 근거
- 이전 검색 근거 중 관련 구절 2~3개를 고른다.
- 각 구절이 후속 질문과 어떻게 연결되는지 2문장 이상 설명한다.

3. 쉬운 설명
- 성경공부 초보자도 이해할 수 있도록 3~4문단으로 쉽게 풀어쓴다.
- 필요하면 비유, 관계, 상황을 나누어 설명한다.
- 근거에 없는 내용은 단정하지 않는다.

4. 적용 질문 2개
- 이전 주제를 실제 삶에 연결할 수 있는 질문으로 작성한다.
"""

            try:
                llm_response = openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                )
                llm_answer = llm_response.choices[0].message.content

            except Exception as e:
                llm_answer = f"LLM 답변 생성 중 오류가 발생했습니다: {e}"

            chat_memory.append({
                "question": q0,
                "topic": previous_topic,
                "summary": llm_answer[:250],
                "evidence": previous_evidence,
                "evidence_debug": last_item.get("evidence_debug", "")
            })

            chat_memory_store[session_id] = chat_memory[-MAX_MEMORY:]

            return JSONResponse({
                "question_type": "followup_explanation",
                "answer_text": llm_answer + "\n\n---\n[이전 검색 근거]\n" + previous_evidence
            })

    # --------------------------------------------------
    # 16. 비유/사건 직접 매핑
    # --------------------------------------------------
    matched_story = None

    for story_key, story_info in story_map.items():
        if story_key in retrieval_query_norm:
            matched_story = story_info
            break

    if matched_story is not None:
        story_df = bible_verses[
            (bible_verses["book_kor"] == matched_story["book"]) &
            (bible_verses["chapter"] == matched_story["chapter"]) &
            (bible_verses["verse"] >= matched_story["start_verse"]) &
            (bible_verses["verse"] <= matched_story["end_verse"])
        ].copy().sort_values("verse")

        evidence_lines = []
        for _, r in story_df.iterrows():
            evidence_lines.append(
                f"{r['book_kor']} {int(r['chapter'])}:{int(r['verse'])} {r['text']}"
            )

        evidence_text = "\n".join(evidence_lines)

        system_prompt = """
너는 성경공부를 돕는 조심스러운 AI 도우미다.
반드시 제공된 성경 근거 안에서만 답변한다.
근거에 없는 내용은 단정하지 않는다.
오탈자 없이 자연스러운 한국어로 답변한다.
답변은 한국어로 한다.
"""

        user_prompt = f"""
최근 대화 맥락:
{recent_memory_text}

사용자 질문:
{q0}

성경 근거:
{evidence_text}

답변 형식:
1. 핵심 답변
- 이 사건 또는 비유가 말하는 중심 메시지를 4~6문장으로 설명한다.
- 등장인물, 상황, 결론을 간단히 연결해서 설명한다.

2. 본문 흐름 설명
- 본문의 시작, 전개, 결말을 3~4문단으로 나누어 설명한다.
- 각 인물의 행동과 그 의미를 본문 근거 안에서 설명한다.

3. 중요한 구절 2~3개
- 중요한 구절을 2~3개 선택한다.
- 각 구절마다 왜 핵심인지 2문장 이상 설명한다.

4. 적용 질문 2개
- 오늘의 신앙생활이나 관계에 연결할 수 있는 질문으로 작성한다.
"""

        try:
            llm_response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            llm_answer = llm_response.choices[0].message.content

        except Exception as e:
            llm_answer = f"LLM 답변 생성 중 오류가 발생했습니다: {e}"

        chat_memory.append({
            "question": q0,
            "topic": matched_story["title"],
            "summary": llm_answer[:250],
            "evidence": evidence_text,
            "evidence_debug": evidence_text
        })

        chat_memory_store[session_id] = chat_memory[-MAX_MEMORY:]

        return JSONResponse({
            "question_type": "story_explanation",
            "answer_text": llm_answer + "\n\n---\n[검색 근거]\n" + evidence_text
        })

    # --------------------------------------------------
    # 17. 일반 주제검색 / 일반 설명형 질문
    # --------------------------------------------------
    if question_type == "topic_search" or question_type == "explanation":
        found_topics = []

        for topic_name, keywords in topic_dict.items():
            if topic_name in retrieval_query:
                found_topics.append(topic_name)
                continue

            for kw in keywords:
                if kw in retrieval_query:
                    found_topics.append(topic_name)
                    break

        found_topics = list(dict.fromkeys(found_topics))

        search_keywords = []
        for t in found_topics:
            search_keywords.extend(topic_dict[t])

        search_keywords = list(dict.fromkeys(search_keywords))

        expanded_terms = []
        for t in found_topics:
            expanded_terms.extend(query_expand_dict.get(t, []))

        expanded_terms = list(dict.fromkeys(expanded_terms))
        expanded_query = retrieval_query + " " + " ".join(expanded_terms)

        query_embedding = embedding_model.encode(
            ["query: " + expanded_query],
            normalize_embeddings=True
        ).tolist()

        vector_results = vector_collection.query(
            query_embeddings=query_embedding,
            n_results=100
        )

        vector_rows = []

        ids = vector_results["ids"][0]
        docs = vector_results["documents"][0]
        metas = vector_results["metadatas"][0]
        distances = vector_results["distances"][0]

        for i in range(len(ids)):
            vector_score = 1 / (1 + float(distances[i]))

            vector_rows.append({
                "chunk_id": ids[i],
                "book_kor": metas[i]["book_kor"],
                "chapter": int(metas[i]["chapter"]),
                "text_chunk": docs[i],
                "vector_score": vector_score,
                "keyword_score": 0.0
            })

        vector_df = pd.DataFrame(vector_rows)

        keyword_df = bible_chunks.copy()
        keyword_df["keyword_score"] = 0.0

        for kw in search_keywords:
            keyword_df["keyword_score"] += keyword_df["text_chunk"].astype(str).str.count(re.escape(kw))

        keyword_df = keyword_df[keyword_df["keyword_score"] > 0].copy()

        if len(keyword_df) > 0:
            keyword_df = keyword_df.sort_values("keyword_score", ascending=False).head(50)
            keyword_df = keyword_df[[
                "chunk_id",
                "book_kor",
                "chapter",
                "text_chunk",
                "keyword_score"
            ]].copy()
            keyword_df["vector_score"] = 0.0
        else:
            keyword_df = pd.DataFrame(columns=[
                "chunk_id", "book_kor", "chapter", "text_chunk",
                "vector_score", "keyword_score"
            ])

        candidate_df = pd.concat([vector_df, keyword_df], ignore_index=True)

        if len(candidate_df) == 0:
            return JSONResponse({
                "question_type": question_type,
                "answer_text": "관련 성경 근거를 찾지 못했습니다."
            })

        candidate_df = (
            candidate_df
            .groupby(["chunk_id", "book_kor", "chapter", "text_chunk"], as_index=False)
            .agg({
                "vector_score": "max",
                "keyword_score": "max"
            })
        )

        max_kw = candidate_df["keyword_score"].max()

        if max_kw > 0:
            candidate_df["keyword_score_norm"] = candidate_df["keyword_score"] / max_kw
        else:
            candidate_df["keyword_score_norm"] = 0.0

        candidate_df["boost_score"] = 0.0

        for t in found_topics:
            boost_targets = representative_boost.get(t, [])

            for book_name, chapter_num in boost_targets:
                mask = (
                    (candidate_df["book_kor"] == book_name) &
                    (candidate_df["chapter"].astype(int) == int(chapter_num))
                )
                candidate_df.loc[mask, "boost_score"] += 0.2

        candidate_df["final_score"] = (
            candidate_df["vector_score"] * 0.4 +
            candidate_df["keyword_score_norm"] * 0.4 +
            candidate_df["boost_score"] * 0.2
        )

        rerank_candidate_df = candidate_df.sort_values("final_score", ascending=False).head(20).copy()

        rerank_pairs = []
        for _, r in rerank_candidate_df.iterrows():
            rerank_pairs.append([retrieval_query, str(r["text_chunk"])])

        if len(rerank_pairs) > 0:
            rerank_scores = reranker_model.predict(rerank_pairs)
            rerank_candidate_df["rerank_score"] = rerank_scores

            min_score = rerank_candidate_df["rerank_score"].min()
            max_score = rerank_candidate_df["rerank_score"].max()

            if max_score > min_score:
                rerank_candidate_df["rerank_score_norm"] = (
                    (rerank_candidate_df["rerank_score"] - min_score) / (max_score - min_score)
                )
            else:
                rerank_candidate_df["rerank_score_norm"] = 0.0

            rerank_candidate_df["final_score"] = (
                rerank_candidate_df["final_score"] * 0.5 +
                rerank_candidate_df["rerank_score_norm"] * 0.5
            )

            candidate_df = rerank_candidate_df.sort_values("final_score", ascending=False).head(5)
        else:
            candidate_df = candidate_df.sort_values("final_score", ascending=False).head(5)

        evidence_lines = []
        for _, r in candidate_df.iterrows():
            evidence_lines.append(
                f"- {r['book_kor']} {int(r['chapter'])}장: {r['text_chunk']}"
            )

        evidence_text = "\n".join(evidence_lines)

        lines = []
        lines.append("[하이브리드 주제검색 결과]")
        lines.append(f"인식 주제: {found_topics}")
        lines.append(f"검색 키워드: {search_keywords}")
        lines.append("")

        for _, r in candidate_df.iterrows():
            lines.append(
                f"- {r['book_kor']} {int(r['chapter'])}장 "
                f"| score={r['final_score']:.3f} "
                f"| keyword={int(r['keyword_score'])} "
                f"| boost={r['boost_score']:.1f}"
            )
            lines.append(f"  {r['text_chunk']}")
            lines.append("")

        system_prompt = """
너는 성경공부를 돕는 조심스러운 AI 도우미다.
반드시 제공된 성경 근거 안에서만 답변한다.
검색 근거가 질문과 관련 없어 보이면, 억지로 답하지 말고 '제공된 구절만으로는 정확히 답하기 어렵습니다'라고 말한다.
특정 교단의 교리 논쟁은 단정하지 말고 본문 중심으로 설명한다.
오탈자 없이 자연스러운 한국어로 답변한다.
인물명과 성경 용어의 오탈자를 특히 주의한다.
예: 맏아들, 둘째 아들, 아버지, 탕자.
답변을 생성한 뒤 어색한 표현이나 오탈자가 없는지 한 번 점검한 후 출력한다.
답변은 한국어로 한다.
"""

        user_prompt = f"""
최근 대화 맥락:
{recent_memory_text}

사용자 질문:
{q0}

검색된 성경 근거:
{evidence_text}

답변 형식:
1. 핵심 답변
- 질문에 대해 4~6문장으로 자연스럽게 답한다.
- 너무 짧게 요약하지 말고, 사용자가 성경공부를 하는 상황이라고 생각하고 설명한다.

2. 관련 구절 설명
- 핵심 구절 2~3개를 선택한다.
- 각 구절마다 2~3문장으로 왜 중요한지 설명한다.

3. 본문 설명
- 제공된 성경 근거를 바탕으로 3~4문단으로 설명한다.
- 본문의 흐름, 핵심 의미, 오늘날 적용을 나누어 설명한다.
- 근거에 없는 내용은 단정하지 않는다.

4. 적용 질문 2개
- 단순 질문이 아니라 실제 삶을 돌아볼 수 있는 질문으로 작성한다.
- 전체 답변은 너무 짧게 끝내지 말고, 성경공부 모임에서 설명하듯 충분히 풀어서 작성한다.
"""

        try:
            llm_response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            llm_answer = llm_response.choices[0].message.content

        except Exception as e:
            llm_answer = f"LLM 답변 생성 중 오류가 발생했습니다: {e}"

        topic_label = ", ".join(found_topics) if len(found_topics) > 0 else q0

        chat_memory.append({
            "question": q0,
            "topic": topic_label,
            "summary": llm_answer[:250],
            "evidence": evidence_text,
            "evidence_debug": "\n".join(lines)
        })

        chat_memory_store[session_id] = chat_memory[-MAX_MEMORY:]

        return JSONResponse({
            "question_type": question_type,
            "answer_text": llm_answer + "\n\n---\n[검색 근거]\n" + "\n".join(lines)
        })

    # --------------------------------------------------
    # 18. 최종 fallback
    # --------------------------------------------------
    return JSONResponse({
        "question_type": question_type,
        "answer_text": "질문 유형을 아직 인식하지 못했습니다."
    })