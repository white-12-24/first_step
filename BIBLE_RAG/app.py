# app.py
# --------------------------------------------------
# 성경공부 도우미 FastAPI 프로토타입 백엔드
# --------------------------------------------------

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
import re

app = FastAPI()

# --------------------------------------------------
# 1. 기본 경로 설정
# --------------------------------------------------
BASE_DIR = r"C:\py_temp\new_proj\BIBLE_RAG"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
ENV_PATH = os.path.join(BASE_DIR, ".env")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# --------------------------------------------------
# 2. .env 로드 / OpenAI 설정
# --------------------------------------------------
load_dotenv(dotenv_path=ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if OPENAI_API_KEY is None or OPENAI_API_KEY.strip() == "":
    print("[경고] OPENAI_API_KEY를 찾지 못했습니다. .env 파일을 확인하세요.")
    openai_client = None
else:
    OPENAI_API_KEY = OPENAI_API_KEY.strip().replace('"', '').replace("'", "")
    print("[확인] OPENAI_API_KEY 로드됨:", OPENAI_API_KEY[:10] + "..." + OPENAI_API_KEY[-4:])
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

print("[확인] OPENAI_MODEL:", OPENAI_MODEL)

# --------------------------------------------------
# 3. CSV 로드
# --------------------------------------------------
bible_verses = pd.read_csv(os.path.join(PROCESSED_DIR, "bible_verses.csv"))
bible_chunks = pd.read_csv(os.path.join(PROCESSED_DIR, "bible_chunks.csv"))

# --------------------------------------------------
# 4. 벡터DB / 임베딩 모델 로드
# --------------------------------------------------
embedding_model_name = "intfloat/multilingual-e5-small"
embedding_model = SentenceTransformer(embedding_model_name)

chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
vector_collection = chroma_client.get_collection(name="bible_chunks")

print("[확인] vector collection count:", vector_collection.count())

# --------------------------------------------------
# 5. 책 이름 별칭표
# --------------------------------------------------
book_alias_rows = [
    ["창세기", "창세기"], ["창", "창세기"],
    ["출애굽기", "출애굽기"], ["출", "출애굽기"],
    ["레위기", "레위기"], ["레", "레위기"],
    ["민수기", "민수기"], ["민", "민수기"],
    ["신명기", "신명기"], ["신", "신명기"],
    ["여호수아", "여호수아"], ["수", "여호수아"],
    ["사사기", "사사기"], ["삿", "사사기"],
    ["룻기", "룻기"], ["룻", "룻기"],
    ["사무엘상", "사무엘상"], ["삼상", "사무엘상"],
    ["사무엘하", "사무엘하"], ["삼하", "사무엘하"],
    ["열왕기상", "열왕기상"], ["왕상", "열왕기상"],
    ["열왕기하", "열왕기하"], ["왕하", "열왕기하"],
    ["역대상", "역대상"], ["대상", "역대상"],
    ["역대하", "역대하"], ["대하", "역대하"],
    ["에스라", "에스라"], ["스", "에스라"],
    ["느헤미야", "느헤미야"], ["느", "느헤미야"],
    ["에스더", "에스더"], ["에", "에스더"],
    ["욥기", "욥기"], ["욥", "욥기"],
    ["시편", "시편"], ["시", "시편"],
    ["잠언", "잠언"], ["잠", "잠언"],
    ["전도서", "전도서"], ["전", "전도서"],
    ["아가", "아가"],
    ["이사야", "이사야"], ["사", "이사야"],
    ["예레미야", "예레미야"], ["렘", "예레미야"],
    ["예레미야 애가", "예레미야 애가"], ["애", "예레미야 애가"],
    ["에스겔", "에스겔"], ["겔", "에스겔"],
    ["다니엘", "다니엘"], ["단", "다니엘"],
    ["호세아", "호세아"], ["호", "호세아"],
    ["요엘", "요엘"], ["욜", "요엘"],
    ["아모스", "아모스"], ["암", "아모스"],
    ["오바댜", "오바댜"], ["옵", "오바댜"],
    ["요나", "요나"], ["욘", "요나"],
    ["미가", "미가"], ["미", "미가"],
    ["나훔", "나훔"], ["나", "나훔"],
    ["하박국", "하박국"], ["합", "하박국"],
    ["스바냐", "스바냐"], ["습", "스바냐"],
    ["학개", "학개"], ["학", "학개"],
    ["스가랴", "스가랴"], ["슥", "스가랴"],
    ["말라기", "말라기"], ["말", "말라기"],
    ["마태복음", "마태복음"], ["마", "마태복음"],
    ["마가복음", "마가복음"], ["막", "마가복음"],
    ["누가복음", "누가복음"], ["눅", "누가복음"],
    ["요한복음", "요한복음"], ["요", "요한복음"],
    ["사도행전", "사도행전"], ["행", "사도행전"],
    ["로마서", "로마서"], ["롬", "로마서"],
    ["고린도전서", "고린도전서"], ["고전", "고린도전서"],
    ["고린도후서", "고린도후서"], ["고후", "고린도후서"],
    ["갈라디아서", "갈라디아서"], ["갈", "갈라디아서"],
    ["에베소서", "에베소서"], ["엡", "에베소서"],
    ["빌립보서", "빌립보서"], ["빌", "빌립보서"],
    ["골로새서", "골로새서"], ["골", "골로새서"],
    ["데살로니가전서", "데살로니가전서"], ["살전", "데살로니가전서"],
    ["데살로니가후서", "데살로니가후서"], ["살후", "데살로니가후서"],
    ["디모데전서", "디모데전서"], ["딤전", "디모데전서"],
    ["디모데후서", "디모데후서"], ["딤후", "디모데후서"],
    ["디도서", "디도서"], ["딛", "디도서"],
    ["빌레몬서", "빌레몬서"], ["몬", "빌레몬서"],
    ["히브리서", "히브리서"], ["히", "히브리서"],
    ["야고보서", "야고보서"], ["약", "야고보서"],
    ["베드로전서", "베드로전서"], ["벧전", "베드로전서"],
    ["베드로후서", "베드로후서"], ["벧후", "베드로후서"],
    ["요한1서", "요한1서"], ["요일", "요한1서"],
    ["요한2서", "요한2서"], ["요이", "요한2서"],
    ["요한3서", "요한3서"], ["요삼", "요한3서"],
    ["유다서", "유다서"], ["유", "유다서"],
    ["요한계시록", "요한계시록"], ["계", "요한계시록"],
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
    "위로": ["위로", "평안", "안위", "두려워말라", "근심", "염려"],
    "기도": ["기도", "간구", "구하라", "부르짖", "기도하라"],
    "구원": ["구원", "영생", "멸망치", "구속", "복음"],
    "회개": ["회개", "돌이키", "죄인", "죄를", "회개하라"],
    "순종": ["순종", "지키", "계명", "듣고", "행하라"],
    "지혜": ["지혜", "명철", "훈계", "슬기", "깨닫"],
    "불안": ["두려워 말라", "염려하지 말라", "평안", "강하고 담대하라", "근심"],
    "고난": ["고난", "환난", "시험", "핍박", "눈물"],
}

query_expand_dict = {
    "불안": ["두려워 말라", "염려하지 말라", "평안을 너희에게 주노라", "강하고 담대하라", "근심하지 말라"],
    "위로": ["위로", "평안", "소망", "환난 중 위로", "두려워 말라"],
    "사랑": ["하나님의 사랑", "서로 사랑하라", "사랑은 하나님께 속한 것", "독생자를 주신 사랑"],
    "기도": ["기도하라", "간구", "부르짖음", "구하라", "의인의 간구"],
    "용서": ["용서하라", "죄 사함", "서로 용서", "긍휼", "사하여 주옵소서"],
    "고난": ["고난", "환난", "시험", "위로", "인내"],
}

representative_boost = {
    "불안": [("마태복음", 6), ("빌립보서", 4), ("요한복음", 14), ("이사야", 41), ("시편", 23)],
    "사랑": [("요한복음", 3), ("요한1서", 4), ("고린도전서", 13), ("로마서", 5)],
    "기도": [("마태복음", 6), ("야고보서", 5), ("빌립보서", 4)],
    "용서": [("마태복음", 6), ("골로새서", 3), ("누가복음", 23)],
    "위로": [("고린도후서", 1), ("이사야", 41), ("시편", 23), ("요한복음", 14)],
    "고난": [("고린도후서", 1), ("로마서", 5), ("야고보서", 1), ("베드로전서", 4)],
}

# --------------------------------------------------
# 7. 메인 페이지
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --------------------------------------------------
# 8. 질문 처리
# --------------------------------------------------
@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    q0 = str(body.get("question", "")).strip()
    q1 = q0.lower().replace(" ", "")

    question_type = "unknown"

    if re.search(r"[0-9]+장[0-9]+절", q1) or re.search(r"[0-9]+:[0-9]+", q1):
        question_type = "verse_lookup"
    elif re.search(r"[0-9]+장", q1) or re.search(r"[0-9]+편", q1):
        question_type = "chapter_lookup"
    elif ("말씀" in q0) or ("구절" in q0) or ("추천" in q0) or ("알려" in q0):
        question_type = "topic_search"
    elif ("왜" in q0) or ("무엇" in q0) or ("설명" in q0) or ("의미" in q0):
        question_type = "explanation"

    found_book = None
    found_alias = None

    for i in range(len(book_alias_df)):
        alias_norm = book_alias_df.loc[i, "alias_norm"]
        book_kor = book_alias_df.loc[i, "book_kor"]

        if alias_norm in q1:
            if found_alias is None or len(alias_norm) > len(found_alias):
                found_alias = alias_norm
                found_book = book_kor

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

    # 절 직접조회
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

    # 장 조회
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

    # 주제검색
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

        expanded_terms = []

        for t in found_topics:
            expanded_terms.extend(query_expand_dict.get(t, []))

        expanded_terms = list(dict.fromkeys(expanded_terms))
        expanded_query = q0 + " " + " ".join(expanded_terms)

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
            keyword_df = keyword_df[["chunk_id", "book_kor", "chapter", "text_chunk", "keyword_score"]].copy()
            keyword_df["vector_score"] = 0.0
        else:
            keyword_df = pd.DataFrame(columns=[
                "chunk_id", "book_kor", "chapter", "text_chunk", "vector_score", "keyword_score"
            ])

        candidate_df = pd.concat([vector_df, keyword_df], ignore_index=True)

        if len(candidate_df) == 0:
            return JSONResponse({
                "question_type": question_type,
                "answer_text": "관련 주제의 구절을 찾지 못했습니다."
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

        candidate_df = candidate_df.sort_values("final_score", ascending=False).head(5)

        evidence_lines = []

        for _, r in candidate_df.iterrows():
            evidence_lines.append(
                f"- {r['book_kor']} {int(r['chapter'])}장: {r['text_chunk']}"
            )

        evidence_text = "\n".join(evidence_lines)

        search_lines = []
        search_lines.append("[하이브리드 주제검색 결과]")
        search_lines.append(f"인식 주제: {found_topics}")
        search_lines.append(f"검색 키워드: {search_keywords}")
        search_lines.append("")

        for _, r in candidate_df.iterrows():
            search_lines.append(
                f"- {r['book_kor']} {int(r['chapter'])}장 "
                f"| score={r['final_score']:.3f} "
                f"| keyword={int(r['keyword_score'])} "
                f"| boost={r['boost_score']:.1f}"
            )
            search_lines.append(f"  {r['text_chunk']}")
            search_lines.append("")

        system_prompt = """
너는 성경공부를 돕는 조심스러운 AI 도우미다.
반드시 제공된 성경 근거 안에서만 답변한다.
근거에 없는 내용은 단정하지 말고 '제공된 구절만으로는 확정하기 어렵습니다'라고 말한다.
특정 교단의 교리 논쟁은 단정하지 말고 본문 중심으로 설명한다.
답변은 한국어로 한다.
"""

        user_prompt = f"""
사용자 질문:
{q0}

검색된 성경 근거:
{evidence_text}

답변 형식:
1. 핵심 답변
2. 관련 구절
3. 짧은 설명
4. 묵상 질문 2개
"""

        if openai_client is None:
            llm_answer = (
                "LLM 답변을 생성하지 못했습니다.\n"
                "원인: OPENAI_API_KEY가 로드되지 않았습니다.\n"
                ".env 파일의 OPENAI_API_KEY 값을 확인하세요."
            )
        else:
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
                llm_answer = (
                    "LLM 답변 생성 중 오류가 발생했습니다.\n\n"
                    f"{e}\n\n"
                    "대부분의 경우 .env의 OPENAI_API_KEY가 잘못되었거나, "
                    "키가 비활성화되었거나, 프로젝트/결제 설정 문제가 원인입니다."
                )

        return JSONResponse({
            "question_type": question_type,
            "answer_text": llm_answer + "\n\n---\n[검색 근거]\n" + "\n".join(search_lines)
        })

    if question_type == "explanation":
        return JSONResponse({
            "question_type": question_type,
            "answer_text": "설명형 질문은 다음 단계에서 검색 결과 + 설명 생성으로 연결할 예정입니다."
        })

    return JSONResponse({
        "question_type": question_type,
        "answer_text": "질문 유형을 아직 인식하지 못했습니다."
    })