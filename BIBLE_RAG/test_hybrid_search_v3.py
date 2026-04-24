# test_hybrid_search_v3.py
# --------------------------------------------------
# 목적:
# 저장된 ChromaDB 벡터DB는 그대로 사용하고,
# 검색 방식만 개선해서 테스트하는 코드
#
# 개선점:
# 1) vector distance를 양수 점수로 변환
# 2) 벡터검색 후보 + 키워드검색 후보를 합쳐서 검색
# 3) keyword_score가 높은 구절도 후보에서 빠지지 않게 처리
# --------------------------------------------------

import os
import re
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# 1. 경로 설정
# --------------------------------------------------
BASE_DIR = r"C:\py_temp\new_proj\BIBLE_RAG"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")

chunks_df = pd.read_csv(os.path.join(PROCESSED_DIR, "bible_chunks.csv"))

# --------------------------------------------------
# 2. 모델 / 벡터DB 로드
# --------------------------------------------------
model_name = "intfloat/multilingual-e5-small"
model = SentenceTransformer(model_name)

client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_collection(name="bible_chunks")

print("collection count:", collection.count())

# --------------------------------------------------
# 3. 주제 사전
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
}

# --------------------------------------------------
# 4. 테스트 질문
# --------------------------------------------------
test_queries = [
    "불안할 때 읽을 말씀",
    "하나님의 사랑에 대한 구절",
    "기도에 대한 말씀",
    "용서에 대한 성경구절",
    "고난 중에 위로가 되는 말씀",
]

# --------------------------------------------------
# 5. 하이브리드 검색 테스트
# --------------------------------------------------
for q in test_queries:
    print("\n" + "=" * 100)
    print("질문:", q)

    # --------------------------------------------------
    # 5-1. 벡터검색 후보 Top50
    # --------------------------------------------------
    query_embedding = model.encode(
        ["query: " + q],
        normalize_embeddings=True
    ).tolist()

    vector_results = collection.query(
        query_embeddings=query_embedding,
        n_results=50
    )

    vector_rows = []

    ids = vector_results["ids"][0]
    docs = vector_results["documents"][0]
    metas = vector_results["metadatas"][0]
    distances = vector_results["distances"][0]

    for i in range(len(ids)):
        # distance가 작을수록 좋으므로 1/(1+distance)로 양수 점수 변환
        vector_score = 1 / (1 + float(distances[i]))

        vector_rows.append({
            "chunk_id": ids[i],
            "book_kor": metas[i]["book_kor"],
            "chapter": int(metas[i]["chapter"]),
            "text_chunk": docs[i],
            "vector_score": vector_score,
            "keyword_score": 0.0,
            "source_type": "vector"
        })

    vector_df = pd.DataFrame(vector_rows)

    # --------------------------------------------------
    # 5-2. 질문에서 주제 인식
    # --------------------------------------------------
    found_topics = []

    for topic_name, keywords in topic_dict.items():
        if topic_name in q:
            found_topics.append(topic_name)
            continue

        for kw in keywords:
            if kw in q:
                found_topics.append(topic_name)
                break

    found_topics = list(dict.fromkeys(found_topics))

    # --------------------------------------------------
    # 5-3. 주제별 키워드 확장
    # --------------------------------------------------
    search_keywords = []

    for t in found_topics:
        search_keywords.extend(topic_dict[t])

    search_keywords = list(dict.fromkeys(search_keywords))

    # --------------------------------------------------
    # 5-4. 키워드검색 후보 Top50
    # --------------------------------------------------
    keyword_df = chunks_df.copy()
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
        keyword_df["source_type"] = "keyword"
    else:
        keyword_df = pd.DataFrame(columns=[
            "chunk_id", "book_kor", "chapter", "text_chunk",
            "vector_score", "keyword_score", "source_type"
        ])

    # --------------------------------------------------
    # 5-5. 벡터 후보 + 키워드 후보 합치기
    # --------------------------------------------------
    candidate_df = pd.concat([vector_df, keyword_df], ignore_index=True)

    # 같은 chunk가 벡터/키워드 양쪽에서 잡히면 점수 합산
    candidate_df = (
        candidate_df
        .groupby(["chunk_id", "book_kor", "chapter", "text_chunk"], as_index=False)
        .agg({
            "vector_score": "max",
            "keyword_score": "max"
        })
    )

    # --------------------------------------------------
    # 5-6. keyword_score 정규화
    # --------------------------------------------------
    # keyword_score는 0,1,2,3... 식으로 커지기 때문에 0~1 범위로 축소
    max_kw = candidate_df["keyword_score"].max()

    if max_kw > 0:
        candidate_df["keyword_score_norm"] = candidate_df["keyword_score"] / max_kw
    else:
        candidate_df["keyword_score_norm"] = 0.0

    # --------------------------------------------------
    # 5-7. 최종 점수 계산
    # --------------------------------------------------
    # 현재는 키워드 비중을 조금 더 높임
    candidate_df["final_score"] = (
        candidate_df["vector_score"] * 0.4 +
        candidate_df["keyword_score_norm"] * 0.6
    )

    candidate_df = candidate_df.sort_values("final_score", ascending=False).head(5)

    # --------------------------------------------------
    # 5-8. 결과 출력
    # --------------------------------------------------
    print("인식 주제:", found_topics)
    print("검색 키워드:", search_keywords)

    for _, row in candidate_df.iterrows():
        print()
        print(f"- {row['book_kor']} {int(row['chapter'])}장")
        print(
            f"  final_score={row['final_score']:.4f}, "
            f"vector={row['vector_score']:.4f}, "
            f"keyword={row['keyword_score']}, "
            f"keyword_norm={row['keyword_score_norm']:.4f}"
        )
        print(" ", row["text_chunk"])