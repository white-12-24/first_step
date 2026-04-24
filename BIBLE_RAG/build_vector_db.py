# build_vector_db.py
# --------------------------------------------------
# 이 코드는 bible_chunks.csv를 읽어서
# 성경 청크를 임베딩한 뒤 ChromaDB 벡터DB에 저장하는 코드입니다.
#
# 역할:
# 1) bible_chunks.csv 로드
# 2) 각 청크 text_chunk를 임베딩
# 3) ChromaDB에 chunk_id, 본문, 메타데이터 저장
# --------------------------------------------------

import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# 1. 경로 설정
# --------------------------------------------------
BASE_DIR = r"C:\py_temp\new_proj\BIBLE_RAG"
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "processed", "bible_chunks.csv")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")

# --------------------------------------------------
# 2. 청크 데이터 로드
# --------------------------------------------------
chunks_df = pd.read_csv(CHUNKS_PATH)

print("청크 개수:", len(chunks_df))
print(chunks_df.head())

# --------------------------------------------------
# 3. 임베딩 모델 로드
# --------------------------------------------------
# 최초 실행 시 모델 다운로드가 발생할 수 있음
# 한국어도 어느 정도 처리 가능한 다국어 임베딩 모델
# --------------------------------------------------
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

# --------------------------------------------------
# 4. ChromaDB 준비
# --------------------------------------------------
client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

# 기존 collection이 있으면 삭제 후 다시 생성
# 데이터가 바뀌었을 때 중복 저장 방지 목적
collection_name = "bible_chunks"

try:
    client.delete_collection(collection_name)
    print("기존 collection 삭제 완료")
except Exception:
    print("기존 collection 없음")

collection = client.get_or_create_collection(name=collection_name)

# --------------------------------------------------
# 5. 청크 텍스트 / 메타데이터 준비
# --------------------------------------------------
ids = []
documents = []
metadatas = []

for i in range(len(chunks_df)):
    row = chunks_df.iloc[i]

    chunk_id = str(row["chunk_id"])
    text_chunk = str(row["text_chunk"])

    ids.append(chunk_id)
    documents.append(text_chunk)

    metadatas.append({
        "book_kor": str(row["book_kor"]),
        "chapter": int(row["chapter"]),
        "start_verse_id": str(row["start_verse_id"]),
        "end_verse_id": str(row["end_verse_id"])
    })

# --------------------------------------------------
# 6. 배치 단위로 임베딩 생성 후 ChromaDB 저장
# --------------------------------------------------
batch_size = 256

for start in range(0, len(documents), batch_size):
    end = start + batch_size

    batch_ids = ids[start:end]
    batch_docs = documents[start:end]
    batch_meta = metadatas[start:end]

    print(f"임베딩/저장 중: {start} ~ {min(end, len(documents))}")

    embeddings = model.encode(
        batch_docs,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    ).tolist()

    collection.add(
        ids=batch_ids,
        documents=batch_docs,
        metadatas=batch_meta,
        embeddings=embeddings
    )

print("벡터DB 생성 완료")
print("저장 위치:", VECTOR_DB_DIR)
print("collection count:", collection.count())