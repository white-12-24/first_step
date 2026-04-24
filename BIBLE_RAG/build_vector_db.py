# build_vector_db.py
# --------------------------------------------------
# 목적:
# bible_chunks.csv를 읽어서 검색용 임베딩을 만들고
# ChromaDB에 저장하는 코드
#
# 개선점:
# - multilingual-e5-small 모델 사용
# - passage: 접두어 사용
# - cosine 거리 사용
# --------------------------------------------------

import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = r"C:\py_temp\new_proj\BIBLE_RAG"
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "processed", "bible_chunks.csv")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")

chunks_df = pd.read_csv(CHUNKS_PATH)

print("청크 개수:", len(chunks_df))

model_name = "intfloat/multilingual-e5-small"
model = SentenceTransformer(model_name)

client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

collection_name = "bible_chunks"

try:
    client.delete_collection(collection_name)
    print("기존 collection 삭제 완료")
except Exception:
    print("기존 collection 없음")

collection = client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)

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

batch_size = 256

for start in range(0, len(documents), batch_size):
    end = start + batch_size

    batch_ids = ids[start:end]
    batch_docs = documents[start:end]
    batch_meta = metadatas[start:end]

    print(f"임베딩/저장 중: {start} ~ {min(end, len(documents))}")

    # e5 계열은 passage: 접두어 권장
    embed_docs = ["passage: " + doc for doc in batch_docs]

    embeddings = model.encode(
        embed_docs,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=batch_ids,
        documents=batch_docs,
        metadatas=batch_meta,
        embeddings=embeddings
    )
print("embedding sample:", embeddings[0][:5])

print("벡터DB 생성 완료")
print("collection count:", collection.count())