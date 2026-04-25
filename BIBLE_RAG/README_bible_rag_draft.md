# 성경공부 도우미 RAG 챗봇

## 프로젝트 개요
성경 원문 데이터를 기반으로 사용자의 성경 질문에 대해 관련 구절을 검색하고, 검색 근거를 바탕으로 LLM이 설명형 답변을 생성하는 RAG 챗봇입니다.

## 핵심 기능
- 절 직접조회: `요3:16`
- 장 조회/요약: `시편 23편 요약해줘`
- 주제 검색: 사랑, 용서, 불안, 우울, 분노, 재물, 가족 등
- 비유/사건 직접 매핑: 탕자의 비유, 주기도문, 십계명, 다윗과 골리앗 등
- 대화형 기억: session_id별 최근 대화와 검색 근거 저장
- 근거 접기 UI: 답변과 검색 근거 분리
- LLM intent classifier: 질문 의도 분석 후 검색 방향 결정
- 대표 후보군 안전장치: 감정/생활 주제에서 대표 본문을 후보에 강제 추가

## RAG 파이프라인
1. 사용자 질문 입력
2. LLM intent classifier로 route/topics/story/followup/search_query 분석
3. 직접조회 / story_map / topic_search / followup 중 경로 결정
4. Vector search + keyword search + priority candidate 결합
5. Reranker로 후보 재정렬
6. LLM이 제공된 근거 안에서 답변 생성
7. 답변과 검색 근거를 UI에 분리 표시

## 평가 방법
`bible_rag_eval_questions.csv`에 테스트 질문과 기대 근거를 정의하고, `run_evaluation.py`로 로컬 FastAPI 서버에 자동 요청을 보냅니다.

실행:
```bash
uvicorn app:app --port 8000
python run_evaluation.py
```

결과:
- `evaluation_results.csv` 생성
- 질문 유형, 검색 근거 hit, 답변 preview, verdict 저장

## 향후 개선
- 실패 케이스를 기반으로 topic_dict/story_map/priority_reference_map 보강
- 사용자 로그 기반 질문 유형 분석
- 평가 자동화 점수 고도화
- UI 섹션 카드화
- 배포 시 개인정보 보호 및 로그 익명화
