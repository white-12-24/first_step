# 싱크홀 종합 위험도 대시보드

## 1. 프로젝트 소개
학습된 싱크홀 위험도 예측 모델을 기반으로 grid 단위 위험도를 지도 위에 시각화하는 FastAPI 기반 대시보드입니다.

이 대시보드는 다음 기능을 제공합니다.
- grid 단위 위험확률 계산 및 5단계 위험도 시각화
- 지역 선택 필터
- 위험도 분포 차트
- 지역별 발생건수 TOP 10
- 월별 발생 추이
- 최신 df 파일 업로드 후 id 기준 merge 및 위험도 재계산

## 2. 주요 파일
- `app.py` : FastAPI 백엔드
- `templates/index.html` : 메인 화면
- `static/style.css` : 스타일
- `static/app.js` : 프론트 동작 로직
- `sinkhole_logi_model.pkl` : 학습된 예측 모델
- `final_df_4326.geojson` : 공간 틀(지도용 grid)
- `full_df_1210-1.xlsx` : 기본 속성 데이터
- `싱크홀발생건수_위도경도_최종.xlsx` : 발생 통계 데이터

## 3. 실행 방법
### 패키지 설치
```bash
pip install -r requirements.txt

서버 실행
uvicorn app:app --reload --port 8010

접속 주소
http://127.0.0.1:8010

업로드 파일
확장자: .xlsx, .xls, .csv
필수 컬럼: id
모델 입력 컬럼이 포함되어야 정상 반영 가능
공간 파일
final_df_4326.geojson
id 컬럼 포함
5. 기술 스택
Python
FastAPI
pandas
joblib
Leaflet
Chart.js
6. 위험도 로직
모델은 각 grid의 발생 확률(risk_prob)을 산출
지도는 확률값을 5단계 위험도로 시각화
별도로 threshold 이상 grid는 고위험 grid로 집계
7. 주의사항
업로드 파일에 id가 없으면 반영되지 않음
모델 입력 컬럼이 누락되면 업로드가 실패할 수 있음
실행 전 필수 데이터 파일이 프로젝트 루트에 있어야 함