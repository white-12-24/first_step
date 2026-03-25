실행 전 같은 폴더에 아래 파일을 넣어주세요.

1) sinkhole_logi_model.pkl
2) final_df_4326.geojson   (권장)
   또는 final_df.gpkg + 레이어명 final_df
3) 싱크홀발생건수_위도경도_최종.xlsx   (선택)
4) full_df_1210-1.xlsx                  (선택, 기본 시작용)

핵심 구조
- 공간 뼈대: final_df_4326.geojson
- 예측 엔진: sinkhole_logi_model.pkl
- 최신 입력 데이터: 업로드한 df 파일
- merge 기준: id

실행
1. pip install -r requirements.txt
2. uvicorn app:app --reload
3. 브라우저에서 http://127.0.0.1:8000 접속
