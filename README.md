# 한국어 서술형 글쓰기 자동 채점 (AES)

Term Project #1. AI Hub 「서술형 글쓰기 평가 데이터」(dataSetSn=71818)로 종합 점수(1~4)를 예측하고 QWK로 비교한다.

| 모델 | QWK(채점자 평균) |
|---|---|
| 표면 특징 + 문자 TF-IDF 릿지 (1세대) | 0.457 |
| KLUE-RoBERTa 다중 과제 회귀 (2세대) | 0.538 |
| 사람-사람 (상한) | 0.693 |

보고서: `1102543003.pdf`

## 데이터
원본 데이터는 AI Hub 이용 신청 후 내려받아야 하며 이 저장소에 포함하지 않는다. `extract.py`의 `base` 경로를 내려받은 폴더(`3.개방데이터/1.데이터/`)로 맞춘다.

## 실행 순서
```bash
pip install "numpy<2" "transformers==4.44.2" pandas scikit-learn scipy matplotlib torch
python extract.py                                   # 라벨링 zip -> essays.csv (36,006편)
python baseline.py                                  # 1세대 기준선 -> res_baseline.json, pred_ridge.npy
python bert.py klue/roberta-base klue_rb 3          # 2세대, RTX 3060 Ti 기준 약 25분
python analysis.py                                  # 표, 그림 생성
```

## 파일
- `common.py`: QWK, 평가 함수, 데이터 로드
- `extract.py`: JSON 파싱, `essays.csv` 생성
- `baseline.py`: 표면 특징 / TF-IDF 릿지
- `bert.py`: [지시문, 글] 쌍 입력, 종합 + 분석 8개 항목 동시 회귀
- `analysis.py`: 부분집단·항목별 분석, 그림
- `res_*.json`, `table_*.csv`, `fig_*.png`, `pred_*.npy`: 결과물
- `log_klue_rb.txt`: 학습 로그

## 설정
평가는 Validation 4,000편, 학습은 Training 32,006편 중 5%를 dev로 분리한 나머지. 시드 0 단일 실행. 3세대(LLM 채점)는 구현하지 않았다.
