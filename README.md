# 한국어 서술형 글쓰기 자동 채점 (AES)

Term Project #1. AI Hub 「서술형 글쓰기 평가 데이터」(dataSetSn=71818, 36,006편)로 종합 점수(1~4)를 예측하고 QWK로 세 세대를 비교한다. 보고서: `1102543003.pdf`

| 모델 | QWK(채점자 평균) |
|---|---|
| 표면 특징 + 문자 TF-IDF 릿지 (1세대) | 0.457 |
| CNN+LSTM (2세대, 3시드 평균) | 0.461 ± 0.004 |
| KLUE-RoBERTa (2세대, 3시드 평균) | **0.551 ± 0.007** |
| KLUE-RoBERTa 3시드 앙상블 / + dev 구간 보정 | 0.559 / 0.579 |
| 사람-사람 (상한) | 0.693 |
| LLM-as-a-Judge, 평가 샘플 300편 (zero-shot / few-shot) | 0.369 / 0.378 (같은 샘플 KLUE 0.572) |

## 데이터
원본은 AI Hub 이용 신청 후 내려받아야 하며 저장소에 포함하지 않는다. `extract.py`의 `base`를 내려받은 폴더(`3.개방데이터/1.데이터/`)로 맞춘다.

## 실행 순서
```bash
pip install "numpy<2" "transformers==4.44.2" pandas scikit-learn scipy matplotlib torch
python extract.py                                     # 라벨링 zip -> essays.csv
python baseline.py                                    # 1세대
bash run_seeds.sh                                     # KLUE-RoBERTa 6에폭 x 시드 0,1,2 (RTX 3060 Ti, 약 2.2시간)
for s in 0 1 2; do python cnnlstm.py $s; done         # CNN+LSTM
python dev_pred.py klue_rb6_s0 klue_rb6_s1 klue_rb6_s2
python final.py                                       # 시드 평균/앙상블/구간 보정/핵심어 스태킹/항목별 -> res_final.json
python llm_prep.py                                    # LLM 심사용 300편 샘플 (원문 입력 파일 생성)
python llm_eval.py final_ens                          # LLM 점수 평가 + 부트스트랩 CI
python analysis.py; python rule_check.py              # 표/그림, 핵심어 규칙 검증
```
LLM 채점 방식은 `llm/PROMPTS.md`.

## 파일
- `common.py` QWK·평가·로드 / `extract.py` JSON 파싱 / `baseline.py` 1세대
- `bert.py` KLUE-RoBERTa(지시문+글 쌍, 종합+분석 8개 동시 회귀) / `cnnlstm.py` CNN+LSTM
- `dev_pred.py`, `final.py`, `analysis.py`, `rule_check.py` 분석
- `llm_prep.py`, `llm_eval.py`, `llm/` LLM 심사(출력 `out_*.json`, 정답 `sample_key.csv`)
- `res_*.json`, `table_*.csv`, `fig_*.png`, `pred_*.npy`, `log_*.txt` 결과물

## 설정
평가는 Validation 4,000편, 학습은 Training 32,006편 중 5%를 dev로 분리한 나머지. KLUE는 dev QWK 최고 에폭을 선택. 시드 0,1,2.
