# LLM-as-a-Judge 설정 (300편 샘플)

- 샘플: `llm_prep.py`가 Validation에서 300편을 `random_state=0`으로 추출. 정답 점수는 `sample_key.csv`에만 있고 심사자에게 주지 않음.
- 심사자: Claude(Sonnet 계열) 하위 에이전트 12개(zero-shot 6, few-shot 6), 에이전트당 50편.
- 입력: 지시문, 핵심어, 글, 해당 qid의 루브릭(8개 항목 x 1~5단계 설명). 원문 데이터는 AI Hub 라이선스상 저장소에 포함하지 않음.
- few-shot: 학습 세트에서 점수 1,2,3,4별 2편씩 8편(`random_state=1`)을 보정용 예시로 제공.
- 출력: 종합 점수 1~4 정수와 한국어 근거 한 줄. 척도(1=매우 미흡, 4=우수)는 프롬프트에 명시.
- 첫 시도는 루브릭 파일이 길어 일부 에이전트가 읽지 못해 폐기했고, 루브릭을 offset/limit로 나눠 끝까지 읽게 한 뒤 재채점했다. 결과는 `out_zs_*.json`, `out_fs_*.json`.
- 평가: `python llm_eval.py final_ens`
