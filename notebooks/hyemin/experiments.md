# Experiments 요약

## 피처 생성 요약 (공통)
- Q_A 태도 요약: `neg_att`, `pos_att` (부정/긍정 문항 평균)
- 확신/중도 비율: `confident_ratio`, `neutral_ratio` (응답한 문항 기준 비율)
- Q_A 결측비율: `qa_missing_ratio` (무응답 비율)
- Big5 성향: 각 성향 `diff` + `strength(|diff|)` 10개
- 단어 인지: `wr_sum`, `wf_sum`, `word_credibility = wr_sum - wf_sum`, `cred_bin`
- 인구통계 파생: `age_group_ord`, `education`, `urban_ord`
- 무응답 indicator: `education_is_missing`, `urban_is_missing` `hand_is_missing`, `married_is_missing`
- 범주형: `race/religion`(또는 단순화된 버전), `hand_cat`, `married_cat`

## 기록 형식
- 날짜/시간, 모델/설정, 피처 특징, AUC, 제출파일, 메모 순으로 기록

[2026-01-28 10:45] - 02_first_test.ipynb
- 모델: VotingMLP (Residual MLP)
- 학습셋팅: AdamW lr=3e-4, weight_decay=3e-4, pos_weight, ReduceLROnPlateau, early stopping, StandardScaler
- 피처: neg_att, pos_att, neutral_ratio, confident_ratio, Big5 diff/strength, wr_sum, wf_sum, word_credibility, cred_bin,
- age_group_ord, education, urban_ord, hand_cat, married_cat, race_simple, religion_simple
- 라벨: voted==2 -> 1 (class2 양성)
- AUC: 0.76
- 제출파일: submission_prob_class2.csv
- 메모: 02_first_test 기준 파이프라인 적용



[2026-01-28 14:36] - 03_second_test.ipynb
- 모델: FT-Transformer + Optuna(03_second_test)
- 피처: 02 기반 + cred_bin, hand/married 원본 카테고리
- 라벨: voted==2 -> 1 (class2 양성)
- AUC: 0.7707
- 제출파일: submission_prob_second_test.csv (아직 해커톤에 제출 X)
- 메모: 제출 전, best params 적용 결과 기록



[2026-01-28 14:45] - 04_third_test.ipynb
- 모델: FT-Transformer (04_third_test, 기본 파라미터)
- 피처: 03 기반 + missing indicator + qa_missing_ratio + race/religion 원본 + 스케일링 분리
- 라벨: voted==2 -> 1 (class2 양성)
- AUC: 0.7648 (5-fold mean)
- 제출파일: submission_prob_third_test.csv (아직 해커톤에 제출 X)
- 메모: KFold 평균 결과 기록




[2026-01-28 16:18] - 05_third_test(seedensemble).ipynb
모델: FT-Transformer (05_seed_ensemble_test, seed 앙상블)
피처: 04_third_test와 동일 (missing indicator/qa_missing_ratio 포함)
라벨: voted==2 -> 1 (class2 양성)
AUC: 0.7638 (seed 42, 202, 777 mean)
제출파일: submission_prob_third_test(seed_ensemble).csv (아직 해커톤에 제출 X)
메모: SEEDS=[42,202,777] 앙상블, seed별 test_probs 저장 후 평균


---------- 여기서부터 새롭게 feature engineering ----------

[2026-01-29 17:19] - 15_new_5th_test.ipynb
- 모델: SimpleMLPWithEmbedding + Target Encoding + 5-Fold
- 피처: 원본 Q_A/Q_E log/WR/WF + 다수 파생(인구통계 flags, QA/QE 집계, TP Big5, vocab, interaction)
- 범주형: gender/race/religion
- 라벨: voted==2 -> 1 (class2 양성)
- 결과: 코드 내 OOF 0.767 - LB 0.775
- 제출파일: submission_15_new_5th_test.csv (해커톤 제출 완료)



[2026-01-30 10:00] - 15-1_test(FTtransformer).ipynb
- 모델: MLP + FT-Transformer 앙상블 (예측 평균)
- 피처: v4 기반 피처 + FT-Transformer용 전처리
- 라벨: voted==2 -> 1 (class2 양성)
- AUC: OOF AUC: 0.77078 - LB 0.778 (해커톤 제출 완료)
- 제출파일: submission_ensemble.csv



[2026-01-30 10:00] - 17_new2_test.ipynb
- 모델: FT-Transformer + 5-Fold (seed=42)
- 피처: 기존 FE + raw center 버전 (코드 내 run_name=ft_rawcenter)
- 라벨: voted==2 -> 1 (class2 양성)
- AUC: oof 0.76594 (해커톤 제출 X)
- 제출파일: submission_ft_rawcenter_seed42.csv



[2026-01-30 10:00] - 18_new2_test.ipynb
- 모델: FT-Transformer + 5-Fold (seed=42)
- 피처: FEATURE_SET=full
- 라벨: voted==2 -> 1 (class2 양성)
- AUC: oof 0.75926 (해커톤 제출 X)
- 제출파일: submission_ft_full_seed42.csv



---------- 19~21 및 colab 정리 ----------

[날짜 미정] - 19_v5_optuna_mlp.ipynb
- 모델: PyTorch MLP + Optuna (v5 피처셋 117개)
- 제출파일: submission_19_v5_optuna_mlp.csv

[날짜 미정] - 20_mlp_optuna_claude18.ipynb
- 모델: PyTorch MLP + Optuna (CLAUDE18 피처)
- 제출파일: submission_20_mlp_optuna_claude18_*.csv

[날짜 미정] - 21_v5_fe_swap_ensemble.ipynb
- 모델: MLP + FT-Transformer 앙상블 (v5 FE 교체 버전) -> 태헌님 요청해주신 테스트 
- 제출파일: submission_21_v5_fe_swap_ensemble.csv / submission_21_v5_fe_swap_mlp.csv / submission_21_v5_fe_swap_ft.csv

[날짜 미정] - colab_v7_optuna.ipynb
- 모델: v7 Optuna (MLP/FT/앙상블)
- 제출파일: submission_colab_v7_optuna_ensemble.csv / submission_colab_v7_optuna_mlp.csv / submission_colab_v7_optuna_ft.csv
- 리포트: report_colab_v7_optuna.json
