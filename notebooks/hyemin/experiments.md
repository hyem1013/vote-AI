# Experiments 요약 (로그 기반)

## 공통 피처 요약
- Q_A 태도 요약: `neg_att`, `pos_att`
- 확신/중도 비율: `confident_ratio`, `neutral_ratio`
- Q_A 결측비율: `qa_missing_ratio`
- Big5 성향: 각 성향 `diff` + `strength(|diff|)` 10개
- 단어 인지: `wr_sum`, `wf_sum`, `word_credibility`, `cred_bin`
- 인구통계 파생: `age_group_ord`, `education`, `urban_ord`
- 무응답 indicator: `education_is_missing`, `urban_is_missing`, `hand_is_missing`, `married_is_missing`
- 범주형: `race/religion`(또는 단순화 버전), `hand_cat`, `married_cat`

## 기록 형식
- 모델 / 피처 / 모델셋팅 / AUC(로그) / 시행착오·다음

---

01_MLP_v1_baseline.ipynb
- 모델: VotingMLP (Residual MLP)
- 피처: neg_att, pos_att, neutral_ratio, confident_ratio, Big5 diff/strength, wr_sum, wf_sum, word_credibility, cred_bin, age_group_ord, education, urban_ord, hand_cat, married_cat, race_simple, religion_simple
- 모델셋팅: AdamW lr=3e-4, weight_decay=3e-4, pos_weight, ReduceLROnPlateau, early stopping, StandardScaler
- AUC(로그): Validation ROC-AUC 0.7620
- 시행착오/다음: 기본 파이프라인 확인 → Transformer + Optuna로 탐색

02_FTTransformer_v1_optuna.ipynb
- 모델: FT-Transformer + Optuna
- 피처: 01 기반 + cred_bin, hand/married 원본 카테고리
- 모델셋팅: Optuna best params 로그 출력(예: d_token=128, n_heads=8, n_layers=2, dropout≈0.26)
- AUC(로그): Best Validation ROC-AUC 0.7707
- 시행착오/다음: 단일 split 결과 → KFold로 안정성 검증

03_FTTransformer_v2_kfold.ipynb
- 모델: FT-Transformer
- 피처: 02 기반 + missing indicator + qa_missing_ratio + race/religion 원본 + 스케일링 분리
- 모델셋팅: KFold 5
- AUC(로그): Mean Validation ROC-AUC 0.7648
- 시행착오/다음: 평균 안정적이나 개선폭 작음 → seed 앙상블

04_FTTransformer_v3_seed_ensemble.ipynb
- 모델: FT-Transformer seed 앙상블
- 피처: 03 동일
- 모델셋팅: SEEDS=[42,202,777] (로그 기준 seed별 평균)
- AUC(로그): [Seed 777] Mean Validation ROC-AUC 0.7638
- 시행착오/다음: 구조 개선보다 FE 개선 필요 → 새로운 FE 파이프라인 전환

05_FTTransformer_v4_gemini_feats.ipynb
- 모델: FT-Transformer
- 피처: Base FE + Gemini
- 모델셋팅: 단일 split
- AUC(로그): 로그 출력 없음
- 시행착오/다음: 로그 재실행 필요

06_FTTransformer_v5_optuna_full.ipynb
- 모델: FT-Transformer + Optuna
- 피처: QE/TP + demo + cat (full)
- 모델셋팅: Optuna
- AUC(로그): 로그 출력 없음
- 시행착오/다음: 로그 재실행 필요

07_FTTransformer_v6_optuna_min.ipynb
- 모델: FT-Transformer + Optuna
- 피처: 최소 피처(나이/학력 + race/religion + age_edu + married)
- 모델셋팅: Optuna
- AUC(로그): 로그 출력 없음
- 시행착오/다음: 로그 재실행 필요

08_FTTransformer_v7_optuna_mid.ipynb
- 모델: FT-Transformer + Optuna
- 피처: v6 + reliability_score/total_time_log/tp_emo
- 모델셋팅: Optuna
- AUC(로그): 로그 출력 없음
- 시행착오/다음: 로그 재실행 필요

09_Ensemble_MLP_FT_v1.ipynb
- 모델: MLP + FT-Transformer 앙상블 (예측 평균)
- 피처: Q_A/Q_E log + WR/WF + TP Big5 + demo flags + interaction + TE
- 모델셋팅: KFold 5, seed=42
- AUC(로그): OOF AUC 0.77078
- 시행착오/다음: 앙상블 안정화 → TE/정규화/스케일링 개선

10_MLP_v2_te.ipynb
- 모델: SimpleMLPWithEmbedding + TE
- 피처: Q_A/Q_E log/WR/WF + demo flags + QA/QE 집계 + TP Big5 + vocab + interaction
- 모델셋팅: KFold 5
- AUC(로그): OOF AUC 0.76740
- 시행착오/다음: 분포 안정화 → QuantileTransformer 시도

11_MLP_v3_te_quantile.ipynb
- 모델: SimpleMLPWithEmbedding + TE
- 피처: v2 + QuantileTransformer
- 모델셋팅: KFold 5
- AUC(로그): OOF AUC 0.76806
- 시행착오/다음: raw/center/full 비교로 확장

12_FTTransformer_v8_rawcenter.ipynb
- 모델: FT-Transformer
- 피처: 기존 FE + raw center 버전
- 모델셋팅: KFold 5
- AUC(로그): OOF AUC 0.76594
- 시행착오/다음: full feature set 확인

13_FTTransformer_v9_full.ipynb
- 모델: FT-Transformer
- 피처: FEATURE_SET=full
- 모델셋팅: KFold 5
- AUC(로그): OOF AUC 0.75926
- 시행착오/다음: FT 성능 하락 → 다른 모델/앙상블로 전환

14_Ensemble_FT_MLP_v2.ipynb
- 모델: FT-Transformer + MLP 앙상블
- 피처: new3 버전(원문항 + 파생 혼합)
- 모델셋팅: KFold 5, SEED 앙상블
- AUC(로그): 로그 출력 없음
- 시행착오/다음: 로그 재실행 필요

15_MLP_v4_new3.ipynb
- 모델: AutoGluon (NN 포함)
- 피처: new3 축소 버전
- 모델셋팅: AutoGluon 앙상블
- AUC(로그): Best CV Score (roc_auc) 0.7796167
- 시행착오/다음: AutoGluon vs 수동 MLP/FT 비교

--- 새 FE 라인 ---

15_new_5th_test.ipynb
- 모델: SimpleMLPWithEmbedding + Target Encoding + 5-Fold
- 피처: 원본 Q_A/Q_E log/WR/WF + 다수 파생(인구통계 flags, QA/QE 집계, TP Big5, vocab, interaction)
- 모델셋팅: KFold 5
- AUC(로그): OOF AUC 0.76740
- 시행착오/다음: MLP 단독 한계 → FT-Transformer와 앙상블

15-1_test(FTtransformer).ipynb
- 모델: MLP + FT-Transformer 앙상블 (예측 평균)
- 피처: v4 기반 피처 + FT-Transformer용 전처리
- 모델셋팅: 앙상블 평균
- AUC(로그): OOF AUC 0.77078
- 시행착오/다음: FE 개선 여지 탐색 → rawcenter / full feature 실험

17_new2_test.ipynb
- 모델: FT-Transformer + 5-Fold (seed=42)
- 피처: 기존 FE + raw center 버전 (run_name=ft_rawcenter)
- 모델셋팅: KFold 5, seed=42
- AUC(로그): OOF AUC 0.76594
- 시행착오/다음: rawcenter 효과 제한적 → full feature set 확인

18_new2_test.ipynb
- 모델: FT-Transformer + 5-Fold (seed=42)
- 피처: FEATURE_SET=full
- 모델셋팅: KFold 5, seed=42
- AUC(로그): OOF AUC 0.75926
- 시행착오/다음: FT 성능 하락 → MLP Optuna 재탐색

--- 19~21 및 colab ---

19_v5_optuna_mlp.ipynb
- 모델: PyTorch MLP + Optuna (v5 피처셋 117개)
- 피처: v5 FE + TE 조합
- 모델셋팅: Optuna 탐색(3-fold) → best로 5-fold 전체 학습
- AUC(로그): OOF AUC 0.76663 (Best AUC 0.76798)
- 시행착오/다음: v5+Optuna 기준점 확보 → CLAUDE18 축소 피처 비교

20_mlp_optuna_claude18.ipynb
- 모델: PyTorch MLP + Optuna (CLAUDE18 피처)
- 피처: CLAUDE18 고정 피처셋
- 모델셋팅: Optuna 탐색 → best로 5-fold 전체 학습
- AUC(로그): BEST OOF AUC 0.76570 (fold mean 0.76590)
- 시행착오/다음: 축소 피처 성능 확인 → v5 FE 교체/앙상블과 비교

21_v5_fe_swap_ensemble.ipynb
- 모델: MLP + FT-Transformer 앙상블 (v5 FE 교체 버전)
- 피처: v5 FE 교체 버전
- 모델셋팅: 앙상블 평균
- AUC(로그): OOF AUC 0.76917
- 시행착오/다음: 앙상블 vs 단일 비교 → v7 Optuna 파이프라인으로 정리

colab_v7_optuna.ipynb
- 모델: v7 Optuna (MLP/FT/앙상블)
- 피처: v7 파이프라인
- 모델셋팅: seed 앙상블(42/202/777/1024/2048)
- AUC(로그): FINAL OOF AUC 0.77427
- 시행착오/다음: 최종 제출 후보 통합 및 문서화

--- 22~24 최근 앙상블/비율 테스트 ---

22_bestmodel_test(m1+tabnet).ipynb
- 모델: TabNet (m1 + TabNet 조합을 위한 TabNet 단독 학습)
- 피처: TabNet용 전처리(일부 QxE drop, hand drop, 범주형 string 처리 후 one-hot)
- 모델셋팅: TabNetClassifier, N_REPEAT=3, N_SKFOLD=5
- AUC(로그): TabNet CV AUC 0.76354, 조합 예상 AUC 0.78046
- 시행착오/다음: m1과 소량 혼합에서 기대값 확인 → SNN 포함 3-way 앙상블로 확장

23_bestmodel_test(m1+m4+tabnet).ipynb
- 모델: 3-way 앙상블 (nn + SNN + TabNet)
- 피처: 각 모델 산출물(blend)
- 모델셋팅: 후보 가중치 조합 비교
- AUC(로그): 예상AUC 최고 0.78114 (0.97/0.03 등 유사)
- 시행착오/다음: 예상치 기준 후보 추려 실제 제출/검증 필요

24_THmodel_ratio_test.ipynb
- 모델: Rank-averaging (m1 OOF + m7 OOF)
- 피처: OOF 예측값 rank 변환 후 가중 평균
- 모델셋팅: weights 0.46~0.54 스윕
- AUC(로그): 로그 출력 없음(미실행)
- 시행착오/다음: 실행 로그 확보 후 최적 비율 기록 필요
