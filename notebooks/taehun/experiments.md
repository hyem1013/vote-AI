## Experiment 01: Baseline Deep MLP with Target Encoding
- **Date:** 2026-01-28
- **Model:** 3-Layer MLP (128 → 64 → 32)
- **Status:** Completed (Overfitted)

### 1. 사용 피처 (Feature Engineering)
- **집계 지표:** `mach_score`, `wr_total`, `wf_total`
- **성격 지표 (TIPI):** `tp_Extraversion`, `tp_Agreeableness`, `tp_Conscientiousness`, `tp_EmotionalStability`, `tp_Openness`
- **타겟 인코딩 (m=10):** `race`, `religion`, `urban`, `education`, `hand`, `married`, `engnat`
- **기타 수치/범주:** `age_encoded`, `familysize`, `gender`, `Q_A`, `Q_E`(로그 변환)

### 2. 하이퍼파라미터 (Hyperparameters)
- **Optimizer:** Adam (lr=0.001)
- **Loss Function:** BCELoss
- **Batch Size:** 1024
- **Epochs:** 100 (Best at 19)
- **Regularization:** BatchNorm, Dropout (0.3, 0.2)

### 3. 검증 성능 (Validation Results)
- **Best Validation AUC:** **0.7673** (at Epoch 19)
- **Best Validation Loss:** 0.5588
- **Final Validation ACC:** 0.6692
- **Dacon Score:** 0.7570061244(Public) / 0.7478870474(Private)
- **Dacon Score(try2):** 0.7675567885(Public) / 0.7654633026(Private) (epoch 100->25)

### 4. 분석 및 계획 (Analysis & Next Steps)
- **진단:** 19회차 이후 검증 손실(Val Loss)이 지속적으로 상승하며 심한 과적합 발생.
- **조치:** 차기 실험 시 조기 종료(Early Stopping) 적용 및 최대 에포크 25회 제한.
- **계획:** Dropout 비율 상향 및 Weight Decay 설정을 통해 모델의 일반화 성능 강화 시도.




## Experiment 02: Optimized MLP with Optuna & 3rd Place Strategy
- **Date:** 2026-01-28
- **Model:** Tuned 3-Layer MLP (Hidden: 128~256)
- **Status:** Completed (SOTA Reached)

### 1. 사용 피처 (Feature Engineering)
- **집계 지표:** `mach_score`, `wr_total`, `wf_total`
- **성격 지표 (TIPI):** `tp_Extraversion`, `tp_Agreeableness`, `tp_Conscientiousness`, `tp_EmotionalStability`, `tp_Openness`
- **타겟 인코딩 (m=10):** `race`, `religion`, `urban`, `education`, `hand`, `married`, `engnat`
- **기타 수치/범주:** `age_encoded`, `familysize`, `gender`, `Q_A`, `Q_E`(로그 변환)
- Experiment 01과 동일

### 2. 하이퍼파라미터 (Hyperparameters)
- **Optimizer:** AdamW (Weight Decay=0.01 설정)
- **Loss Function:** BCEWithLogitsLoss (수치적 안정성을 위해 Sigmoid 미포함 Logit 출력 방식 사용)
- **Batch Size:** 1024 (Optuna 탐색 범위 내 최적값 선택)
- **Epochs:** 25 (Training Trend 시각화 결과, 25회차 이후 검증 AUC 상승폭 둔화에 따른 고정)
- **Tuning:** Optuna (30 Trials)를 통해 `lr`, `hidden_dim`, `dropout` 자동 최적화 수행

### 3. 검증 성능 (Validation Results)
- **Best Validation AUC (Optuna):** **0.7644** (Trial 평균 기준)
- **Final Validation AUC (Local):** **0.7698** (8:2 분할 재학습 25 Epoch 최종 결과)
- **Dacon Score:** - 0.7682606379(Public) / 0.7654592368 (Private)
- **비고:** Experiment 01 대비 Public Score 약 **+0.0007** 상승 및 Private Score 안정화 확인

### 4. 분석 및 계획 (Analysis & Next Steps)
- **진단:** Optuna 최적화와 AdamW/BCEWithLogitsLoss 조합이 학습의 안정성을 크게 개선함. 특히 시각화를 통해 25 에포크가 과적합을 피하면서 성능을 극대화하는 지점임을 확인.
- **조치:** 에포크 수를 공격적으로 제한(100 -> 25)하여 일반화 성능(Private Score)을 방어함.
- **계획:** 단일 MLP 모델의 한계를 극복하기 위해, 트리 기반 모델(LGBM, CatBoost)과의 앙상블(Soft Voting) 혹은 모델 구조를 Deep & Cross Network(DCN) 형태로 고도화 시도 예정.