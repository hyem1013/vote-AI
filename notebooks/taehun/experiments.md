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