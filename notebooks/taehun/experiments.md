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
- **Best Validation AUC:** 0.7673 (at Epoch 19)
- **Best Validation Loss:** 0.5588
- **Final Validation ACC:** 0.6692
- **Dacon Score:** 0.7570061244(Public) / 0.7478870474(Private)
- **Dacon Score(try2):** 0.7675567885(Public) / 0.7654633026(Private) (epoch 100->25)

### 4. 분석 및 계획 (Analysis & Next Steps)
- **진단:** 19회차 이후 검증 손실(Val Loss)이 지속적으로 상승하며 심한 과적합 발생.
- **조치:** 차기 실험 시 조기 종료(Early Stopping) 적용 및 최대 에포크 25회 제한.
- **계획:** Dropout 비율 상향 및 Weight Decay 설정을 통해 모델의 일반화 성능 강화 시도.

---

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
- **Best Validation AUC (Optuna):** 0.7644 (Trial 평균 기준)
- **Final Validation AUC (Local):** 0.7698 (8:2 분할 재학습 25 Epoch 최종 결과)
- **Dacon Score:** - 0.7682606379(Public) / 0.7654592368 (Private)
- **비고:** Experiment 01 대비 Public Score 약 **+0.0007** 상승 및 Private Score 안정화 확인

### 4. 분석 및 계획 (Analysis & Next Steps)
- **진단:** Optuna 최적화와 AdamW/BCEWithLogitsLoss 조합이 학습의 안정성을 크게 개선함. 특히 시각화를 통해 25 에포크가 과적합을 피하면서 성능을 극대화하는 지점임을 확인.
- **조치:** 에포크 수를 공격적으로 제한(100 -> 25)하여 일반화 성능(Private Score)을 방어함.
- **계획:** 단일 MLP 모델의 한계를 극복하기 위해, 트리 기반 모델(LGBM, CatBoost)과의 앙상블(Soft Voting) 혹은 모델 구조를 Deep & Cross Network(DCN) 형태로 고도화 시도 예정.

---

## Experiment 03: FT-Transformer with Embedding & Attention
* **Date:** 2026-01-29
* **Model:** FT-Transformer (via `rtdl_revisiting_models`)
* **Status:** Completed

### 1. 사용 피처 (Feature Engineering)
* **범주형 피처 (Embedding 대상):** `race`, `religion`, `urban`, `education`, `hand`, `married`, `engnat`, `gender`
* **수치형 피처 (Continuous):** `mach_score`, `wr_total`, `wf_total`, `age_encoded`, `familysize`, `Q_A` 시리즈, `Q_E` 시리즈(**로그 변환**), `tp_` 파생 변수들
* **변경 사항:** 기존 Experiment 02의 타겟 인코딩을 제거하고, 범주형 데이터를 원본 정수 형태(**Label Encoding**)로 입력하여 모델 내부의 **Embedding Layer**가 피처 간 관계를 직접 학습하도록 개선.

### 2. 하이퍼파라미터 (Hyperparameters)
* **Model Structure:** `n_blocks` (Layer 수), `d_block` (임베딩 차원), `attention_n_heads=8`
* **Optimizer:** **AdamW** (Weight Decay=0.01 적용)
* **Batch Size:** **512 / 1024** (Optuna 탐색 결과에 따름)
* **Epochs:** **25** (Final Model Trend 시각화 결과, 20~25 에포크 사이 최적 AUC 도달 확인)
* **Tuning:** **Optuna (15 Trials)**를 통해 `lr`, `n_blocks`, `d_block`, `dropout` 최적화 수행 (약 3시간 소요)

### 3. 검증 성능 (Validation Results)
* **Best Validation AUC (Optuna):** 0.7689 (5-Fold CV 평균)
* **Final Validation AUC (Local):** 0.7760 (8:2 분할 재학습 25 Epoch 최종 결과)
* **Dacon Score:** - 0.7778068368 (Public) / 0.7716801382 (Private)
* **비고:** 단일 모델 기준으로 현재까지 가장 높은 성능을 기록함. MLP 대비 Public Score 약 **+0.009** 상승 달성.

### 4. 분석 및 계획 (Analysis & Next Steps)
* **진단:** **Attention 메커니즘**이 수치형/범주형 피처 간의 복잡한 상호작용을 MLP보다 훨씬 정교하게 포착함. 특히 타겟 인코딩 같은 수동적 압축보다 **고차원 임베딩** 방식이 본 데이터셋에 더 효과적임을 증명함.
* **조치:** 3시간 이상의 긴 학습 시간에도 불구하고 유의미한 성능 향상을 이끌어냄. 시각화를 통해 25 에포크가 일반화 성능을 방어하는 적절한 지점임을 확인.
* **계획:** 현재 FT-Transformer의 단일 성능이 매우 우수하므로 feature engineering을 통해 개선 예정.

---

## Experiment 04: Feature Selection & Analysis via AutoML (PyCaret)
- **Date:** 2026-01-29
- **Model:** CatBoost (Best for Feature Importance Analysis)
- **Status:** Completed (Selected 100+ Features)
- **file:** 01_Feature_Selection_PyCaret.ipynb

### 1. 주요 분석 결과 (AutoML Insights)
- **배경 정보의 지배력:** `education`, `age_group_10s`가 중요도 1, 2위를 차지함. 투표 여부 결정에 있어 심리 성향보다 인구통계적 배경이 핵심 지표임을 확인.
- **반응 시간(Q_E)의 재발견:** `Q_E` 시리즈가 중요도 상위권에 대거 포진함. "무엇을 답했는가"만큼 "얼마나 고민했는가"가 투표 행위 예측에 유의미한 단서임. 차라리 이상치를 제거하고 로그변환을 해서 사용하는게 나을것 같아 보임.
- **Q_A의 상대적 낮은 순위:** 의외로 개별 문항 답변(`Q_A`)은 인적 사항이나 응답 시간 패턴에 비해 단일 변수로서의 영향력은 다소 낮음.

### 2. 피처 선정 및 처리 전략 (Feature Strategy)
- **A급 유지:** Importance 상위 25개 (`education`, `age_group`, `Q_E` 전수 등) 및 주요 인종(`White`, `Asian`) 포함.
- **B급 유지:** 비선형 관계 학습을 위해 중요도 0.4 이상의 성향(`tp`) 및 주요 심리 질문(`Q_A`) 유지.
- **C급 제거:** 모델 노이즈 방지를 위해 중요도 0.1 미만 및 극소수 범주(`wr_12`, 특정 소수 종교 등) 제외.
- **전처리 가이드:** 시간 데이터(`Q_E`)의 큰 편차를 줄이기 위해 `Log1p` 변환 필수 적용.

### 3. 향후 계획 (Next Steps)
- **TabNet:** TabNet을 통하여 MLP > FT-Transformer - TabNet 순으로 동일 피처에 대한 각 모델의 성능을 대략적으로 체크 이후 AutoML로 선별된 정예 피처셋을 적용하여 불필요한 노이즈를 제거한 모델 학습 진행.
- **DCN(Deep & Cross Network):** 피처 간 상호작용(Feature Interaction)을 명시적으로 학습하는 구조 도입 검토.

<details> <summary> 사용 피처 목록(FEATURES_TO_USE) 전체 보기 (클릭)</summary>
```python
FEATURES_TO_USE = [
    # 인구통계 및 환경
    'education', 'age_group', 'race', 'married', 'familysize', 'engnat', 'gender', 'religion', 'urban',
    # 답변 시간 (상위권 위주 혹은 전체) - Log 변환 대상, cliping도 고려해야함 (이상치 정리용으로 1%가 적당해보임)
    'QaE', 'QbE', 'QcE', 'QdE', 'QeE', 'QfE', 'QgE', 'QhE', 'QiE', 'QjE', 
    'QkE', 'QlE', 'QmE', 'QnE', 'QoE', 'QpE', 'QqE', 'QrE', 'QsE', 'QtE',
    # 성향 (tp)
    'tp01', 'tp02', 'tp03', 'tp04', 'tp05', 'tp06', 'tp07', 'tp08', 'tp09', 'tp10',
    # 질문 답변 (상위권 위주)
    'QcA', 'QrA', 'QqA', 'QpA', 'QeA', 'QdA', 'QtA', 'QjA', 'QfA', 'QhA'
]
```
</details>

---

## Experiment 04: AutoGluon Multi-modal Ensemble (MLP + FT-Transformer)
- **Date:** 2026-01-30
- **Model:** AutoGluon (TabularPredictor)
- **Status:** ❌ FAILED (Kernel Crash / Memory Out)

### 1. 시도 내용 (Intended Strategy)
- **목표:** 단일 FT-Transformer의 성능 한계를 넘기 위해 AutoGluon을 활용한 MLP 및 Transformer 모델 앙상블 시도.
- **설정:** `presets='best_quality'`, `hyperparameters={'NN_TORCH': {}, 'FT_TRANSFORMER': {}}` 적용.

### 2. 발생 문제 (Issue Diagnosis)
- **증상:** 학습 진행 중 커널 연결 끊김(Dead Kernel) 및 시스템 프리징 발생.
- **원인 분석:** 1. **메모리 부족 (OOM):** AutoGluon의 `best_quality` 설정은 5-Fold 이상 스태킹을 기본으로 하므로, M1 Pro의 통합 메모리(Unified Memory) 한계를 초과함.
    2. **장치 충돌:** AutoGluon 내부의 Ray/Dask 병렬 처리 프로세스가 PyTorch MPS(Metal) 가속과 충돌했을 가능성.

### 3. 향후 계획 (Next Steps / Pivot)
- **경량화 시도:** `presets='high_quality'` 또는 `medium_quality`로 낮추어 스태킹 깊이 제한.
- **수동 앙상블:** AutoGluon에 의존하지 않고, 이미 학습 완료된 **Exp 02(MLP)**와 **Exp 03(FT-Transformer)**의 가중 평균(Soft Voting)을 코드로 직접 구현하여 메모리 부하 최소화.

---

## Experiment 05: AutoGluon Multi-modal Ensemble (MLP + FT-Transformer)
- **Date:** 2026-01-30
- **Model:** AutoGluon (TabularPredictor)
- **Status:** ❌ FAILED (Kernel Crash / Memory Out)

### 1. 발생 문제 (Issue Diagnosis)
- **증상:** 학습 진행 중 커널 연결 끊김(Dead Kernel) 및 시스템 프리징 발생.
- **원인 분석:** 1. **메모리 부족 (OOM):** AutoGluon의 `best_quality` 설정은 5-Fold 이상 스태킹을 기본으로 하므로, M1 Pro의 통합 메모리(Unified Memory) 한계를 초과함.
    2. **장치 충돌:** AutoGluon 내부의 Ray/Dask 병렬 처리 프로세스가 PyTorch MPS(Metal) 가속과 충돌했을 가능성.
    3. 04와 동일.