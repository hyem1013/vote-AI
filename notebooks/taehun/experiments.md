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

---

### Experiment 06: ResNet-style Deep MLP 시도
파일: 05_ResNet-style Deep MLP_v1.ipynb
결과: 실패 (AUC 0.7662)

원인: > 
1. 모델의 깊이 증가로 인한 과적합 발생 및 데이터 스케일링 부재. 
2. LayerNorm보다 기존의 수동 Feature Scaling이 본 데이터셋의 척도(1-5점)에 더 적합함 확인.

Action Item: 모델 구조를 단순화하고 성공했던 수동 스케일링 로직으로 회귀.

---

### Experiment 07: Wide MLP with BatchNorm & Manual Scaling
Date: 2026-01-31
Model: Wide MLP (91 → 512 → 128 → 1)
Status: Completed (Performance Recovered)

1. 시도 내용 (Intended Strategy)
모델 구조 변경: 너무 깊었던 ResNet 구조를 버리고, 층의 너비를 넓힌(512 node) Wide MLP 구조 채택.
정규화 층 도입: BatchNorm1d를 추가하여 내부 공변량 변화를 제어하고 학습 속도 및 안정성 개선.
로직 복구: Experiment 06에서 실패 원인이었던 스케일링을 성공 사례였던 '수동 스케일링' 및 2. - sigmoid 로직으로 원복.
검증 체계 강화: roc_auc_score를 도입하여 매 폴드/에포크마다 실시간 AUC 모니터링 수행.

2. 결과 분석 (Result Analysis)
Validation AUC: 0.77123 (+/- 0.00529)

분석: 1. 구조적 적합성: 본 데이터셋에는 복잡한 잔차 연결(ResNet)보다 적당한 깊이와 넓은 너비를 가진 MLP가 더 강건(Robust)하게 작동함을 확인. 2. 스케일링의 중요성: LayerNorm 단독 사용보다 도메인 지식이 반영된 수동 스케일링이 AUC 확보에 필수적임. 3. 성능 회복: Experiment 05(0.7662) 대비 약 +0.005 포인트 반등 성공.

3. 향후 계획 (Next Steps / Pivot)
과적합 방지: 현재 모델도 0.781대보다는 낮은 것으로 보아, Dropout 비율을 조금 더 높이거나(0.4 -> 0.5) Weight Decay를 미세 조정할 필요가 있음.
앙상블 전략: 현재의 Wide MLP 구조에서 Seed를 더 다양하게 가져가는 N_REPEAT 증가 전략 고려.

---

### Experiment 08: Model Capacity Reduction & Regularization Boost
Date: 2026-01-31
Model: Wide MLP (93 → 256 → 128 → 1) [파생변수 2개 포함]
file: 06_Wide_MLP_v2_256_reg.ipynb
Status: Failed to Surpass Baseline (AUC 0.7723)

1. 시도 내용 (Intended Strategy)
모델 경량화: 과적합 방지를 위해 첫 번째 레이어 노드 수를 절반으로 축소 (512 → 256).
규제 강화: Dropout 비율 상향 (0.4/0.2 → 0.5/0.3)을 통해 일반화 성능 유도.
변수 유지: 지난 실험에서 추가한 qa_std, tp_sum 파생 변수를 그대로 유지한 채 학습.

2. 결과 분석 (Result Analysis)
Validation AUC: 0.77230 (+/- 0.00511)

분석: 
1. 미미한 반등: Exp 07(0.7712) 대비 약 +0.001 상승했으나, 여전히 베이스라인(0.7811)에 한참 못 미침. 
2. 변수 오염 가능성: 모델 구조를 경량화하고 규제를 걸었음에도 점수 회복이 안 되는 것으로 보아, 추가된 파생 변수(qa_std, tp_sum)가 AUC 최적화에 오히려 방해 요인이 되고 있다고 판단됨. 
3. 구조적 한계: BatchNorm과 LeakyReLU의 조합이 현재 데이터 전처리 로직과 완벽히 결합되지 않았을 가능성 확인.

3. 향후 계획 (Next Steps / Pivot)
전략적 회귀: 파생 변수를 모두 제거하고, 가장 점수가 좋았던 코드(0.7811)의 데이터 셋팅으로 완전히 복귀.
미세 조정: 데이터는 건드리지 않고, 모델의 레이어 크기나 학습률(Learning Rate)만 미세하게 조정하여 단일 AUC 0.778 이상 확보 시도.

---

### Experiment 09: Back to 0781 Baseline with Node Expansion (256)
Date: 2026-01-31
Model: Wide MLP (91 → 256 → 32 → 1)
file: 07_Back_to_0781_v1.ipynb
Status: Completed (Significant Potential Found)

1. 시도 내용 (Intended Strategy)
전략: 성능이 가장 좋았던 0.781 코드의 데이터 전처리(91개 피처, 수동 스케일링)로 완전히 회귀.
모델 수정: 기존 180 노드에서 256 노드로 확장하여 모델의 표현력 증대 시도.
로직: Baseline 특유의 2. - torch.sigmoid 결과 산출 방식 유지.

2. 결과 분석 (Result Analysis)
평균 Validation AUC: 0.77272
최고 AUC 기록: 0.78607 (특정 폴드에서 매우 높은 성능 발휘)

분석: 
1. 변수 제거의 정당성: 불필요한 파생 변수를 제거하자마자 최고 AUC가 0.786까지 치솟음. 피처 정제(Cleaning)가 효과적이었음. 
2. 불안정성 존재: 최고점은 높으나 평균이 0.772인 것으로 보아, 시드(Seed)나 폴드 구성에 따라 성능 편차가 큼. 
3. 가능성 확인: 모델 구조는 이미 충분히 고득점 가능성을 내포하고 있음.

3. 향후 계획 (Next Steps / Pivot)
Seed Ensemble 본격 가동: 단일 모델 튜닝보다는 여러 시드의 결과물을 합쳐 편차를 줄이는 전략이 필요함.
제출 전략: N_REPEAT를 대폭 늘려(10~20) 안정적인 0.78대 평균 점수 확보.

---

### Experiment 10: Heavy Seed Ensemble (Robustness Test)
Date: 2026-01-31
Model: Wide MLP (91 → 256 → 32 → 1)
File: 08_Seed_Esemble_v1.ipynb
Status: Completed (Baseline Stabilized)

1. 시도 내용 (Intended Strategy)
목표: Exp 09에서 확인된 모델의 잠재력을 시드 앙상블을 통해 일반화하고 리더보드 점수 안정화.
설정 변경: - N_REPEAT: 5 → 15 (시드 다양성 확보)
           N_SKFOLD: 7 → 5 (학습 효율성 및 폴드당 데이터 비중 조정)
로직 유지: 91개 순수 피처, 수동 스케일링, 2. - torch.sigmoid 출력 방식 유지.

2. 결과 분석 (Result Analysis)
평균 Validation AUC: 0.77233
최고 AUC 기록: 0.77892
분석: 
1. 평균의 수렴: 반복 횟수를 15회로 대폭 늘렸음에도 평균 AUC가 0.772대에 머무는 것으로 보아, 현재 모델 구조와 전처리 하에서는 이 점수가 임계치(Baseline Peak)인 것으로 판단됨. 
2. 최고점 하락 원인: 폴드 수를 줄임으로써(7→5) 개별 모델이 학습하는 데이터의 분산이 달라졌고, 특정 폴드에서 발생하던 '럭키샷(0.786)'이 앙상블 과정에서 희석됨. 
3. 안정성 확보: 표준 편차가 줄어들며 리더보드 제출 시 Public/Private 점수 차이가 크지 않을 것으로 기대됨.

3. 향후 계획 (Next Steps / Pivot)
피처 엔지니어링 재검토: 모델 구조(MLP)와 시드 앙상블만으로는 0.78 벽을 넘기 어려움이 확인됨. 다시 '데이터'로 돌아가 노이즈가 적은 강력한 파생 변수(예: 질문 간 상관관계 등) 1~2개만 선별 도입 필요.
Learning Rate 스케줄링: 현재 5e-3이 다소 공격적일 수 있으므로, 조금 더 낮은 LR로 더 오래 학습시키는 전략 고려.

---

### Experiment 11: TabNet for Feature Interaction & Attention
Date: 2026-01-31
Model: TabNet (Attention-based Tabular DL)
File: 09_Tabnet_v1.ipynb
Status: FAILED (Performance Degradation & Tech Issue)

1. 시도 내용 (Intended Strategy)
목표: MLP의 한계를 극복하기 위해 Sparsemax Attention을 활용하여 중요한 피처에 집중하고, 노이즈를 억제하는 정형 데이터 특화 아키텍처 도입.
설정: 91개 피처 유지, 수동 스케일링 적용, n_d=32, n_a=32, n_steps=3 설정.
로직: pytorch-tabnet 라이브러리를 사용하여 모델 스스로 피처 중요도를 학습하게 함.

2. 결과 분석 (Result Analysis)
평균 Validation AUC: 0.76093 (최고점 대비 약 -0.02 하락)
발생 문제: 
1. 성능 하락: Attention 메커니즘이 이 데이터셋의 심리 지표 간 상관관계를 MLP보다 정교하게 잡아내지 못함. 과적합이 빠르게 발생. 
2. 기술적 오류: TypeError: Cannot convert a MPS Tensor to float64 발생. pytorch-tabnet 내부 로직이 M1 Pro의 MPS(float32 전용) 환경과 호환되지 않아 학습이 중단됨.
결론: 현 데이터셋과 환경에서 TabNet 도입은 실익이 없으며, 오히려 안정성을 해침.

3. 향후 계획 (Next Steps / Pivot)
근본으로의 회귀: 검증된 0.78116 MLP 구조로 완전히 복귀.
최적화 타겟 수정: 단순히 Loss가 낮은 모델이 아닌, AUC가 가장 높은 시점의 가중치를 저장하는 로직으로 변경하여 실질적인 점수 향상 도모.
환경 최적화: 모든 텐서를 float32로 강제하여 MPS 호환성 완벽 확보.

---

## Experiment 12: Baseline Recovery with AUC-Targeted Optimization
Date: 2026-01-31
Model: Wide MLP (91 → 180 → 32 → 1)
File: 07_Back_to_0781_v2.ipynb
Status: Success (SOTA Reached)

1. 시도 내용 (Intended Strategy)
전략적 회귀: 0.78116 성공 당시의 전처리(91개 피처, 수동 스케일링)를 100% 복구하여 데이터의 무결성 확보.
최적화 지표 변경: 모델 저장 기준을 Loss에서 Validation AUC로 변경. 학습 중 가장 높은 판별력을 보인 시점의 가중치를 포착.
연산 안정화: MPS 환경에서 발생하던 데이터 타입 충돌을 float32 강제 변환과 view(-1) 적용으로 해결.

2. 하이퍼파라미터 (Hyperparameters)
N_REPEAT / N_SKFOLD: 5 / 7
Architecture: 180 (LeakyReLU) → 32 (ReLU) → 1 (Output)
Optimizer: AdamW (lr=5e-3, Weight Decay=7.8e-2)
Scheduler: CosineAnnealingWarmRestarts (T_0=8)

3. 검증 성능 (Validation Results)
Mean Validation AUC: 0.77312
Dacon Score (Public): **0.7810869275**

비고: 로컬 검증 점수(0.773)보다 리더보드 점수가 높게 형성됨. 이는 AUC 최적화 방식이 리더보드 셋의 랭킹을 더 정확하게 예측하고 있음을 시사.

---

## Experiment 13: Automated Architecture Search via Optuna
Date: 2026-01-31
Model: Dynamic MLP (91 → 288 → 64 → 1)
File: 07_Back_to_0781_v2_Optuna.ipynb
Status: Success (Stable Baseline)

1. 시도 내용 (Intended Strategy)
구조 최적화: 고정된 노드 수(180-32)에서 벗어나 Optuna를 통해 은닉층 노드 수(h1, h2)와 Dropout 비율을 실시간 탐색.
하이브리드 앙상블: 각 폴드별로 AUC 최고점을 기록한 시점의 가중치를 저장하여 최종 결과물 산출.
모니터링 강화: tqdm을 도입하여 학습 진행 상황과 실시간 AUC 추이를 시각화.

2. 최적 파라미터 (Optuna Best - Trial 7)
h1 (Hidden Layer 1): 288
h2 (Hidden Layer 2): 64
drop_rate: 0.3792 (약 38%)
Tuning Metric: Mean AUC 0.77067 (3-Fold CV 기준)

3. 검증 성능 (Validation Results)
Mean Validation AUC: 0.77272
Dacon Score (Public): **0.7809398379**

분석: 수동으로 설정한 Exp 12(0.78108)와 거의 대등한 성과를 보임. 특히 첫 번째 레이어의 노드 수를 180에서 288로 확장한 구조가 이 데이터셋의 복잡도를 더 잘 포착하는 것으로 판명됨.

4. 분석 및 향후 계획 (Analysis & Next Steps)
진단: 
1. Optuna를 통해 찾아낸 288-64 구조가 기존 180-32 구조보다 수치상 안정적임. 
2. 다만 리더보드 점수는 Exp 12가 근소하게 높으므로, 특정 시드(Seed)에 의한 운적인 요소 혹은 과적합 경계선에 걸쳐있을 가능성이 있음.
조치: Optuna가 제안한 288-64 구조를 표준으로 삼되, 과적합을 방지하기 위해 drop_rate를 0.4 수준으로 고정하는 것도 고려 가능.

계획: - Strategy B 실행: 최고점 모델(Exp 12: 0.78108)과 Optuna 모델(Exp 13: 0.78093)의 결과를 5:5 혹은 7:3으로 앙상블하여 리더보드 0.782 돌파 시도.
현재 모델의 한계를 돌파하기 위해 학습률(lr) 스케줄러를 ReduceLROnPlateau 등으로 변경하여 더 세밀한 수렴 시도.

---

### Experiment 14: Simple Ensemble (1:1:1) of Top MLP Models
Date: 2026-02-01
Model: Ensemble (m1 + m2 + m3)
File: 08_Simple_Ensemble_111.ipynb
Status: Failed to Surpass Baseline

1. 시도 내용 (Intended Strategy)
목표: 서로 다른 최적화 기준(Loss vs AUC)과 구조(180 vs 288)를 가진 상위 3개 모델을 단순 평균하여 리더보드 점수 갱신 시도.
대상 모델:
m1 (0.78116): Baseline (Loss-based, 180-32)
m2 (0.78108): Recovery (AUC-based, 180-32)
m3 (0.78093): Optuna (AUC-based, 288-64)

방법: 각 모델의 예측값(voted)을 1:1:1 산술 평균.

2. 분석 결과 (Analysis Results)
상관관계(Correlation): - m1-m2: 0.99969
                       m1-m3: 0.99938
                       m2-m3: 0.99936
서브리더보드 성능:
Ensemble: 0.782612 (Public) / 0.779847 (Private)
Best Single (m1): 0.782615 (Public) / 0.779909 (Private)

결과: 앙상블 점수가 단일 최고점 모델(m1)보다 낮게 형성됨.

3. 분석 및 향후 계획 (Analysis & Next Steps)
진단: 모델 간 상관관계가 0.999 이상으로 지나치게 높아 앙상블을 통한 오차 상쇄 효과가 발생하지 않음. 오히려 성능이 낮은 모델이 섞이면서 최고점 모델의 성능을 희석(Dilution)시키는 결과 초래.
조치: 단순 산술 평균 대신, 순위 기반의 Rank Averaging이나 Private 점수가 더 높았던 모델(m2)에 가중치를 두는 Weighted Blending 검토 필요.
계획: 머신러닝 모델 사용이 불가능하므로, MLP 내부에서 **Activation Function 변경(ELU, SELU 등)**이나 **입력 피처의 도메인 결합(Feature Interaction)**을 통해 모델 간의 상관관계를 강제로 낮춘 '이질적 MLP'를 생성하여 재앙상블 시도.

---

### Experiment 15: Diversity-Driven Ensemble (Focal-SNN)
Date: 2026-02-sed + m4: Focal-SNN)
File: 09_Diversity_Focal_S01
Model: Ensemble (m1: Loss-baNN_Model.ipynb / Experimental_Blend_92_08.csv
Status: Success (New Public/Private SOTA)

1. 시도 내용 (Intended Strategy)
- 가설: 모델 간 상관관계($Correlation$)를 낮추면, 단일 모델의 성능이 낮더라도 앙상블 시 오차 상쇄 효과로 인해 전체 성능이 향상될 것이다.
- 실행: 기존 LeakyReLU + BatchNorm + BCE 조합을 탈피하여, SELU + AlphaDropout + Focal Loss 구조의 이질적 MLP(m4)를 신규 학습.
- 앙상블: 성능 차이를 고려하여 92:8(m1:m4) 비율의 초보수적 가중 평균 수행.

2. 검증 수치 및 상관관계
m4 (SNN-Focal) 성능: Mean Validation AUC 0.77077 (기준치 0.779 미달)
상관계수 (m1 vs m4): 0.9751기존 앙상블(Exp 14)의 0.999 대비 다양성이 약 25배 확보됨.

서브리더보드 결과:
Ensemble (92:8): **0.782622** (Public) / **0.779936** (Private)
Best Single (m1): 0.782615 (Public) / 0.779909 (Private)
결과: Public 점수와 Private 점수 모두에서 기존 최고점 경신 성공.

3. 분석 및 향후 계획 (Analysis & Next Steps)
진단: 0.770점대의 낮은 성능임에도 불구하고 0.975라는 '상대적으로 낮은' 상관관계가 앙상블 시너지의 핵심 동력으로 작용함. 특히 Focal Loss가 포착한 'Hard Sample'에 대한 정보가 m1의 예측을 미세 보정한 것으로 판단됨.
조치: 0.781대의 높은 점수를 유지하면서도 Private 방어력이 검증된 모델(m2)을 추가 결합하여 안정성 극대화 필요.
계획: 마지막 남은 제출 기회 1회를 위해 **[Public SOTA(m1) + Private SOTA(m2) + Diversity(m4)]**의 6:3:1 삼각 가중 앙상블을 최종 시도함.

---

| 번호 | 유형 | 핵심 전략 | 비고 |
| :--- | :--- | :--- | :--- |
| 03 | **Single MLP (180-32)** | Loss 기반 최적화 | Baseline (0.78116) |
| 12 | **Single MLP (180-32)** | AUC 기반 최적화 | Private 최적화 모델 |
| 13 | **Optuna MLP (288-64)** | 구조 자동 탐색 | 구조 다양성 확보 |
| 15 | **Single SNN (SELU)** | Focal Loss | Diversity 모델 (0.9751 Corr) |
| 10 | **Ensemble** | Triangular Weight (6:3:1) | 현재 최고점 갱신용 |

---

### Experiment 16: Triangular Weighted Ensemble (6:3:1)
Date: 2026-02-01
Model: Ensemble (m1: 60%, m2: 30%, m4: 10%)
File: 16_Triangular_Ensemble_631_.csv
Status: Success (Private SOTA / Public Minor Drop)

1. 분석 결과
Public: 0.78255 (Exp 15 대비 -0.00007)
Private: 0.77997 (Exp 15 대비 +0.00004)

결론: Public 점수의 미세 하락은 아쉽지만, 실제 순위 결정에 중요한 Private 점수가 올랐으므로 모델의 일반화 성능은 강화됨.

---

### Experiment 17: OOF (Out-of-Fold) System Infrastructure
Date: 2026-02-01
Model: m1(Exp11-SOTA), m2(Exp12-AUC), m4(Exp15-SNN)
File: 17_OOF_Generation_m1_m2_m4.ipynb
Status: Completed (System Built)

1. 시도 내용 (Intended Strategy)
- 가설: 각 모델의 학습 데이터에 대한 검증 예측값(OOF)을 확보하면, 메인 리더보드 제출 없이도 로컬에서 수학적 가중치 최적화 및 앙상블 성능 예측이 가능할 것이다.
- 실행: m1, m2, m4의 원본 하이퍼파라미터(Batch Size 72/128, Epoch 48/60, Repeat 5, Fold 7)를 엄격히 준수하여 실 제출물과 정합성이 일치하는 OOF 데이터(.npy) 생성.
- 조치: BatchNorm 연산 시 마지막 배치 데이터 부족 에러를 해결하기 위해 `drop_last=True` 옵션 적용.

2. 검증 수치 (Local OOF AUC Results)
- m1 (Exp11-SOTA) OOF AUC: **0.77212**
- m2 (Exp12-AUC) OOF AUC: **0.77233**
- m4 (Exp15-SNN) OOF AUC: **0.77330**
- 결과: 단일 모델 기준 로컬 검증 성능은 SNN(m4)이 가장 우수함을 확인.

3. 분석 및 향후 계획 (Analysis & Next Steps)
- 진단: 로컬 AUC 지표 확보를 통해 "감"에 의존하던 앙상블에서 벗어나 데이터 기반의 의사결정 체계 구축. SNN 모델의 높은 OOF 점수는 앙상블 시 강력한 보정 엔진이 될 가능성을 시사함.
- 조치: 파일명에 AUC 점수를 명기하여(`[모델명]_AUC_[점수].npy`) 실험 관리 효율성 증대.
- 계획: **Exp 18**에서 `scipy.optimize`를 활용하여 위 3개 모델의 로컬 AUC를 극대화하는 수학적 최적 가중치 도출 시도.

---

### Experiment 18: Local Ensemble Weight Optimization
Date: 2026-02-01
Model: Ensemble (m1: 33%, m2: 33%, m4: 34%)
File: 18_Local_Ensemble_Optimization.ipynb
Status: Success (Found Mathematical Optimal Weights)

1. 시도 내용 (Intended Strategy)
- 가설: 직관에 의존한 6:3:1 가중치보다 수학적 최적화(SLSQP)를 통해 도출된 가중치가 로컬 AUC 및 실제 성능을 더 효과적으로 개선할 것이다.
- 실행: Exp 17에서 생성된 m1, m2, m4의 OOF(.npy) 데이터를 활용하여 `scipy.optimize` 기반의 가중치 최적화 수행.

2. 검증 수치 및 결과
- 기존 가중치 (6:3:1) Local AUC: 0.77293
- 최적 가중치 (33:33:34) Local AUC: **0.77351** (약 +0.00058 개선)
- 도출된 비율: m1(0.3300) : m2(0.3299) : m4(0.3400)

3. 분석 및 향후 계획 (Analysis & Next Steps)
- 진단: 세 모델의 가중치가 거의 균등하게 수렴함. 이는 m4(SNN)가 단순히 성능이 낮은 모델이 아니라, m1/m2와 낮은 상관관계를 가지며 강력한 보완 역할을 수행하고 있음을 정량적으로 증명함.
- 조치: 발견된 수학적 황금 비율을 적용하여 메인 리더보드 타격용 최종 앙상블 파일 생성 준비.
- 계획: **Exp 19**에서 최적 비율 기반의 최종 제출물 생성.

---

### Experiment 19: Mathematical Optimal Ensemble (33:33:34)
Date: 2026-02-01
Model: Ensemble (m1: 33%, m2: 32.99%, m4: 34%)
File: 19_Optimal_Ensemble_3334_.csv
Status: Completed (Local SOTA / Leaderboard Minor Drop)

1. 시도 내용 (Intended Strategy)
- 가설: Exp 18에서 수학적으로 도출된 로컬 최적 가중치(33:33:34)를 적용하면, 직관적 6:3:1 조합보다 실제 성능(AUC)이 개선될 것이다.
- 실행: 가장 높은 로컬 OOF AUC(0.77351)를 기록한 비율을 최종 테스트 세트에 적용하여 서브 리더보드 점수 확인.
- 특징: m4(SNN) 모델의 비중을 10% -> 34%로 대폭 상향하여 m1, m2와 동등한 영향력을 부여함.

2. 검증 수치 및 결과
- 로컬 OOF AUC: **0.77351** (전체 실험 중 로컬 최고점)
- 서브 리더보드 결과:
    - Public: **0.7825735829** (Exp 16 대비 +0.000022 상승)
    - Private: **0.7798704512** (Exp 16 대비 -0.000103 하락)
- 결과: Public 점수는 미세하게 올랐으나, 순위에 결정적인 Private 점수가 하락하며 일반화 성능 저하 확인.

3. 분석 및 향후 계획 (Analysis & Next Steps)
- 진단: 로컬 AUC가 높음에도 Private 점수가 떨어진 것은 SNN(m4) 모델이 학습 데이터의 특정 노이즈까지 학습하여 실제 테스트셋(Private)의 분포를 벗어났을(Overfitting) 가능성이 큼. 34%의 비중은 SNN의 개성을 반영하기에 너무 과도했음이 입증됨.
- 조치: 향후 앙상블 시 m4의 비중을 다시 10% 내외로 하향 조정하거나, 모델 자체의 규제(Regularization) 강화 필요.
- 계획: 단순 가중치 조절의 한계를 인지. **Exp 20**에서는 모델의 예측치를 다시 학습 데이터로 사용하는 **Stacking(Meta-Learning)** 시스템을 구축하거나, 파생 변수(Feature Engineering) 생성을 통해 m1 자체의 체력을 키우는 전략으로 선회.

---

### Experiment 20: Feature Engineering (FE) Model
Date: 2026-02-01
Model: m5 (MLP + FE + Cleaning)
Status: Success (New Single Model OOF SOTA)

1. 시도 내용 (Intended Strategy)
- 가설: 이상치(시간 빌런, 가족 빌런)를 정제하고, 심리학적 지표(Mach_Score)를 추가하면 모델의 근본적인 판별 능력이 상승할 것이다.
- 실행: 99% Winzorization 적용, 역채점 기반 Mach_Score 및 응답 시간 통계 피처 생성 후 MLP 학습.

2. 검증 수치 및 결과
- m5 OOF AUC: **0.77338** (단일 모델 중 최고 기록)
- 특징: m4(0.77330)보다 높은 점수를 기록하며, 데이터 정제의 중요성 입증.

3. 분석 및 향후 계획 (Analysis & Next Steps)
- 진단: 파생 변수가 유의미한 정보를 제공함. 특히 이상치 제거로 인해 모델이 훨씬 안정적인 Gradient를 확보함.
- 계획: m1, m2, m4, m5를 모두 결합하는 4종 앙상블(Exp 21) 진행.

---

### Experiment 21: Optuna-based 4-Way Ensemble Optimization
Date: 2026-02-01
Model: 4-Way Ensemble (m1, m2, m4, m5)
File: 21_Optuna_Ensemble_4Way.ipynb
Status: Success (New Local SOTA: 0.77437)

1. 시도 내용 (Intended Strategy)
- 가설: 모델의 개수가 늘어남에 따라(4종), 단순 가중치 합보다 베이지안 최적화(Optuna)를 통한 정밀한 가중치 배분이 더 높은 AUC를 산출할 것이다.
- 실행: m5 학습 시 제거된 이상치를 반영하여 m1, m2, m4의 인덱스를 재정렬(Alignment)한 후, 200회의 Trial을 통해 최적 가중치 도출.
- 모델 구성: 
    - m1: Exp11-SOTA (0.78116)
    - m2: Exp12-AUC (0.78108)
    - m4: Exp15-SNN (Diversity 모델)
    - m5: Exp20-FE (Feature Engineering 모델)

2. 검증 수치 및 결과 (Local AUC)
- **최적화된 로컬 AUC: 0.77437** (Exp 18 대비 +0.00086 개선)
- **도출된 원본 가중치 (Raw Weights):**
    - w1 (m1): 0.0210
    - w2 (m2): 0.0913
    - w4 (m4): 0.9788
    - w5 (m5): 0.5170
- **정규화 가중치 (비중 %):**
    - m4(SNN): **60.9%** / m5(FE): **32.1%** / m2(AUC): **5.7%** / m1(SOTA): **1.3%**

3. 분석 및 향후 계획 (Analysis & Next Steps)
- 진단: **SNN(m4)의 압도적 비중(61%)**과 **FE(m5)의 강력한 기여(32%)**가 확인됨. 기존 에이스였던 m1, m2의 비중이 극도로 낮아진 것은, m4의 구조적 다양성과 m5의 정제된 피처가 기존 모델의 성능을 상회하고 있음을 의미함.
- 조치: 발견된 수학적 황금 비율(약 1 : 6 : 61 : 32)을 적용하여 메인 리더보드 0.79 돌파를 위한 최종 제출물 생성.
- 계획: **Exp 22**에서 해당 가중치를 적용한 최종 Submission 파일 생성 및 제출.

---

### Experiment 22: Optuna-Weighted Final Ensemble
Date: 2026-02-01
File: 22_Optuna-Weighted Ensemble.ipynb
Status: Fail (Public 0.718 / Private 0.715)

1. 분석 및 진단 (Post-Mortem)
- **원인:** OOF 검증 데이터에 대한 과적합으로 인해 m4(SNN)에 비정상적으로 높은 가중치(61%)가 부여됨.
- **결과:** 실제 테스트 데이터와의 분포 차이로 인해 점수가 0.07 가량 폭락하는 참사 발생.
- **교훈:** Optuna 가중치가 특정 모델에 과도하게 쏠릴 경우(예: 특정 모델 50% 이상), 이는 일반화 성능 저하의 강력한 신호이므로 즉시 폐기해야 함.

2. 조치 사항
- 확률값 직접 평균 대신 **Rank Averaging** 기법 도입 결정.
- 검증된 SOTA 모델(m1, m2)의 비중을 80% 이상으로 강제 고정하여 베이스라인 복구 시도.

---

### Experiment 23: Safety Recovery Rank Ensemble (Simulation)
Date: 2026-02-01
Model: Rank Ensemble (m1:0.4, m2:0.4, m5:0.2)
file: 23_Safety_Recovery_Rank_82.ipynb
Status: Completed (Simulation Only - Not Submitted)

1. 시도 내용 (Intended Strategy)
- 가설: 0.71 참사를 복구하기 위해 확률값 대신 순위(Rank)를 사용하고, 검증된 모델(m1, m2)의 비중을 80%로 높여 안전성을 확보한다.
- 실행: m1_rank(0.4) + m2_rank(0.4) + m5_rank(0.2) 조합으로 로컬 AUC 시뮬레이션 수행.

2. 검증 수치 및 결과
- **최종 로컬 AUC: 0.77327**
- 분석: m5 단일 모델(0.77353)보다 낮은 수치를 기록함. Rank Averaging이 변동성을 줄여주지만, 성능이 뛰어난 m5의 신호를 m1, m2가 다소 희석(Dilution)시키는 효과가 발생함.

3. 향후 계획 (Next Steps / Pivot)
- 진단: 단순 Rank 앙상블로는 m5의 잠재력을 온전히 끌어내지 못함. 
- 조치: 제출권 낭비를 막기 위해 해당 버전은 제출하지 않기로 결정.
- 계획: m5 모델의 피처 엔지니어링을 더 강화하거나, m5와 m1/m2의 가중치 비율을 다시 조정하여 0.774 이상의 로컬 AUC가 나올 때까지 제출 보류.

---

## Experiment 24: Deep Feature Engineering (DFE) v1
- **Date:** 2026-02-02
- **Model:** m6 (Deep MLP: 288 → 64 → 1)
- **File:** 24_Deep_Feature_Engineering_v1.ipynb
- **Status:** FAILED (Performance Degradation)

### 1. 시도 내용 (Intended Strategy)
- **가설:** 단순히 문항을 합치는 것을 넘어, 응답자의 '심리적 태도'와 '시간 배분 전략'을 수치화하면 모델의 변별력이 극대화될 것이다.
- **추가 파생 변수:**
    - **심리학적 하위 척도:** `Views_Score` (냉소), `Tactics_Score` (조종 전략)
    - **응답 모순 지표:** `Conflict_Index` (QqA와 QcA의 응답 괴리율)
    - **상대적 시간 비중 (20개):** 각 문항의 응답 시간을 전체 소요 시간으로 나눈 `Relative_E_time` 시리즈
- **모델 구조:** 늘어난 피처 수(약 120개)를 수용하기 위해 은닉층 노드를 **288-64**로 확장.

### 2. 검증 성능 (Validation Results)
- **최종 OOF AUC:** **0.77275**
- **비교 지표:** m5(0.77353) 대비 **-0.00078** 하락
- **진단:** 단일 모델 성능이 기존 SOTA(m1, m2) 수준 이하로 떨어짐에 따라 리더보드 제출 보류 결정.

### 3. 분석 및 패착 원인 (Analysis & Post-Mortem)
- **피처 희석 (Feature Dilution):** 20개의 상대적 시간 비중 피처들이 서로 높은 상관관계를 가지며 대거 유입됨에 따라, 모델이 `Mach_Score`와 같은 핵심 신호(Signal)에 집중하지 못하고 노이즈를 학습함.
- **다중공선성 문제:** 시간 비중 피처들은 합이 1이 되는 구조적 특성상 강한 상관관계를 가지며, 이는 MLP 가중치 학습의 불안정성을 초래함.
- **모델 용량 과잉:** 노드 수를 늘린 것이 데이터의 본질적 패턴이 아닌 지엽적 노이즈를 암기하는 과적합(Overfitting)으로 이어짐.

### 4. 향후 계획 (Next Steps / Pivot)
- **전략적 다이어트:** 통계적으로 유의미했던 `Conflict_Index`와 `Sub-scales`만 남기고 20개의 시간 비중 피처를 과감히 삭제.
- **구조 원복:** 모델 구조를 가장 안정적이었던 **180-32** 또는 **256-32**로 회귀하여 일반화 성능 확보.
- **신규 지표:** 응답의 일관성을 단일 변수로 보여주는 **'응답 분산(Variance)'** 피처 도입 검토 (Exp 25).

---

## Experiment 25: Refined Deep Feature Engineering (DFE) v2
- **Date:** 2026-02-02
- **Model:** m7 (Refined MLP: 256 → 32 → 1)
- **File:** `25_Refined_Deep_Feature_Engineering.ipynb`
- **Status:** FAILED (Direction Mismatch)

### 1. 사용 피처 (Feature Engineering Strategy)
- **전략:** "Less is More" - 노이즈를 유발하는 변수를 과감히 삭제하고 심리학적 본질에 집중.
- **삭제:** `Relative_E_time` (Exp 24에서 실패 원인이 된 20개 시간 비중 피처 전량 제거).
- **유지:** `Views_Score`, `Tactics_Score`, `Conflict_Index` (검증된 심리 지표).
- **신규:** **`Q_Var`** (20개 질문 답변의 분산) - 응답자가 얼마나 극단적으로 혹은 일관되게 답했는지를 보여주는 강력한 성실도 지표.
- **이상치:** `familysize > 50` 제거 및 시간 데이터 상위 1% Winzorization 유지.

### 2. 검증 성능 (Validation Results)
- **최종 OOF AUC:** **0.77375** 👑
- **모델 간 비교 (Local AUC):**
    - **m1 (LB 0.78116):** 0.77212 (대비 **+0.00163** 상승)
    - **m5 (기존 최고점):** 0.77353 (대비 **+0.00022** 상승)
    - **m6 (실패작):** 0.77275 (대비 **+0.00100** 반등)
- **진단:** 불필요한 변수를 걷어내고 `Q_Var`를 도입한 것이 역대 최고 로컬 점수로 이어짐.
- **Public LB Score:** 0.2179331762 (Inverted AUC)
- **Private LB Score:** 0.2206507587 (Inverted AUC)

### 3. 분석 및 향후 계획 (Analysis & Next Steps)
- **원인 분석:** - 모델 m7의 성능(AUC 0.782 수준)은 확보되었으나, 예측값의 방향이 리더보드 기준과 정반대로 생성됨.
    - m1(작은 값=투표)과 m7(큰 값=투표)의 불일치로 인해 앙상블 및 단일 제출 시 점수 폭락 발생.
- **교훈:** 제출 전 반드시 베이스라인 모델(m1)과의 상관계수(Correlation)가 음수인지 양수인지 확인하여 방향을 일치시켜야 함.
- **계획:** 남은 제출 기회(2회) 중 1회를 활용하여 **m7 단일 모델**을 제출, 0.71 참사를 복구하고 실제 리더보드 실력을 확인. 이후 최고점 모델(m1/m2)과의 최종 Rank 앙상블 검토.

---

## Experiment 26: Master Rank Blending (m1 + m7)
- **Date:** 2026-02-02
- **Model:** Rank Ensemble (m1: 0.5, m7: 0.5)
- **Status:** **Success (Project SOTA achieved)**

### 1. 주요 전략 및 해결 과제
- **데이터 정합성:** m1(45,529행)과 m7(45,490행)의 행 수 차이를 인덱스 기반 재정렬(reindex)로 매칭하여 데이터 밀림 현상 방지.
- **방향성 통일:** 진단 코드로 두 모델 모두 정방향(Large=Voted)임을 확인. 제출 시 m1 방식(Small=Voted)인 `2.0 - rank_p`를 적용하여 방향성 일치.
- **기법:** 전처리와 구조가 다른 두 모델의 확률 스케일을 표준화하기 위해 Rank Averaging(5:5) 적용.

### 2. 최종 성과 분석
- **메인 리더보드 (Public):** **0.78150** (m1+m4 대비 +0.00031 상승, 목표치 0.78141 초과 달성)
- **서브 리더보드 (Public):** **0.78295** (역대 최고점)
- **서브 리더보드 (Private):** **0.78021** (최초 0.780 돌파 및 일반화 성능 입증)
- **결론:** 로컬 성능이 가장 높았던 exp27(4:6)보다 exp26(5:5)이 실제 리더보드에서 더 높은 점수를 기록함. 이는 모델 간 균형이 무너질 때보다 동등하게 결합될 때 변동성에 더 강한(Generalizable) 모델이 생성됨을 시사함.

### 3. 프로젝트 마무리
- 0.71/0.21 점수 폭락 위기를 정밀한 인덱스 정렬과 방향성 분석으로 극복하고 최종 0.7815 고지 점령.
- 최종 모델: `26_m1_m7_Rank_Ensemble_0.77364.csv`

---

## Experiment 27: Fine-tuned Rank Blending (m1:m7 = 4:6)
- **Date:** 2026-02-02
- **Model:** Rank Ensemble (m1: 0.4, m7: 0.6)
- **Status:** Completed (Lower than Exp 26)

### 1. 결과 분석
- **Local AUC:** 0.77379 (역대 최고점)
- **리더보드 역전:** 로컬 성능은 상승했으나 서브 리더보드(Pub/Priv) 점수는 하락. 이는 m7 모델의 특성이 특정 검증 셋에 더 최적화되었음을 시사함.
- **결론:** 모델 간 비중이 균등한 **5:5 Rank Averaging(Exp 26)**이 가장 우수한 일반화 성능을 보임.

### 2. 최종 결정
- 메인 리더보드 타격을 위한 최종 모델로 **Exp 26**을 선정.

---

### 모델별 최종 성적 요약표 (최종 업데이트: 2026-02-02)

| 모델 ID | 주요 특징 | Local AUC | 메인 LB (Pub) | 서브 LB (Pub) | 서브 LB (Priv) | 비고 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| m1 | 기본 MLP (Base) | 0.77212 | 0.78116 | 0.78261 | 0.77990 | 베이스라인 |
| m2 | AUC 최적화 모델 | 0.77312 | 0.78108 | 0.78225 | 0.78004 | - |
| m1+m4 | 92:8 가중 블렌딩 | - | 0.78119 | 0.78262 | 0.77993 | 기존 최고점 |
| exp16 | Triangular (6:3:1) | - | - | 0.78255 | 0.77997 | - |
| m7 | Refined Deep FE | 0.77375 | - | 0.21793 | 0.22065 | Inverted 제출 |
| **exp26** | **m1+m7 Rank (5:5)** | **0.77364** | **0.78150** | **0.78295** | **0.78021** | **Main LB SOTA** |
| exp27 | m1+m7 Rank (4:6) | 0.77379 | - | 0.78286 | 0.78014 | Local SOTA |

---

## Experiment 28: FT-Transformer with m7 Refined FE (m8)
- **Date:** 2026-02-03
- **Model:** m8 (FT-Transformer via `rtdl`)
- **Status:** Completed (Lower single performance than expected)

### 1. 사용 피처 (Feature Engineering)
- **m7 정제 로직 계승:** `familysize > 50` 제거 및 질문 답변(`Q_A`)의 표준편차가 0인 불성실 응답자 제거.
- **DFE 피처 도입:** 응답 일관성을 나타내는 `Q_Var`, 역채점 기반 `Mach_Score`, 문항 간 모순을 측정하는 `Conflict_Index` 적용.
- **스케일링:** `exp03`에서 성공적이었던 `Q_E` 로그 변환 및 수치형 데이터 전체에 대한 `StandardScaler` 적용.

### 2. 하이퍼파라미터 (Hyperparameters)
- **구조 최적화:** `exp03` Trial 13의 성공 파라미터인 `d_block=128`, `n_blocks=2`, `n_heads=8` 준수.
- **학습 설정:** `lr=0.000975`, `batch_size=256`, `epochs=25` 설정.
- **Optimizer:** AdamW (Weight Decay=0.01).

### 3. 검증 성능 (Validation Results)
- **Fold 1:** 0.76268
- **Fold 2:** 0.77388
- **Fold 3:** 0.77160
- **Fold 4:** 0.76268
- **Fold 5:** 0.77115
- **Fold 6:** 0.76522
- **Fold 7:** 0.77198
- **Final OOF AUC: 0.76664**

### 4. 분석 및 계획 (Analysis & Next Steps)
- **진단:** `m7`의 강력한 피처들을 이식했음에도 불구하고, 단일 모델 AUC는 `exp03`(0.7760)이나 `m7`(0.7737)의 기록에 미치지 못함. 수치형 피처의 복잡도가 Transformer의 어텐션 학습에 노이즈로 작용했을 가능성 확인.
- **의의:** 단일 성능은 낮으나, MLP 기반인 `m1`, `m7`과는 완전히 다른 Attention 메커니즘을 사용하므로 앙상블 시 오차 상쇄 효과(Diversity)를 기대할 수 있음.
- **계획:** `m1(안정성) + m7(FE 엣지) + m8(구조 다양성)`을 결합한 **exp29: 3-Way Rank Ensemble**을 통해 메인 리더보드 0.79 돌파 시도.

---

## Experiment 29: 3-Way Rank Ensemble (m1 + m7 + m8) - Direction Corrected
- **Date:** 2026-02-03
- **Status:** FAILED (Significant CV/LB Gap)

### 1. 결과 분석
- **Local AUC:** 0.77387 (최고점 경신)
- **Public LB:** 0.77137 (Exp 26 대비 -0.010 하락)
- **Private LB:** 0.77199 (Exp 26 대비 -0.008 하락)

### 2. 패착 원인
- **m8의 낮은 일반화 성능:** 단독 성능이 낮은 m8을 1:1:1 비율로 섞으며 앙상블 전체의 리더보드 점수 하락 초래.
- **피처 부작용:** m7의 DFE 피처들이 트랜스포머 모델에서 과적합을 유발하여 테스트셋 예측력을 저하시킴.

---

## Experiment 30: Weighted Rank Ensemble (m1 + m7 + m8) - Final Test
- **Date:** 2026-02-03
- **Status:** CRITICAL FAILURE (Discarding m8)

### 1. 결과 분석
- **Local AUC:** 0.77384 (수학적 최적화 결과)
- **Public LB:** 0.73934 (Exp 26 대비 -0.042 폭락)
- **Private LB:** 0.74466 (Exp 26 대비 -0.035 폭락)

### 2. 결론 및 향후 계획
- **m8의 정체:** FT-Transformer는 로컬 검증 셋에만 극도로 과적합되며, 테스트 셋에 대해서는 노이즈를 생성함.
- **결정:** 앙상블에서 m8을 완전히 제외함. 다시 m1 + m7의 2-Way 체제로 복귀하여 0.79 고지를 노림.

---

---

## Experiment 38: Master Rank Ensemble (5:5 Balanced)
- **Date:** 2026-02-03
- **Model:** Rank Ensemble (Exp 26: 0.5, Exp 37: 0.5)
- **Status:** Completed (Sub LB Validation)

### 1. 시도 내용 (Intended Strategy)
- **가설:** 전처리가 다른 두 모델(m1+m7 기반 Exp 26, PL 기반 Exp 37)을 동등하게 결합하면 특정 모델의 오차를 상쇄하고 일반화 성능이 극대화될 것이다.
- **실행:** Exp 26과 Exp 37의 예측치를 순위화(Rank)하여 5:5 비율로 산술 평균 수행.

### 2. 결과 분석 (Result Analysis)
- **진단:** 서브 리더보드 기준, Exp 26 단독 성적보다 하락함.
- **원인:** Pseudo-Labeling(Exp 37)의 노이즈가 Exp 26의 정밀도를 희석(Dilution)시키는 효과가 발생함.

### 3. 검증 성능 (Validation Results)
- **서브 리더보드 (Public):** 0.7826128286
- **서브 리더보드 (Private):** 0.7800646157

---

## Experiment 39: Weighted Rank Ensemble (7:3)
- **Date:** 2026-02-03
- **Model:** Rank Ensemble (Exp 26: 0.7, Exp 37: 0.3)
- **Status:** Completed (Main LB Impact Test)

### 1. 시도 내용 (Intended Strategy)
- **가설:** Exp 38의 실패를 바탕으로, 검증된 SOTA 모델인 Exp 26의 비중을 높여(70%) 안정성을 확보하고 Exp 37의 새로운 시각을 30%만 반영한다.
- **실행:** Exp 26과 Exp 37의 Rank 가중 평균 (7:3) 적용.

### 2. 결과 분석 (Result Analysis)
- **진단:** 메인 리더보드 점수가 0.78144로 수렴하며 기존 SOTA(Exp 26: 0.78150)를 넘어서지 못함.
- **결론:** 단순 가중치 조절보다는 Pseudo-Labeling 데이터의 순도를 높이는(Threshold 강화) 근본적인 개선이 필요함.

### 3. 검증 성능 (Validation Results)
- **메인 리더보드 (Public):** 0.7814468895

---

## Experiment 40: Conservative Pseudo-Labeling (5% Selection)
- **Date:** 2026-02-03
- **Model:** MLP (m7 Architecture) + 5% Pseudo-Labeling
- **Status:** Success (New Local SOTA)

### 1. 시도 내용 (Intended Strategy)
- **가설:** Exp 37(10% PL)의 실패 원인은 과도한 데이터 선별로 인한 노이즈임. 선별 임계치를 상위/하위 5%로 강화하여 데이터의 순도를 높이면 일반화 성능이 개선될 것이다.
- **실행:** 하이브리드 피처(시너지 지표) 시도 후 로컬 AUC 하락 확인 -> 즉시 제거 후 m7 순수 피처셋으로 회귀하여 변수 통제.
- **검증:** Pseudo-label을 제외한 원본 학습 데이터에 대해서만 AUC를 측정하는 'Pure OOF AUC' 시스템 도입.

### 2. 결과 분석 (Result Analysis)
- **진단:** 로컬 신기록 달성. 5% 임계치 강화로 노이즈가 효과적으로 억제됨. 
- **성과:** Pure OOF AUC가 0.77376으로 상승하며 m7(0.77375)과 Exp 26(0.77364)의 기록을 모두 경신.

### 3. 검증 성능 (Validation Results)
- **로컬 검증 (Pure OOF AUC):** 0.77376
- **메인 리더보드 (Public):** (단일 제출 없음)
- **서브 리더보드 (Public/Private):** (단일 제출 없음)

---

## Experiment 41: Master Rank Ensemble (6:4)
- **Date:** 2026-02-03
- **Model:** Rank Ensemble (Exp 26: 0.6, Exp 40: 0.4)
- **Status:** Completed (Lower than Exp 26)

### 1. 시도 내용 (Intended Strategy)
- **가설:** 정제된 Exp 40을 40% 비중으로 섞어 Exp 26의 성능을 추월한다.
- **실행:** Exp 26과 Exp 40의 Rank 가중 평균 (6:4) 적용.

### 2. 결과 분석 (Result Analysis)
- **진단:** Exp 38(0.78261) 대비 점수는 상승했으나, 여전히 Exp 26의 SOTA 기록(Sub 0.78296)에는 미치지 못함. 
- **인사이트:** Exp 40의 신호는 유효하지만, 40%의 비중은 Exp 26의 최적화된 가중치를 흐트러뜨림. 더 보수적인 비중(8:2 등)이 필요함.

### 3. 검증 성능 (Validation Results)
- **서브 리더보드 (Public):** 0.7826499834
- **서브 리더보드 (Private):** 0.7801324444

---

## Experiment 42: Conservative Rank Ensemble (8:2)
- **Date:** 2026-02-03
- **Model:** Rank Ensemble (Exp 26: 0.8, Exp 40: 0.2)
- **Status:** Completed (Very close to SOTA)

### 1. 시도 내용 (Intended Strategy)
- **가설:** Exp 41(6:4)의 하락을 바탕으로, Exp 40의 비중을 20%로 줄여 Exp 26의 안정성을 보존하면서 PL의 엣지만 추출한다.
- **실행:** Exp 26과 Exp 40의 Rank 가중 평균 (8:2) 적용.

### 2. 결과 분석 (Result Analysis)
- **진단:** 서브 LB 기준, Exp 26의 기록에 거의 근접함(Public 차이 0.00013 / Private 차이 0.000001). 
- **인사이트:** 가중치를 줄일수록 점수가 우상향하는 추세가 뚜렷함. 이는 Exp 40이 고유한 정보를 담고 있으나 그 비중이 20%를 넘어가면 Public LB의 분포를 흔들 수 있음을 의미함.
- **결정:** 메인 리더보드 제출 시에는 8.5:1.5 또는 9:1의 '극보수적 가중치'를 사용하여 최고점 경신을 노림.

### 3. 검증 성능 (Validation Results)
- **서브 리더보드 (Public):** 0.7828269912
- **서브 리더보드 (Private):** 0.7802140570

---

## Experiment 43: Refined Master Ensemble (8.5:1.5)
- **Date:** 2026-02-03
- **Model:** Rank Ensemble (Exp 26: 0.85, Exp 40: 0.15)
- **Status:** Completed (Extremely close to SOTA)

### 1. 시도 내용 (Intended Strategy)
- **가설:** Exp 42(8:2)의 상승세를 바탕으로 가중치를 8.5:1.5로 더 보수적으로 조정하면, Exp 26의 안정성을 지키면서 Exp 40의 엣지가 최고점을 경신할 것이다.
- **실행:** Exp 26과 Exp 40의 Rank 가중 평균 (8.5:1.5) 적용.

### 2. 결과 분석 (Result Analysis)
- **진단:** 메인 LB 기준 0.781488 기록. SOTA(0.781510)와 단 0.000022 차이로 근접함.
- **인사이트:** 가중치를 줄일수록 SOTA에 수렴하는 속도가 빨라짐. 이는 Exp 40의 신호가 매우 날카롭지만, 동시에 Exp 26이 가진 데이터 분포의 정교함이 그만큼 강력함을 의미함.
- **결정:** 마지막 기회는 더 극단적인 보수적 가중치(9:1)를 적용하여 안전하게 최고점을 탈환함.

### 3. 검증 성능 (Validation Results)
- **메인 리더보드 (Public):** 0.7814885462
- **비고:** SOTA 대비 -0.000022 차이.

---

### Exp 43: Diversity Loss Check
* **Date:** 2026-02-03
* **Method:** Rank Ensemble (Exp 26 + Exp 40)
* **Weights:** 0.85 : 0.15
* **Rationale:** 기존 SOTA(Exp 26)에 5% Pseudo-labeling 모델(Exp 40)을 미세하게 섞어 점수 향상 시도.
* **Result:**
    * **LB Score:** 0.78148 (Drop from 0.78151)
* **Analysis:** 9:1에 가까운 방어적 비율임에도 점수 하락. m7 계열의 비중이 과해지면서 m1과의 다양성 균형이 깨진 것으로 판단됨.

---

### Exp 44 (Skipped): 5:5 Jumper Strategy
* **Date:** 2026-02-03
* **Method:** Rank Ensemble (m1 + Exp 40)
* **Weights:** 0.5 : 0.5
* **Rationale:** m7을 성능이 더 높은 Exp 40으로 완전히 교체하여 시너지 극대화 노림.
* **Local AUC Check:** 0.77362 (Lower than Exp 26's 0.77364)
* **Result:** **Not Submitted**
* **Analysis:** 로컬 시뮬레이션 결과, Exp 40의 Pseudo-labeling이 m1의 일반화 성능을 오염시키는 '마이너스 시너지' 확인. 제출 전 폐기.

---

### Exp 44 (Final): Grand Trinity
* **Date:** 2026-02-03
* **Method:** 3-Way Rank Ensemble (m1 + m2 + m7) with Double-Rank Scale Restoration
* **Weights:** 0.3(m1) : 0.3(m2) : 0.4(m7)
* **Rationale:** * 17번 노트북의 m2(AUC-based)를 재소환하여 m1(Loss-based)과 결합. 
    * 25번 노트북 로직을 이용해 39행의 인덱스 불일치(45529 vs 45490)를 `reindex`로 해결.
    * 앙상블 후 무너진 표준편차를 복구하기 위해 Double-Rank 기법 적용.
* **Local AUC Check:** **0.77371** (Highest Local SOTA)
* **Result:**
    * **LB Score: 0.7700287945 (Massive Drop)**
* **Critical Failure Analysis:** * **Index Alignment Failure:** Double-Rank 과정에서 최종 순위를 m1의 값 분포에 매핑할 때, 행(Row)과 ID의 정렬을 명시적으로 보장하지 못함. 
    * **Data Shuffling:** 결과적으로 예측값들이 무작위로 뒤섞인 채 제출되어 성능이 'Random Guess' 수준으로 폭락함. 
    * **Lesson:** 앙상블 기법이 복잡해질수록 데이터 정렬(Alignment) 무결성 검증이 모델 성능보다 수만 배 중요함을 확인.

---

