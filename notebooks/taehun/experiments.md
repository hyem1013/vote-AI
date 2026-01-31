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