# Voted Prediction Project: Final Analysis Report

## 1. 개요 (Overview)
본 프로젝트는 정형 데이터(Tabular Data) 예측 문제에서 GBDT(CatBoost, LGBM 등) 머신러닝 모델을 배제하고, 오직 **딥러닝(Deep Learning)** 기법만으로 성능을 극대화하는 것을 목표로 했습니다.
Exp 01부터 Exp 50까지 MLP, SNN, Transformer 등 다양한 아키텍처를 실험했으며, 최종적으로는 **Deep Feature Engineering**과 **Rank Ensemble** 전략을 통해 Public SOTA 갱신과 Private 방어력을 동시에 확보했습니다.

## 2. 최종 제출 모델 선정 (Final Selection)

### 제 1선택 (Best Public Score)
* **모델명:** Exp 45 (Trinity Fixed)
* **구성:** Rank Ensemble (m1 30% + m2 30% + **m7 40%**)
* **점수:** **0.78152** (Main LB SOTA)
* **선정 근거:** Main Leaderboard 기준 가장 높은 점수를 기록한 모델입니다. **m7(Deep Feature Engineering)**의 비중을 40%로 높게 설정하여 모델의 변별력(Discrimination)을 극대화했습니다. 3위와의 격차를 벌리기 위한 공격적인 카드로 활용합니다.

### 제 2선택 (Best Private Score)
* **모델명:** Exp 47 (Golden Trinity)
* **구성:** Rank Ensemble (m1 20% + **m2 50%** + m7 30%)
* **점수:** 0.78032 (Sub Private LB Best) / 0.78150 (Main LB)
* **선정 근거:** **m2(AUC Optimized)**의 비중을 50%로 대폭 늘려 과적합을 억제하고 일반화 성능(Generalization)을 강화했습니다. Sub Leaderboard의 Private 점수가 가장 높았기 때문에, 최종 순위 산정 시 순위 방어 및 상승에 가장 유리한 모델입니다.

## 3. 핵심 전략: Feature Engineering 및 이상치 처리 (Key Strategy: FE & Outlier Handling)

최고 성능을 낸 모델(Exp 45, 47)의 핵심 엔진인 **`m7` 모델**은 딥러닝이 포착하기 어려운 비선형적 패턴을 학습하기 위해 다음과 같은 고도화된 전처리를 수행했습니다.

### 3.1 정밀 이상치 처리 (Advanced Outlier Handling)
MLP 모델은 Tree 모델과 달리 이상치(Outlier)에 매우 민감하여 Gradient를 왜곡시킬 수 있습니다. 이를 방지하기 위해 엄격한 필터링을 적용했습니다.
* **가족 수(familysize) 필터링:** `familysize > 50`인 데이터는 허위 응답일 가능성이 높아 학습 데이터에서 제거했습니다.
* **불성실 응답자 제거 (Unfaithful Respondents):** 질문 답변(`Q_A`)의 표준편차(std)가 0인 경우(모든 문항을 같은 번호로 찍은 경우)를 노이즈로 간주하여 필터링했습니다. 이는 모델이 유의미한 패턴에만 집중하게 만들었습니다.
* **응답 시간 Winzorization:** `Q_E`(응답 시간) 데이터의 극단적인 값(상위 1%)을 Clipping 처리한 후 로그 변환(Log1p)을 적용하여 분포를 정규화했습니다.

### 3.2 Deep Feature Engineering (DFE)
MLP는 변수 간의 상호작용(Interaction)을 스스로 찾아내는 능력이 부족합니다. 이를 보완하기 위해 심리학적 도메인 지식을 활용한 파생 변수를 생성했습니다.
* **응답 분산 (Q_Var):** 20개 문항에 대한 답변의 분산을 계산하여 응답자의 **일관성(Consistency)**을 피처로 수치화했습니다. 이는 투표 여부와 높은 상관관계를 보였습니다.
* **마키아벨리즘 점수 (Mach_Score):** 설문의 의도에 맞춰 역채점(Reverse Scoring) 로직을 적용한 합산 점수를 생성, 단순 합계보다 정교한 성향 지표를 제공했습니다.
* **모순 지표 (Conflict_Index):** 상관관계가 높은 문항 쌍(Pair)에 대해 상반된 답변을 한 경우를 포착하여, 응답의 신뢰도를 모델에게 전달했습니다.

## 4. 기술적 회고 및 분석 (Technical Retrospective)

### 4.1 MLP 모델의 한계점 (Limitations of MLP)
* **연속적 매니폴드 가정 (Smooth Manifold):** MLP는 데이터를 연속적이고 부드러운 곡면으로 해석하는 경향이 있습니다. 그러나 본 설문 데이터는 '특정 나이대', '특정 응답'에서 결과가 계단식으로 변하는 불연속적 특성을 가집니다.
* **해결책:** 위에서 언급한 DFE를 통해 불연속적인 경계 정보를 명시적인 피처로 제공함으로써 한계를 일부 극복했습니다.

### 4.2 Transformer 계열(FT-Transformer, TabNet)의 실패 원인
이론적으로는 정형 데이터에 강력하다고 알려진 Transformer 계열이 본 프로젝트에서는 튜닝된 MLP보다 낮은 성능을 보였습니다.
* **데이터 부족 (Data Hunger):** Transformer 아키텍처가 Attention Map을 제대로 학습하기 위해서는 보통 10만 건 이상의 데이터가 필요합니다. 본 데이터셋(약 4.5만 건)은 이 모델들이 수렴하기에 부족하여 Fold 간 성능 편차가 크게 발생했습니다.
* **과적합(Overfitting):** `FT-Transformer`(Exp 28)의 경우 로컬 검증셋에 빠르게 과적합되는 경향을 보였으며, `TabNet`(Exp 09)은 하이퍼파라미터 민감도가 지나치게 높아 제한된 시간 내에 최적해에 도달하지 못했습니다.

### 4.3 향후 발전 가능성 (What if: DAE & DCN)
만약 딥러닝만으로 성능을 더 끌어올려야 한다면, 다음 기술들의 도입이 GBDT와의 격차를 줄이는 열쇠가 되었을 것입니다.
* **DAE (Denoising AutoEncoder):** 레이블이 없는 테스트 데이터까지 활용하여 비지도 사전 학습(Unsupervised Pre-training)을 수행했다면, 데이터의 내재적 구조(Manifold)를 더 잘 파악하여 MLP의 일반화 성능을 높였을 것입니다.
* **DCN (Deep Cross Network):** `m7`에서 수행한 수동 피처 엔지니어링 대신, DCN의 **Cross Layer**를 도입했다면 변수 간의 고차원 상호작용을 모델이 자동으로 학습하여 인간의 직관이 놓친 패턴까지 포착했을 가능성이 높습니다.

## 5. 종합 실험 이력 (Comprehensive Experiment History)

주요 모델의 성능 변화와 전략적 의사결정 과정을 요약한 표입니다.

| Exp ID | Model / Strategy | Local AUC | Main LB | Sub LB (Public) | Sub LB (Private) | 비고 (Remarks) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **01** | Baseline MLP (3-Layer) | 0.7673 | 0.75701 | - | 0.74789 | Overfitted Baseline |
| **02** | Optuna Optimized MLP | 0.7698 | 0.76826 | - | 0.76546 | Tuned with AdamW |
| **03** | FT-Transformer | 0.7760 | 0.77781 | - | 0.77168 | Better than early MLPs |
| **11** | **m1** (Standard Base) | 0.77212 | 0.78116 | 0.78261 | 0.77990 | **Baseline SOTA** |
| **12** | **m2** (AUC Optimized) | 0.77312 | 0.78108 | 0.78225 | 0.78004 | **Private Strongest** |
| **15** | m4 (SNN-Focal) | 0.77077 | - | - | - | Diversity Model |
| **16** | Triangular Ens (6:3:1) | - | - | 0.78255 | 0.77997 | m1+m2+m4 |
| **25** | **m7** (Refined DFE) | 0.77375 | - | 0.21793 | 0.22065 | **Local SOTA** (Inverted) |
| **26** | Rank Ens (m1+m7 5:5) | 0.77364 | **0.78150** | **0.78295** | 0.78021 | **Major Milestone** |
| **27** | Rank Ens (m1+m7 4:6) | 0.77379 | - | 0.78286 | 0.78014 | Overfitted to Local |
| **28** | m8 (FT-Transformer+FE) | 0.76664 | - | - | - | Failed to Generalize |
| **38** | Rank Ens (Exp26 + Exp37) | - | - | 0.78261 | 0.78006 | PL Noise Issue |
| **40** | m7 + 5% Pseudo-Label | 0.77376 | - | - | - | Pure OOF Check |
| **43** | Rank Ens (Exp26 + Exp40) | - | 0.78148 | - | - | 8.5 : 1.5 Ratio |
| **44** | Grand Trinity (Failed) | 0.77371 | 0.77002 | - | - | Index Misalignment |
| **45** | **Trinity Fixed** | - | **0.78152** | 0.78291 | 0.78028 | **Current Best (Public)** |
| **46** | The Quartet (+m4) | - | - | 0.78293 | 0.78014 | Diversity Failed (Noise) |
| **47** | **Golden Trinity** | - | **0.78150** | 0.78285 | **0.78032** | **Current Best (Private)** |
| **50** | Geometric Mean Shot | - | 0.78145 | - | - | Conservative Ensemble |