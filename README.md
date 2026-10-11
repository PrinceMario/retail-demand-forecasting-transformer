 Explainable Context-Aware Transformer Framework for Retail Demand Forecasting

## Phase 1: Lag-Aware Sparse Transformer Replication and Evaluation

This repository contains the Phase-1 implementation of my M.Tech project on retail demand forecasting.

The work replicates and critically evaluates **Vaticinator**, a lag-aware sparse Transformer proposed by Surana et al. (2026). The model uses selected historical lag values instead of a complete dense look-back sequence, reducing temporal input size while retaining forecasting capability.

---

## Phase-1 Objectives

- Replicate the Vaticinator architecture
- Analyse temporal dependencies using ACF/PACF
- Compare sparse and dense Transformer forecasting
- Evaluate different lag-selection strategies
- Compare against statistical, ML and DL baselines
- Evaluate recursive multi-step forecasting
- Compare lag-aware and standard positional encoding
- Measure sparse-vs-dense inference efficiency
- Identify limitations that motivate Phase 2

---

## Dataset

A public retail sales dataset containing approximately **5 million transaction records** was used.

For Phase 1:

- Target: Monthly Total Revenue
- Complete monthly observations: 128
- Period: January 2010 – August 2020
- Training months: 102
- Testing months: 26
- Split: Chronological 80:20
- Maximum dense look-back: 24 months

The raw dataset is not included in this repository due to file size.

---

## Models Evaluated

- Vaticinator — Lag-Aware Sparse Transformer
- Dense Transformer
- Seasonal Naive
- SARIMA
- XGBoost
- LSTM

---

## Vaticinator Architecture

The sparse Transformer pipeline is:

**Selected historical lags → Scalar embedding → Lag-aware positional encoding → Transformer Encoder → Forecast head**

Core model configuration:

- `d_model = 32`
- `8 attention heads`
- `3 encoder layers`
- `FFN dimension = 128`
- `Adam learning rate = 1e-4`
- `Batch size = 16`
- `100 epochs`

Sparsity is introduced through **input construction**. Only selected historical lag tokens are passed to the Transformer; no separate sparse-attention mask is used.

---

## Lag Strategies Evaluated

### Paper-defined lags
`[1, 2, 7, 12]`

### ACF-selected lags
`[1, 5, 6, 11, 12, 13, 24]`

### Combined lag set
`[1, 2, 5, 6, 7, 11, 12, 13, 24]`

---

## Final Phase-1 Results

| Model | RMSE (Million) | MAPE (%) |
|---|---:|---:|
| Vaticinator — 4 lags | **266.93** | **0.3749** |
| Dense Transformer | 271.34 | 0.3979 |
| Vaticinator — 9 lags | 281.06 | 0.4108 |
| SARIMA | 299.78 | 0.4631 |
| Seasonal Naive | 338.96 | 0.4119 |
| LSTM | 342.68 | 0.4419 |
| XGBoost | 385.82 | 0.4392 |

---

## Key Finding

The strongest Phase-1 result is:

> **Lag selection matters more than lag count.**

The compact four-lag Vaticinator configuration outperformed both the 24-token Dense Transformer and the larger nine-lag sparse configuration.

This shows that sparse temporal modelling can be accurate and efficient, but its performance depends strongly on selecting appropriate historical periods.

---

## Recursive Forecasting

The 26-month recursive rollout achieved:

- RMSE: **287.63M**
- MAPE: **0.4155%**
- RMSE increase vs one-step forecasting: approximately **2.34%**

This indicates stable multi-step recursive behaviour over the held-out period.

---

## Sparse vs Dense Efficiency

Primary sparse Vaticinator:

- 9 temporal input tokens

Dense Transformer:

- 24 temporal input tokens

This gives:

- **62.5% fewer input tokens**
- approximately **28% lower median inference time** in the measured environment

---

## Base Paper Comparison

Base paper:

**Surana et al. (2026) — An autoregressive lag-aware sparse transformer framework for demand forecasting**

The paper reports:

- Vaticinator lower NRMSE than Dense Transformer
- Dense Transformer slightly lower MAPE
- reduced computation through lag-restricted sparse input

Our replication on a larger retail series shows:

- Dense outperforms the primary 9-lag Vaticinator
- the compact 4-lag Vaticinator outperforms Dense
- sparse-model performance is highly sensitive to lag selection

---

## Phase-1 Conclusion

Phase 1 validates sparse temporal modelling as a viable and efficient forecasting strategy.

However, the results show that one globally fixed lag set is not robust across forecasting settings.

> **Sparse modelling works, but the value of sparsity depends on selecting the right historical lags.**

---

## Phase-2 Direction

Phase 2 investigates:

### Context-Conditioned Adaptive Temporal Relevance

The proposed direction is:

**Candidate lag pool → Context-conditioned relevance scoring → Adaptive sparse lag selection → Transformer forecasting → Interpretable temporal relevance**

Instead of using one fixed global lag set for every forecast, Phase 2 will investigate dynamically selecting or weighting historical periods based on retail context.

---

## Repository Structure

```text
retail-demand-forecasting-transformer/
│
├── notebooks/
│   └── Phase1_Final_ESA_Code_Walkthrough.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
