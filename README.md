# 🧾 Payroll  
**Deep learning for financial analysis in payroll systems**

This project leverages state-of-the-art time series models to help analyze and interpret financial data, specifically tailored for **payroll systems** within companies.

## 🔍 Overview

**Focus Areas:**
- Budget tracking and forecasting  
- Forensic and cost accounting  
- International and multilingual accounting standards  
- Adaptation to diverse financial terminologies  

---

## 🧠 Models Used

### 🔹 [LagLlama](https://github.com/time-series-foundation-models/lag-llama)  
📄 [Paper](https://arxiv.org/abs/2310.08278)  
A **lightweight**, zero-shot time series forecasting model designed for general-purpose use across datasets.

**Key Features:**
- Zero-shot forecasting  
- Works with any frequency and prediction length  
- Fast deployment with minimal tuning  

---

### 🔹 [TinyTimeMixers (TTM1)](https://github.com/glehet/TTM1)  
📄 [Paper](https://arxiv.org/abs/2401.03955)  
Compact pretrained models from IBM Research for **multivariate time series forecasting**.

**Key Features:**
- Fewer than 1 million parameters  
- Designed for complex multivariate trends  
- Efficient for environments with limited compute  

---

### 🔹 [TimesFM](https://github.com/google-research/timesfm)  
📄 [Paper](https://arxiv.org/abs/2310.10688)  
Optimized for **univariate** forecasting with extended context windows.

**Key Features:**
- Supports up to 512 time points of context  
- Provides point forecasts, with optional quantile heads  
- Requires matching frequency and contiguous context for input/output  

---

### 🔹 [MOMENT](https://github.com/moment-timeseries-foundation-model/moment)  
📄 [Paper](https://arxiv.org/abs/2402.03885)  
A general-purpose time series foundation model supporting multiple tasks.

**Key Features:**
- Forecasting, classification, anomaly detection, imputation  
- Effective out-of-the-box with minimal task-specific data  
- Zero-shot and few-shot performance  
- Can be tuned for in-distribution and specific downstream tasks  

---

## ✅ TODO

- [ ] **Synthetic Data Generation**  
  - Generate large volumes of synthetic financial data  
  - Evaluation will be done only on real (human) financial datasets

- [ ] **Downstream Tasks**  
  - Forecasting  
  - Fraud detection  
  - Anomaly detection  
  - ⚠️ *Currently, only LagLlama is supported*

---

## 📌 License & Contributions

This project is open for collaboration. If you'd like to contribute or report issues, feel free to open a pull request or create an issue.

---

