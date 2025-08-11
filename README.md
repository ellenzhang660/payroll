# 🧾 Payroll  
**Deep learning for financial analysis in payroll systems**

This project leverages state-of-the-art time series models to analyze and interpret financial data—particularly for **payroll systems** within companies.

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
🏢 **Institution**: Salesforce AI Research  
📊 **Parameter Count**: ~110M  

A **lightweight**, zero-shot time series forecasting model designed for general-purpose use across datasets.

**Key Features:**
- Zero-shot forecasting  
- Supports arbitrary frequency and prediction lengths  
- Fast and adaptable with minimal tuning  

---

### 🔹 [TinyTimeMixers (TTM1)](https://github.com/glehet/TTM1)  
📄 [Paper](https://arxiv.org/abs/2401.03955)  
🏢 **Institution**: IBM Research  
📊 **Parameter Count**: < 1M  

A compact, pretrained model designed for **multivariate time series forecasting**.

**Key Features:**
- Extremely lightweight (<1M parameters)  
- Designed for deployment in low-resource settings  
- Suited for complex, multivariate temporal patterns  

---

### 🔹 [TimesFM](https://github.com/google-research/timesfm)  
📄 [Paper](https://arxiv.org/abs/2310.10688)  
🏢 **Institution**: Google Research  
📊 **Parameter Count**: ~200M  

An efficient foundation model for **univariate** time series forecasting with long context lengths.

**Key Features:**
- Handles up to 512 context time points  
- Provides point forecasts with optional quantile heads  
- Requires aligned context/horizon frequency  

---

### 🔹 [MOMENT](https://github.com/moment-timeseries-foundation-model/moment)  
📄 [Paper](https://arxiv.org/abs/2402.03885)  
🏢 **Institution**: Salesforce AI Research  
📊 **Parameter Count**: 150M – 1.5B (depending on variant)  

A general-purpose time series foundation model for a wide range of tasks.

**Key Features:**
- Forecasting, classification, anomaly detection, and imputation  
- Works well zero-shot and few-shot  
- Tunable with in-distribution data to boost task-specific performance  

---

## ✅ TODO

- [ ] **Synthetic Data Generation**  
  - Generate large-scale synthetic financial datasets  
  - Evaluate only on real user data for robustness

- [ ] **Downstream Tasks**  
  - Forecasting  
  - Fraud detection  
  - Anomaly detection  
  - ⚠️ *Currently, only LagLlama is supported*

---

## 📌 License & Contributions

This project is open for collaboration. If you'd like to contribute or report issues, feel free to open a pull request or create an issue.

---
