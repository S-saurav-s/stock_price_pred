# 📈 RiskPulse — Market Volatility & Risk Intelligence Engine

<p align="center">
  <b>Multi-Horizon • Interpretable • End-to-End Deployed</b>
</p>

<p align="center">
  🟢 LOW 🟡 NORMAL 🟠 ELEVATED 🔴 HIGH
</p>

---

## 🚀 Overview

**RiskPulse** is a production-style **market risk intelligence system** that forecasts  
**short-term and medium-term volatility** and converts it into a clear, interpretable **risk regime**.

> ⚠️ This is **not** a price prediction system  
> ✅ It is a **risk-awareness and monitoring engine**, aligned with real financial practice

The goal is simple: **identify rising market risk before it impacts decisions.**

---

## ✨ Core Capabilities

- 📊 **5-Day Volatility Forecast** (short-term sensitivity)
- 📉 **10-Day Volatility Forecast** (medium-term regime detection)
- 🧠 **Composite Risk Score**
- 🌍 **Live market data ingestion**
- ⚡ **FastAPI-based inference backend**
- 🖥️ **Streamlit visualization dashboard**
- 🧩 **Fully interpretable logic (no black-box LLM output)**

---

## 🌍 Supported Markets

| Market | Ticker |
|------|------|
| 🇮🇳 NIFTY 50 | `^NSEI` |
| 🇺🇸 S&P 500 | `^GSPC` |
| ₿ Bitcoin | `BTC-USD` |

---

## 🧠 Why Volatility Instead of Price?

<details>
<summary><b>Click to expand</b></summary>

- 📉 Price prediction is highly noisy and unstable
- 📊 Volatility reflects uncertainty and stress
- 📐 Volatility is more stationary than raw prices
- 🏦 Volatility is widely used by risk desks and portfolio managers

**RiskPulse focuses on knowing when risk is rising, not guessing direction.**

</details>

---

## ⚙️ System Design

### 🔹 Multi-Horizon Forecasting

- **5-Day Model**
  - Captures short-term shocks
  - Sensitive to sudden market stress

- **10-Day Model**
  - Captures sustained volatility regimes
  - Filters out transient noise

Both models are trained independently for clarity and robustness.

---

### 🔹 Composite Risk Logic

Composite Risk Score =
0.4 × Z(5-Day Volatility) +
0.6 × Z(10-Day Volatility)


- Z-score normalization ensures asset-agnostic behavior
- Higher weight on 10-day volatility stabilizes regime detection

---

### 🔹 Risk Regimes

| Score Range | Risk Level |
|-----------|-----------|
| < -0.5 | 🟢 LOW |
| -0.5 → 0.5 | 🟡 NORMAL |
| 0.5 → 1.2 | 🟠 ELEVATED |
| > 1.2 | 🔴 HIGH |

✔ Normalized  
✔ Interpretable  
✔ Comparable across assets  

---

## 🏗️ Architecture

📡 Live Market Data
↓
🧪 Feature Engineering
↓
🌲 XGBoost Models
(5-Day & 10-Day)
↓
⚡ FastAPI Backend
↓
🧠 Composite Risk Engine
↓
🖥️ Streamlit Dashboard


---

## 🧩 Tech Stack

| Layer | Technology |
|-----|-----------|
| Machine Learning | XGBoost |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Market Data | Yahoo Finance |
| Deployment | Render + Streamlit Cloud |
| Language | Python 🐍 |

---

## ▶️ Run Locally

### Backend (FastAPI)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open API docs at:
👉 http://127.0.0.1:8000/docs

streamlit run app.py

🟢 Live Demo

🌐 Frontend: Streamlit Cloud

⚙️ Backend API: Render

⏳ Initial load may take a few seconds due to live data fetching and dynamic computation.

🧪 What Makes This Project Different?

✔ Not a notebook-only ML project

✔ No black-box explanations

✔ Real deployment challenges handled

✔ Designed as a system, not a script

✔ Financially meaningful objective (risk, not price)

🛣️ Roadmap

📊 Historical risk regime visualization

⚡ Batch prediction endpoint

🧠 Optional GenAI explanation layer

🚀 Performance optimization & caching<br>

⚠️ Disclaimer
This project is for educational and analytical purposes only.<br>
It does not constitute financial or investment advice.
<br><br>
👤 Author
Om Timbadiya <br>
🎓 3rd Year Engineering Student<br>
🤖 Machine Learning • Forecasting • Applied AI
