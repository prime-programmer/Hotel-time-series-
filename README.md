---
title: V.Ger Forecasting Engine
emoji: 🏨
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

## Overview
The V.Ger Hotel forecasting engine is an end-to-end machine learning pipeline and interactive executive dashboard. It is designed to predict 10-week hotel booking trends and simulate the impact of dynamic pricing strategies on hotel occupancy. 

Instead of relying on a single algorithm, this project utilizes a **dual-engine architecture**:
1. **Operational Baseline (SARIMA):** A highly reliable statistical model that predicts future bookings assuming no business variables change.
2. **Strategic Simulation (Temporal Fusion Transformer - TFT):** A deep learning multivariate model built in PyTorch that allows executives to adjust the Simulated Average Daily Rate (ADR) and instantly see how pricing changes ripple through future demand.

##  Live Demo
* **Interactive Dashboard:** https://vger-hotel-forecast.streamlit.app/
* **API Documentation:** https://chibikeobi-vger-forecast.hf.space/docs

##  Architecture & Tech Stack

* **Machine Learning:** PyTorch, PyTorch Forecasting (TFT), Statsmodels (SARIMA), Pandas, Scikit-Learn.
* **Backend & API:** FastAPI (with LRU Caching for latency reduction), Uvicorn.
* **Database:** Supabase (PostgreSQL) processing 100,000+ rows of historical booking data.
* **Frontend UI:** Streamlit, Plotly Express.
* **DevOps & Cloud:** Docker, Hugging Face Spaces (Backend), Streamlit Community Cloud (Frontend), Git LFS (Large File Storage for model weights).

##  Key Features
* **Real-time "What-If" Scenarios:** Adjust pricing levers (ADR) via a slider and watch the deep learning model recalculate the forecast dynamically.
* **Model Comparison:** Visually track the aggressive AI pricing simulations (TFT) against the safe statistical baseline (SARIMA) on a unified interactive graph.
* **Cloud-Optimized Backend:** Uses `lru_cache` to drop database query times from minutes down to milliseconds. Heavy model `.ckpt` files are managed via Git LFS to bypass standard repository memory limits.

## 💻 Local Installation & Setup

If you want to run this project locally, follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/prime-programmer/Hotel-time-series-.git](https://github.com/prime-programmer/Hotel-time-series-.git)
cd Hotel-time-series-
```

**2. Pull the heavy model weights (Requires Git LFS)**
```bash
git lfs pull
```

**3. Create a virtual environment and install dependencies**
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows Git Bash
pip install -r requirements.txt
```

**4. Boot up the Backend API (FastAPI)**
```bash
uvicorn api.main:app --reload
```


**5. Boot up the Frontend Dashboard (Streamlit)**
Open a new terminal window, activate the environment, and run:
```bash
streamlit run app.py
```


##  Repository Structure
```text
Hotel-time-series-/
│
├── api/                  # FastAPI backend code
│   └── main.py           # API endpoints (/forecast/strategic & /forecast/operational)
├── models/               # Saved model weights (Managed by Git LFS)
│   ├── sarima.pkl
│   └── tft.ckpt
├── src/                  # Core ML and Data Engineering scripts
│   ├── data_loader.py    # Supabase connection & data processing
│   ├── train_tft.py      # PyTorch training loop
│   └── evaluate.py       # Metrics calculation
├── app.py                # Streamlit Dashboard UI
├── requirements.txt      # Python dependencies
├── Dockerfile            # Containerization instructions for Hugging Face
└── README.md             # Project documentation
```

##  Author
**Chibike Ugbam** * Machine Learning | Data Science
* https://www.linkedin.com/in/chibike-ugbam-b16b75ab/
```

