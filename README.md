# Hotel-time-series-
A multivariate deep learning framework for Hotel Demand Forecasting. Features include Survival Analysis for risk modeling, K-Means for segmentation, and a comparative study against statistical baselines (SARIMA) to optimize revenue and staffing.

## 🛠️ Methodology & Technical Strategy

### 1. Data Strategy (The "Distinction" Move)
We utilized the [Hotel Booking Demand Dataset](https://www.sciencedirect.com/science/article/pii/S2352340918315191) (119k records). Instead of standard cleaning, we applied a **Strategic Data Pipeline**:
* **Weekly Aggregation:** Aggregated 2 years of daily data into 114 weeks to balance signal-to-noise ratio for Deep Learning.
* **Segmentation Split:** Modeled **City Hotels** (High Volatility) separately from **Resort Hotels** (High Seasonality) to prevent signal cancellation.
* **Operational Outliers:** Purposefully retained 0-ADR (Complimentary) bookings to ensure operational staffing models reflected true workload, while filtering errors (> €5000 ADR) to protect financial models.

### 2. Modeling: Champion vs. Challenger
We benchmarked three levels of complexity to isolate the source of predictive power:

* **Level 1 (Baseline): SARIMA.** (Univariate). Captures Seasonality ($s=52$) and Trend.
* **Level 2 (Test): SARIMAX.** (Linear Multivariate). Failed due to confounding price/demand correlation.
* **Level 3 (Advanced): Temporal Fusion Transformer.** (Non-Linear Multivariate). Uses Attention mechanisms to weigh the importance of **Lead Time** and **Price**.

* ### Prerequisites
* Python 3.10+
* PyTorch Lightning (v2.0+)
* PyTorch Forecasting (v1.0+)
