# 🌍 Karachi AQI Predictor

An end-to-end MLOps system for predicting Air Quality Index (AQI) in Karachi, Pakistan. It fetches real-time air quality and weather data, engineers features, trains ML models, and serves forecasts through a Streamlit dashboard — fully automated via GitHub Actions CI/CD.

---

## 🗂️ Project Structure

```
karachi-aqi-predictor/
│
├── ci_cd_pipeline/            # GitHub Actions workflow YAMLs
├── data/                      # Raw, processed, historical & final CSVs
├── models/                    # Saved .pkl model files
├── notebooks/                 # EDA notebooks (preprocessing + feature analysis)
├── src/                       # Core pipeline source code
│   ├── config.py              # API URLs, coordinates, paths, env flags
│   ├── fetch_data.py          # Fetch latest AQ + weather from Open-Meteo
│   ├── process_data.py        # Parse raw data into structured DataFrames
│   ├── clean_data.py          # Handle missing values, outliers, datetime
│   ├── process_features.py    # Feature engineering (AQI, lags, rolling, cyclic)
│   ├── merge_features.py      # Merge historical + daily processed data
│   ├── feature_store.py       # Upload/download features via Hopsworks
│   ├── aqi_utils.py           # EPA AQI calculation utilities
│   ├── backfill_data.py       # One-time 1-year historical data fetch
│   ├── run_feature_pipeline.py# Daily orchestrator: fetch → clean → upload
│   ├── train_model.py         # Train & register ML models in Hopsworks
│   └── predict_evaluate.py    # Evaluate models + generate 3-day forecast
│
├── streamlit_app/
│   ├── app.py                 # Streamlit dashboard (UI)
│   ├── alert_system.py        # Discord webhook AQI alerts
│   └── fast_api.py            # FastAPI inference endpoint
│
├── .env                       # API keys (not committed)
└── requirements.txt           # Python dependencies
```

---

## ⚙️ How It Works

```
Open-Meteo APIs (free, no key)
        │
        ▼
  fetch_data.py → process_data.py → clean_data.py
                                          │
                                  process_features.py
                                    (AQI + ML features)
                                          │
                                   feature_store.py
                                   (Hopsworks upload)
                                          │
                                    train_model.py
                                          │
                                  predict_evaluate.py
                                    (3-day forecast)
                                          │
                          ┌───────────────┘
                          ▼               ▼
                       app.py       alert_system.py
                  (Streamlit UI)  (Discord Alerts)
```

---

## 🤖 Model Performance

Four regression models were trained on the engineered AQI feature set, evaluated on a chronological 80/20 train-test split:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **XGBoost** ✅ | 6.70 | 1.28 | **0.9872** |
| Gradient Boosting | 7.91 | 0.58 | 0.9821 |
| Random Forest | 8.17 | 0.34 | 0.9809 |
| Ridge Regression | 34.70 | 16.29 | 0.6554 |

**XGBoost** was selected as the best model with an R² of **0.987**, explaining 98.7% of AQI variance.

---

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone https://github.com/your-username/karachi-aqi-predictor.git
pip install -r requirements.txt

# 2. Set up .env
HOPSWORKS_API_KEY=your_key
DISCORD_WEBHOOK_URL=your_webhook   # optional
SAVE_LOCAL=True

# 3. Backfill historical data (first time only)
python src/backfill_data.py

# 4. Run feature pipeline
python src/run_feature_pipeline.py

# 5. Train models
python src/train_model.py

# 6. Launch dashboard
streamlit run streamlit_app/app.py
```

---

## 🔄 CI/CD Automation

| Workflow | Schedule | What it does |
|---|---|---|
| Feature Pipeline | Daily · 00:00 UTC | Fetch → Clean → Engineer → Upload to Hopsworks |
| Training Pipeline | Weekly · Sunday 02:00 UTC | Retrain all models, register best in Hopsworks |

**Required GitHub Secrets:** `HOPSWORKS_API_KEY`, `DISCORD_WEBHOOK_URL` (optional)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data | Open-Meteo API (free) |
| Feature Store | Hopsworks |
| Models | XGBoost, scikit-learn |
| Dashboard | Streamlit + Plotly |
| API | FastAPI |
| Alerts | Discord Webhooks |
| CI/CD | GitHub Actions |
| Language | Python 3.10+ |

---

## 📊 Data Sources

All free, no authentication required.

- **Air Quality:** `air-quality-api.open-meteo.com` — PM2.5, PM10, CO, NO2, O3, SO2
- **Weather Forecast:** `api.open-meteo.com` — Temperature, Humidity, Wind
- **Historical Archive:** `archive-api.open-meteo.com` — Same variables from Jan 2024

---

## 📄 License

MIT — adapt for any city by changing `LAT` and `LON` in `src/config.py`.
