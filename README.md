# 🌍 Karachi AQI Predictor

An end-to-end MLOps system for predicting Air Quality Index (AQI) in Karachi, Pakistan. It fetches real-time air quality and weather data, engineers features, trains ML models, and serves forecasts through a Streamlit dashboard  fully automated via GitHub Actions CI/CD.

---

## 🗂️ Project Structure

```
karachi-aqi-predictor/
│
├── ci_cd_pipeline/            # GitHub Actions workflow YAMLs
├── data/                      # Raw, processed, historical & final CSVs
├── models/                    # Saved .pkl model files
    ├── train_model.py         # Train & register ML models in Hopsworks
│   └── predict_evaluate.py    # Evaluate models + generate 3-day forecast
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
│   
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

## 🧪 Feature Engineering
Features are engineered in two phases inside process_features.py:

**Phase 1 — Creation:**

EPA AQI sub-indices for all 6 pollutants (PM2.5, PM10, CO, NO2, O3, SO2)
Time features: hour, day, month, weekday
Cyclic encoding: hour_sin, hour_cos (so model knows hour 23 ≈ hour 0)
Rolling averages: 3h, 6h, 24h AQI means
Lag features: AQI 1h, 3h, 6h ago
Derived: pm_ratio, temp_humidity_ratio, wind_effect
Binary flag: high_pollution_flag (1 if AQI > 150)

**Phase 2 — Refinement (from EDA):**
Drops redundant sub-indices, low-variance and weakly correlated features identified during exploratory analysis.

## 🔮 3-Day AQI Forecast
The forecast is generated in predict_evaluate.py using an iterative prediction approach:

The trained XGBoost model takes the last known feature row as input
It predicts the next hour's AQI
That prediction is fed back as a lag feature (aqi_lag_1h) for the next step
This repeats for 72 hours (3 days) to produce a full hourly forecast

The forecast is saved to data/predictions/next_3_days_predictions.csv and visualized as an interactive Plotly chart in the Streamlit dashboard. This approach captures momentum in pollution patterns — if AQI is rising, the model tends to predict continued rise.

## 📈 SHAP Explainability
The project uses SHAP (SHapley Additive exPlanations) via shap_analysis.py to explain what the XGBoost model has learned:

**Feature Importance Bar Plot** — ranks all features by their importance and shows whether higher or lower feature values increase or decrease AQI.

<img width="285" height="311" alt="image" src="https://github.com/user-attachments/assets/5f7ceba7-6c7f-49c2-80dc-fa6cc2942a7f" />

**Waterfall plot** — breaks down a single prediction to show each feature's exact contribution

<img width="390" height="311" alt="image" src="https://github.com/user-attachments/assets/f8d9a701-30ea-455f-bfec-7282dd9b5fec" />

**Dependence plot** — shows how a specific feature (e.g., pm2_5) relates to predicted AQI

<img width="374" height="313" alt="image" src="https://github.com/user-attachments/assets/f3156c08-4ad3-4407-acf1-c06e7aec18f9" />



## 🌐 FastAPI Endpoint
A lightweight REST API is available in streamlit_app/fast_api.py for programmatic AQI predictions.
Start the API server:

uvicorn streamlit_app.fast_api:app --reload --port 8000

**Sample Request**:

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pm2_5": 45.2,
    "pm10": 78.5,
    "temperature_2m": 32.0,
    "relative_humidity_2m": 65.0,
    "wind_speed_10m": 12.3,
    "aqi_lag_1h": 142.0
  }'

**Sample Response**:

{
  "predicted_aqi": 156,
  "category": "Unhealthy",
  "health_advisory": "Everyone should limit outdoor activities."
}

## 🚨 AQI Alert System
The Discord alert system (alert_system.py) sends notifications to a Discord channel when AQI exceeds safe thresholds.

<img width="380" height="206" alt="image" src="https://github.com/user-attachments/assets/73a19912-c0d4-416e-b9c4-720d27a6d8b9" />

Alerts include a health advisory message, color-coded embed, and emoji severity indicator.

## ⚠️ Important: alert_system.py is a script, not a background service. It runs once and exits — it does not monitor AQI continuously by itself. To make alerts automatic, use one of the methods below.


**Option 1 — Continuous loop (local):** Add a loop at the bottom of alert_system.py to keep it running:
import time
while True:
    monitor_predictions()       # check AQI
    time.sleep(3600)            # wait 1 hour, then check again
    
Run it once and leave the terminal open. It will check every hour automatically.

**Option 2 — From the Streamlit app:** Call the alert function inside app.py so every dashboard refresh also triggers an alert check:

python
from alert_system import DiscordAQIAlertSystem
alerter = DiscordAQIAlertSystem()
alerter.monitor_predictions(predictions_df)

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
| Feature Pipeline | Daily · 30:20 UTC | Fetch → Clean → Engineer → Upload to Hopsworks |
| Training Pipeline | Weekly · Sunday 45:20 UTC | Retrain all models, register best in Hopsworks |

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
