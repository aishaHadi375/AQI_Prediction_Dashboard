"""
Purpose:
- Load latest AQI data from Hopsworks Feature Store
- Load BEST model from Hopsworks Model Registry
- Evaluate model using time-based split
- Generate AQI predictions for next 3 days (72 hours)

UPDATED: Fetches model from Hopsworks Model Registry (not local files)
"""

import os
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import hopsworks
import dotenv

# =====================================================
# 1. Environment Setup
# =====================================================
dotenv.load_dotenv()
API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not API_KEY:
    raise ValueError("❌ HOPSWORKS_API_KEY not found in .env file")

# Create output directory
OUTPUT_DIR = os.path.join("data", "predictions")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "next_3_days_predictions.csv")

# =====================================================
# 2. Connect to Hopsworks
# =====================================================
print("="*60)
print("🔗 CONNECTING TO HOPSWORKS")
print("="*60)

project = hopsworks.login(api_key_value=API_KEY)
print(f"✅ Connected to project: {project.name}")

# =====================================================
# 3. Load Data from Feature Store
# =====================================================
print("\n" + "="*60)
print("📊 LOADING DATA FROM FEATURE STORE")
print("="*60)

fs = project.get_feature_store()
print(f"✅ Connected to Feature Store: {fs.name}")

# Get feature group (try version 2 first, then version 1)
fg = None
for version in [2, 1]:
    try:
        fg = fs.get_feature_group("aqi_features", version=version)
        print(f"✅ Found Feature Group 'aqi_features' version {version}")
        break
    except Exception as e:
        print(f"⚠️ Version {version} not found, trying next...")

if fg is None:
    raise ValueError("❌ Could not find 'aqi_features' feature group")

# Read data
df = fg.read()
print(f"✅ Data fetched successfully!")
print(f"   Shape: {df.shape}")
print(f"   Columns: {len(df.columns)}")

# =====================================================
# 4. Process Datetime Column
# =====================================================
print("\n" + "="*60)
print("🕐 PROCESSING DATETIME")
print("="*60)

# Check and convert datetime
if "datetime_str" in df.columns:
    print("Found 'datetime_str' column")
    df["datetime"] = pd.to_datetime(df["datetime_str"])
    df.drop(columns=["datetime_str"], inplace=True)
    print("✅ Converted datetime_str → datetime")
elif "datetime" in df.columns:
    print("Found 'datetime' column")
    df["datetime"] = pd.to_datetime(df["datetime"])
    print("✅ Converted existing datetime")
else:
    # Look for any time-related column
    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    if time_cols:
        print(f"⚠️ Using fallback column: {time_cols[0]}")
        df["datetime"] = pd.to_datetime(df[time_cols[0]])
    else:
        raise ValueError("❌ No datetime column found!")

# Sort by datetime
df = df.sort_values("datetime").reset_index(drop=True)
print(f"✅ Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"   Total records: {len(df)}")

# =====================================================
# 5. Remove Leakage Features
# =====================================================
print("\n" + "="*60)
print("🧹 REMOVING LEAKAGE FEATURES")
print("="*60)

leakage_features = [
    "aqi_rolling_24h",
    "aqi_lag_1h", 
    "high_pollution_flag"
]

dropped = []
for col in leakage_features:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        dropped.append(col)
        print(f"   ✓ Dropped: {col}")

if dropped:
    print(f"✅ Removed {len(dropped)} leakage feature(s)")
else:
    print("ℹ️ No leakage features found")

# =====================================================
# 6. Prepare Features and Target
# =====================================================
print("\n" + "="*60)
print("📋 PREPARING FEATURES & TARGET")
print("="*60)

if 'aqi' not in df.columns:
    raise ValueError("❌ Target column 'aqi' not found!")

X = df.drop(columns=["aqi", "datetime"], errors="ignore")
y = df["aqi"]

print(f"✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")
print(f"\n   Feature columns ({len(X.columns)}):")
for i, col in enumerate(X.columns, 1):
    if i <= 10:  # Show first 10
        print(f"   {i:2d}. {col}")
if len(X.columns) > 10:
    print(f"   ... and {len(X.columns) - 10} more")

# =====================================================
# 7. Load BEST Model from Model Registry
# =====================================================
print("\n" + "="*60)
print("🤖 LOADING MODEL FROM MODEL REGISTRY")
print("="*60)

mr = project.get_model_registry()
print("✅ Connected to Model Registry")

# Define model names to try (in priority order)
model_names_to_try = [
    "Random_Forest",
    "random_forest", 
    "XGBoost",
    "xgboost",
    "Gradient_Boosting",
    "gradient_boosting"
]

model = None
model_name = None
model_version = None

# Try to load best performing model
print("\n🔍 Searching for best model...")

for name in model_names_to_try:
    try:
        print(f"\n   Trying '{name}'...")
        model_meta = mr.get_model(name, version=None)  # Get latest version
        
        # Download model
        model_dir = model_meta.download()
        print(f"   ✓ Downloaded to: {model_dir}")
        
        # Find the .pkl file
        import glob
        from joblib import load
        
        pkl_files = glob.glob(os.path.join(model_dir, "*.pkl"))
        if pkl_files:
            model_path = pkl_files[0]
            model = load(model_path)
            model_name = name
            model_version = model_meta.version
            
            print(f"   ✅ Loaded: {name} (version {model_version})")
            
            # Get model metrics if available
            if hasattr(model_meta, 'training_metrics'):
                metrics = model_meta.training_metrics
                print(f"   📊 Training Metrics:")
                if 'RMSE' in metrics or 'rmse' in metrics:
                    rmse = metrics.get('RMSE', metrics.get('rmse', 'N/A'))
                    print(f"      RMSE: {rmse}")
                if 'MAE' in metrics or 'mae' in metrics:
                    mae = metrics.get('MAE', metrics.get('mae', 'N/A'))
                    print(f"      MAE: {mae}")
                if 'R2' in metrics or 'r2' in metrics:
                    r2 = metrics.get('R2', metrics.get('r2', 'N/A'))
                    print(f"      R²: {r2}")
            
            break  # Found a model, stop searching
            
    except Exception as e:
        print(f"   ✗ Not found: {str(e)[:50]}...")
        continue

if model is None:
    print("\n❌ No models found in registry!")
    print("\n💡 Available options:")
    print("   1. Train models first: python src/train_model.py")
    print("   2. Or load from local file if you have one")
    raise ValueError("No model available for predictions")

print(f"\n✅ Using model: {model_name} (v{model_version})")
print(f"   Model type: {type(model).__name__}")

# =====================================================
# 8. Align Features with Model
# =====================================================
print("\n" + "="*60)
print("🔧 ALIGNING FEATURES WITH MODEL")
print("="*60)

if hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)
    print(f"Model expects {len(expected_features)} features")
    
    # Check for mismatches
    current_features = set(X.columns)
    expected_features_set = set(expected_features)
    
    missing = expected_features_set - current_features
    extra = current_features - expected_features_set
    
    if missing:
        print(f"\n⚠️ Missing features ({len(missing)}):")
        for feat in list(missing)[:5]:
            print(f"   • {feat}")
        if len(missing) > 5:
            print(f"   ... and {len(missing)-5} more")
        
        # Add missing features with defaults
        for feat in missing:
            if 'sin' in feat or 'cos' in feat:
                X[feat] = 0.0
            elif 'ratio' in feat:
                X[feat] = 0.5
            elif feat in ['hour', 'day', 'month', 'weekday']:
                X[feat] = 1
            else:
                X[feat] = 0.0
        print("   ✓ Added missing features with default values")
    
    if extra:
        print(f"\nℹ️ Extra features ({len(extra)}) will be ignored")
    
    # Reorder to match model
    X = X[expected_features]
    print(f"✅ Features aligned: {X.shape}")
else:
    print("⚠️ Model doesn't have feature_names_in_, using as-is")

# =====================================================
# 9. Time-Based Evaluation
# =====================================================
print("\n" + "="*60)
print("📈 MODEL EVALUATION (Time-Based Split)")
print("="*60)

# Split at 80% mark chronologically
split_time = df["datetime"].quantile(0.8)
print(f"Split point: {split_time}")

train_mask = df["datetime"] <= split_time
test_mask = df["datetime"] > split_time

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Evaluate on test set
print("\n🔮 Generating predictions on test set...")
test_preds = model.predict(X_test)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, test_preds))
mae = mean_absolute_error(y_test, test_preds)
r2 = r2_score(y_test, test_preds)

print("\n" + "="*60)
print("📊 EVALUATION RESULTS")
print("="*60)
print(f"RMSE: {rmse:8.3f}")
print(f"MAE:  {mae:8.3f}")
print(f"R²:   {r2:8.3f}")
print("="*60)

# =====================================================
# 10. Generate Future Predictions
# =====================================================
print("\n" + "="*60)
print("🔮 GENERATING 3-DAY FORECAST (72 hours)")
print("="*60)

last_datetime = df["datetime"].max()
print(f"Last known data: {last_datetime}")
print(f"Forecasting: {last_datetime + timedelta(hours=1)} to {last_datetime + timedelta(hours=72)}")

# Create future timestamps
future_datetimes = [last_datetime + timedelta(hours=i) for i in range(1, 73)]

# Use last known features as base
base_features = X.iloc[-1].copy()

# Create 72 copies
future_data = pd.DataFrame([base_features] * 72)
future_data.index = range(72)

# Add realistic variations
print("\n⚙️ Adding realistic variations...")
vary_cols = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "ozone", "sulphur_dioxide", "temperature_2m", 
    "relative_humidity_2m", "wind_speed_10m"
]

vary_cols_present = [col for col in vary_cols if col in future_data.columns]
print(f"   Varying {len(vary_cols_present)} environmental feature(s)")

for col in vary_cols_present:
    # Add ±3% random variation
    noise = np.random.normal(0, 0.03, size=72)
    future_data[col] = future_data[col] * (1 + noise)

# Update time-based features
print("   Updating time-based features...")
for i, dt in enumerate(future_datetimes):
    if 'hour' in future_data.columns:
        future_data.loc[i, 'hour'] = dt.hour
    if 'day' in future_data.columns:
        future_data.loc[i, 'day'] = dt.day
    if 'month' in future_data.columns:
        future_data.loc[i, 'month'] = dt.month
    if 'weekday' in future_data.columns:
        future_data.loc[i, 'weekday'] = dt.weekday()
    if 'hour_sin' in future_data.columns:
        future_data.loc[i, 'hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
    if 'hour_cos' in future_data.columns:
        future_data.loc[i, 'hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)

# Generate predictions
print("\n🔮 Making predictions...")
future_preds = model.predict(future_data)

# Clip to valid AQI range
future_preds = np.clip(future_preds, 0, 500)

# Create results dataframe
future_results = pd.DataFrame({
    "datetime": future_datetimes,
    "predicted_aqi": future_preds
})

print(f"✅ Generated {len(future_results)} hourly predictions")

# =====================================================
# 11. Save Predictions
# =====================================================
future_results.to_csv(OUTPUT_PATH, index=False)
print(f"\n💾 Predictions saved to: {OUTPUT_PATH}")

# =====================================================
# 12. Daily Averages
# =====================================================
print("\n" + "="*60)
print("📅 DAILY AVERAGE PREDICTIONS")
print("="*60)

future_results["date"] = future_results["datetime"].dt.date
daily_avg = future_results.groupby("date")["predicted_aqi"].agg(['mean', 'min', 'max']).reset_index()

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good 🟢"
    elif aqi <= 100:
        return "Moderate 🟡"
    elif aqi <= 150:
        return "Unhealthy (SG) 🟠"
    elif aqi <= 200:
        return "Unhealthy 🔴"
    elif aqi <= 300:
        return "Very Unhealthy 🟣"
    else:
        return "Hazardous ⚫"

for _, row in daily_avg.iterrows():
    category = get_aqi_category(row['mean'])
    print(f"{row['date']} → Avg: {row['mean']:6.1f} (Min: {row['min']:5.1f}, Max: {row['max']:5.1f}) {category}")

# =====================================================
# 13. Trend Analysis
# =====================================================
print("\n" + "="*60)
print("📊 TREND ANALYSIS (24-Hour Comparison)")
print("="*60)

recent_24h = df.sort_values("datetime").tail(24)["aqi"].mean()
forecast_24h = future_results.head(24)["predicted_aqi"].mean()

print(f"Actual AQI (last 24h):     {recent_24h:6.2f}")
print(f"Predicted AQI (next 24h):  {forecast_24h:6.2f}")

delta = forecast_24h - recent_24h
print(f"Expected change:           {delta:+6.2f}")

if delta > 5:
    print("\n🚨 TREND: Air quality likely to WORSEN")
    print("   ⚠️  Recommendation: Limit outdoor activities")
elif delta < -5:
    print("\n🌱 TREND: Air quality likely to IMPROVE")
    print("   ✅ Recommendation: Conditions improving")
else:
    print("\n➖ TREND: Air quality likely to remain STABLE")
    print("   ℹ️  Recommendation: Monitor conditions")

# =====================================================
# 14. Alert Check
# =====================================================
print("\n" + "="*60)
print("⚠️  HAZARD CHECK")
print("="*60)

hazardous = future_results[future_results["predicted_aqi"] > 200]

if len(hazardous) > 0:
    print(f"🚨 WARNING: {len(hazardous)} hour(s) with unhealthy/hazardous AQI!")
    print(f"\nFirst occurrence:")
    first = hazardous.iloc[0]
    print(f"   {first['datetime']} → AQI {first['predicted_aqi']:.1f}")
    
    if len(hazardous) > 1:
        print(f"\nLast occurrence:")
        last = hazardous.iloc[-1]
        print(f"   {last['datetime']} → AQI {last['predicted_aqi']:.1f}")
else:
    print("✅ No hazardous AQI levels predicted")

# =====================================================
# 15. Summary
# =====================================================
print("\n" + "="*60)
print("📋 SUMMARY")
print("="*60)
print(f"Model used:               {model_name} v{model_version}")
print(f"Data source:              Hopsworks Feature Store")
print(f"Training samples:         {len(X_train)}")
print(f"Test samples:             {len(X_test)}")
print(f"Test RMSE:                {rmse:.3f}")
print(f"Test R²:                  {r2:.3f}")
print(f"Predictions generated:    {len(future_results)}")
print(f"Prediction period:        {future_results['datetime'].min()} to {future_results['datetime'].max()}")
print(f"Output file:              {OUTPUT_PATH}")
print("="*60)

print("\n✅ PREDICTION PIPELINE COMPLETE!")
print("\n💡 Next steps:")
print("   1. View predictions: cat", OUTPUT_PATH)
print("   2. Launch dashboard: streamlit run streamlit_app.py")
print("   3. Set up alerts: python alert_system.py")