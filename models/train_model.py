# """
# train_and_register_models.py
# Train Ridge, Random Forest, Gradient Boosting, XGBoost for AQI and register in Hopsworks Model Registry
# """

# import hopsworks
# from dotenv import load_dotenv
# import os
# import pandas as pd
# import numpy as np
# from joblib import dump
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # -------------------------
# # 1️⃣ Load API key and login
# # -------------------------
# load_dotenv()
# api_key = os.getenv("HOPSWORKS_API_KEY")
# if not api_key:
#     raise ValueError("❌ Missing HOPSWORKS_API_KEY in .env file")

# project = hopsworks.login(project="aqi_prediction2", api_key_value=api_key)
# mr = project.get_model_registry()
# print("✅ Connected to Hopsworks Model Registry")

# # -------------------------
# # 2️⃣ Load features from Feature Store
# # -------------------------
# fs = project.get_feature_store()
# fg = fs.get_feature_group("aqi_features", version=2)
# df = fg.read()
# print(f"✅ Fetched {len(df)} rows from Feature Store")

# # -------------------------
# # 3️⃣ Prepare dataset
# # -------------------------
# if "datetime_str" in df.columns:
#     df["datetime"] = pd.to_datetime(df["datetime_str"])
#     df.drop(columns=["datetime_str"], inplace=True)

# df = df.sort_values("datetime").reset_index(drop=True)

# # Drop leakage features
# leakage_features = [c for c in df.columns if "rolling" in c or "lag" in c]
# for c in leakage_features:
#     if c in df.columns:
#         df.drop(columns=[c], inplace=True)

# X = df.drop(columns=["aqi", "datetime"])
# y = df["aqi"]

# # -------------------------
# # 4️⃣ Train-test split (80%/20%)
# # -------------------------
# split_index = int(len(df) * 0.8)
# X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
# y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# # -------------------------
# # 5️⃣ Preprocessing for Ridge Regression
# # -------------------------
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # -------------------------
# # 6️⃣ Define models
# # -------------------------
# models = {
#     "Ridge_Regression": Ridge(alpha=1.0),
#     "Random_Forest": RandomForestRegressor(n_estimators=200, random_state=42),
#     "Gradient_Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42),
#     "XGBoost": XGBRegressor(
#         n_estimators=200,
#         learning_rate=0.1,
#         max_depth=6,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42,
#         tree_method="hist"
#     )
# }

# results = {}
# model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
# os.makedirs(model_dir, exist_ok=True)

# # -------------------------
# # 7️⃣ Train, evaluate & register models
# # -------------------------
# for name, model in models.items():
#     print(f"\n🚀 Training {name}...")

#     # Scale only Ridge
#     if name == "Ridge_Regression":
#         model.fit(X_train_scaled, y_train)
#         preds = model.predict(X_test_scaled)
#     else:
#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)

#     # Evaluate
#     rmse = np.sqrt(mean_squared_error(y_test, preds))
#     mae = mean_absolute_error(y_test, preds)
#     r2 = r2_score(y_test, preds)
#     results[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}

#     print(f"✅ {name} → RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

#     # Save locally
#     model_path = os.path.join(model_dir, f"{name.lower()}.pkl")
#     dump(model, model_path)
#     print(f"💾 Model saved locally at {model_path}")

#     # -------------------------
#     # Register in Hopsworks Model Registry
#     # -------------------------
#     try:
#         model_entry = mr.get_model(name)
#         print(f"ℹ️ Model '{name}' already exists in registry")
#     except hopsworks.client.exceptions.RestAPIError:
#         model_entry = mr.create_model(name, f"{name} for AQI prediction")
#         print(f"✅ Created model '{name}' in registry")

#     version = model_entry.add_version(
#         model_path=model_path,
#         metrics={"RMSE": rmse, "MAE": mae, "R2": r2},
#         description="Initial version trained on historical AQI data"
#     )
#     print(f"✅ Registered '{name}' with version {version.version}")

# # -------------------------
# # 8️⃣ Summary
# # -------------------------
# print("\n📊 Model performance summary:")
# results_df = pd.DataFrame(results).T.sort_values(by="RMSE")
# print(results_df)



"""
train_and_register_models.py
Train Ridge, Random Forest, Gradient Boosting, XGBoost for AQI and register in Hopsworks Model Registry
"""

import hopsworks
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import traceback

# -------------------------
# 1️⃣ Load API key and login
# -------------------------
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")
if not api_key:
    raise ValueError("❌ Missing HOPSWORKS_API_KEY in .env file")

project = hopsworks.login(project="aqi_prediction2", api_key_value=api_key)
mr = project.get_model_registry()
print("✅ Connected to Hopsworks Model Registry")

# -------------------------
# 2️⃣ Load features from Feature Store
# -------------------------
fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_features", version=2)
df = fg.read()
print(f"✅ Fetched {len(df)} rows from Feature Store")

# -------------------------
# 3️⃣ Prepare dataset
# -------------------------
if "datetime_str" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime_str"])
    df.drop(columns=["datetime_str"], inplace=True)

df = df.sort_values("datetime").reset_index(drop=True)

# Drop leakage features
leakage_features = [c for c in df.columns if "rolling" in c or "lag" in c]
for c in leakage_features:
    if c in df.columns:
        df.drop(columns=[c], inplace=True)

X = df.drop(columns=["aqi", "datetime"])
y = df["aqi"]

# -------------------------
# 4️⃣ Train-test split (80%/20%)
# -------------------------
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Save feature names for later use
feature_names = X.columns.tolist()

# -------------------------
# 5️⃣ Preprocessing for Ridge Regression
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
scaler_path = os.path.join(os.path.dirname(__file__), "../models", "scaler.pkl")
dump(scaler, scaler_path)

# -------------------------
# 6️⃣ Define models
# -------------------------
models = {
    "Ridge_Regression": Ridge(alpha=1.0),
    "Random_Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient_Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist"
    )
}

results = {}
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
os.makedirs(model_dir, exist_ok=True)

# -------------------------
# 7️⃣ Train, evaluate & register models
# -------------------------
for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"🚀 Training {name}...")
    print('='*60)

    # Scale only Ridge
    if name == "Ridge_Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        # Save feature names for Ridge
        model.feature_names = feature_names
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}

    print(f"✅ {name} → RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

    # Save locally
    model_path = os.path.join(model_dir, f"{name.lower()}.pkl")
    dump(model, model_path)
    print(f"💾 Model saved locally at {model_path}")

    # -------------------------
    # Register in Hopsworks Model Registry - UPDATED & ROBUST
    # -------------------------
    print(f"📤 Registering {name} in Model Registry...")
    
    # Prepare metrics
    metrics = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2)
    }
    
    # Prepare input example for the model
    if name == "Ridge_Regression":
        input_example = X_train_scaled[:1].tolist()
    else:
        input_example = X_train.iloc[:1].to_dict(orient='records')[0]
    
    # Create a description
    description = f"""{name} for AQI prediction
    - Trained on {len(X_train)} samples
    - Test RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}
    - Features: {len(feature_names)} variables
    - Created: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    # METHOD 1: Try using the recommended approach
    try:
        print(f"  Trying method 1: mr.python.create_model()...")
        
        # Create the model entry
        model_entry = mr.python.create_model(
            name=name,
            metrics=metrics,
            description=description,
            input_example=input_example
        )
        
        # Save the model
        model_entry.save(model_path)
        
        print(f"  ✅ Successfully registered '{name}' using create_model()")
        
    except Exception as e:
        print(f"  ⚠️ Method 1 failed: {str(e)[:100]}...")
        
        # METHOD 2: Try the get_or_create approach
        try:
            print(f"  Trying method 2: Check if model exists then create...")
            
            # First, try to get existing models
            existing_models = mr.get_models()
            model_exists = False
            model_entry = None
            
            for m in existing_models:
                if m.name == name:
                    model_exists = True
                    model_entry = m
                    break
            
            if model_exists:
                print(f"  ℹ️ Model '{name}' already exists, adding new version...")
                # Add new version
                version = model_entry.add_version(
                    model_path=model_path,
                    metrics=metrics,
                    description=description
                )
                print(f"  ✅ Added version {version.version} to existing model '{name}'")
            else:
                print(f"  ℹ️ Model '{name}' doesn't exist, creating new...")
                # Create new model
                model_entry = mr.python.create_model(
                    name=name,
                    metrics=metrics,
                    description=description
                )
                version = model_entry.add_version(
                    model_path=model_path,
                    metrics=metrics,
                    description=description
                )
                print(f"  ✅ Created new model '{name}' with version {version.version}")
                
        except Exception as e2:
            print(f"  ⚠️ Method 2 failed: {str(e2)[:100]}...")
            
            # METHOD 3: Direct upload approach
            try:
                print(f"  Trying method 3: Direct upload...")
                
                # Create a simple model metadata
                from hsml.model_schema import ModelSchema
                from hsml.schema import Schema
                
                # Create input schema
                input_schema = Schema(X_train)
                
                # Create model schema
                model_schema = ModelSchema(
                    input_schema=input_schema,
                    model_name=name
                )
                
                # Upload directly
                mr.upload(
                    model_path,
                    name=name,
                    metrics=metrics,
                    description=description,
                    input_example=input_example,
                    model_schema=model_schema
                )
                
                print(f"  ✅ Successfully uploaded '{name}' using direct upload")
                
            except Exception as e3:
                print(f"  ⚠️ Method 3 failed: {str(e3)[:100]}...")
                
                # METHOD 4: Manual file upload (last resort)
                try:
                    print(f"  Trying method 4: Manual file operations...")
                    
                    # Create model directory in Hopsworks
                    model_dir_name = f"models/{name}"
                    
                    # Get the dataset API
                    dataset_api = project.get_dataset_api()
                    
                    # Upload model file
                    uploaded_path = dataset_api.upload(
                        model_path,
                        model_dir_name,
                        overwrite=True
                    )
                    
                    print(f"  ✅ Uploaded model file to: {uploaded_path}")
                    print(f"  ℹ️ Note: Manual registration needed in Hopsworks UI for '{name}'")
                    
                except Exception as e4:
                    print(f"  ❌ All registration methods failed for '{name}'")
                    print(f"  Last error: {str(e4)[:200]}")
                    print(f"  Please register model manually in Hopsworks UI")

# -------------------------
# 8️⃣ Save performance summary
# -------------------------
print(f"\n{'='*60}")
print("📊 MODEL TRAINING SUMMARY")
print('='*60)

results_df = pd.DataFrame(results).T.sort_values(by="RMSE")
print("\nPerformance Summary (sorted by RMSE):")
print(results_df.to_string())

# Save summary to file
summary_path = os.path.join(model_dir, "model_performance_summary.csv")
results_df.to_csv(summary_path)
print(f"\n💾 Performance summary saved to: {summary_path}")

# -------------------------
# 9️⃣ Save metadata
# -------------------------
metadata = {
    "training_date": pd.Timestamp.now().isoformat(),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "total_samples": len(df),
    "features_used": feature_names,
    "feature_count": len(feature_names),
    "best_model": results_df.index[0],
    "best_rmse": float(results_df.iloc[0]["RMSE"])
}

metadata_path = os.path.join(model_dir, "training_metadata.json")
import json
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"💾 Training metadata saved to: {metadata_path}")

print(f"\n{'='*60}")
print("✅ MODEL TRAINING COMPLETED!")
print('='*60)
print(f"📋 Total models trained: {len(models)}")
print(f"🏆 Best model: {results_df.index[0]} (RMSE: {results_df.iloc[0]['RMSE']:.3f})")
print(f"📁 Models saved in: {model_dir}")
print('='*60)