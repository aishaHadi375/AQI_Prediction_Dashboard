import hopsworks
import pandas as pd
from dotenv import load_dotenv
import os

# 1️⃣ Load API key from .env
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")

if not api_key:
    raise ValueError("❌ Missing HOPSWORKS_API_KEY in .env")

# 2️⃣ Login to your project
project = hopsworks.login(
    project="aqi_prediction2",  # your project name
    api_key_value=api_key
)

fs = project.get_feature_store()

# 3️⃣ Get your Feature Group (version 2)
fg = fs.get_feature_group("aqi_features", version=2)

# 4️⃣ Fetch offline data (all stored rows)
df_offline = fg.read()

# 5️⃣ Optional: Fetch online data (by primary key, if needed)
# df_online = fg.read_online({"datetime_str": "2026-01-24 00:00:00"})

# 6️⃣ Inspect fetched data
print("✅ Successfully fetched data from Feature Store")
print("\n📊 DataFrame shape:", df_offline.shape)
print("\n🧩 Sample rows:")
print(df_offline.head(5))

print("\n📌 Datetime range in Feature Store:")
print("Start:", df_offline["datetime_str"].min())
print("End  :", df_offline["datetime_str"].max())
