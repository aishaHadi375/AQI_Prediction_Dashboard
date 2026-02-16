import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)



st.markdown("""
<style>
    /* Global Reset - Remove all padding/margin */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }
    
    /* Remove spacing between elements */
    .element-container {
        margin-bottom: 0.25rem !important;
    }
    
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.25rem !important;
    }
    
    /* Compact headers - WHITE TEXT */
    h1, h2, h3, h4, h5, h6 {
        margin-top: 0.25rem !important;
        margin-bottom: 0.25rem !important;
        padding: 0 !important;
        color: #ffffff !important;
    }
    
    /* Dark purple background */
    .stApp {
        background: linear-gradient(135deg, #1a0b2e 0%, #2d1b4e 100%);
    }
    
    /* Card styling - minimal padding */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        color: #9370db !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        color: #ffffff !important;
    }
    
    /* Compact metrics */
    div[data-testid="metric-container"] {
        background: rgba(147, 112, 219, 0.1);
        border: 1px solid rgba(147, 112, 219, 0.3);
        border-radius: 8px;
        padding: 0.5rem !important;
        margin: 0 !important;
    }
    
    /* Tab styling - WHITE TEXT */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        background-color: rgba(48, 25, 52, 0.5);
        border-radius: 8px;
        padding: 0.25rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 2rem !important;
        padding: 0.25rem 0.75rem !important;
        background-color: transparent;
        border-radius: 6px;
        color: #ffffff !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(147, 112, 219, 0.5) !important;
        color: #ffffff !important;
    }
    
    /* Compact dataframe */
    .dataframe {
        font-size: 0.75rem !important;
        color: #ffffff !important;
    }
    
    /* Info/Warning boxes - minimal padding */
    .stAlert {
        padding: 0.5rem !important;
        margin: 0.25rem 0 !important;
        color: #ffffff !important;
    }
    
    /* Remove default Streamlit padding */
    section.main > div {
        padding-top: 0 !important;
    }
    
    /* Compact selectbox */
    .stSelectbox {
        margin-bottom: 0.25rem !important;
    }
    
    .stSelectbox label {
        color: #ffffff !important;
    }
    
    /* SELECTBOX (HISTORICAL TAB) DROPDOWN STYLING - PURPLE THEME */
    /* Dropdown container/input */
    div[data-testid="stSelectbox"] > div > div {
        background-color: rgba(147, 112, 219, 0.2) !important;
        border: 1px solid rgba(147, 112, 219, 0.5) !important;
        border-radius: 8px !important;
    }
    
    /* Dropdown input text color */
    div[data-testid="stSelectbox"] > div > div > div {
        color: #ffffff !important;
    }
    
    /* Dropdown menu popup - dark purple background */
    div[data-testid="stSelectbox"] div[role="listbox"] {
        background-color: #2d1b4e !important;
        border: 1px solid rgba(147, 112, 219, 0.5) !important;
        border-radius: 8px !important;
    }
    
    /* Dropdown options */
    div[data-testid="stSelectbox"] div[role="option"] {
        background-color: transparent !important;
        color: #ffffff !important;
    }
    
    /* Dropdown option hover - lighter purple */
    div[data-testid="stSelectbox"] div[role="option"]:hover {
        background-color: rgba(147, 112, 219, 0.4) !important;
        color: #ffffff !important;
    }
    
    /* Selected option in dropdown */
    div[data-testid="stSelectbox"] div[role="option"][aria-selected="true"] {
        background-color: rgba(147, 112, 219, 0.6) !important;
        color: #ffffff !important;
    }
    
    /* Dropdown arrow icon */
    div[data-testid="stSelectbox"] svg {
        fill: #9370db !important;
    }
    
    /* Compact multiselect */
    .stMultiSelect {
        margin-bottom: 0.25rem !important;
    }
    
    .stMultiSelect label {
        color: #ffffff !important;
    }
    
    /* MULTISELECT (POLLUTANTS TAB) DROPDOWN STYLING - PURPLE THEME */
    /* Multiselect container/input box */
    div[data-testid="stMultiSelect"] > div > div {
        background-color: rgba(147, 112, 219, 0.2) !important;
        border: 1px solid rgba(147, 112, 219, 0.5) !important;
        border-radius: 8px !important;
    }
    
    /* Multiselect selected tags/chips */
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background-color: rgba(147, 112, 219, 0.6) !important;
        color: #ffffff !important;
        border: 1px solid rgba(147, 112, 219, 0.8) !important;
    }
    
    /* Multiselect tag close button */
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] button {
        color: #ffffff !important;
    }
    
    /* Multiselect input text */
    div[data-testid="stMultiSelect"] input {
        color: #ffffff !important;
    }
    
    /* Multiselect dropdown menu popup - DARK PURPLE BACKGROUND */
    div[data-testid="stMultiSelect"] div[role="listbox"] {
        background-color: #2d1b4e !important;
        border: 1px solid rgba(147, 112, 219, 0.5) !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Multiselect dropdown options */
    div[data-testid="stMultiSelect"] div[role="option"] {
        background-color: transparent !important;
        color: #ffffff !important;
    }
    
    /* Multiselect dropdown option hover */
    div[data-testid="stMultiSelect"] div[role="option"]:hover {
        background-color: rgba(147, 112, 219, 0.4) !important;
        color: #ffffff !important;
    }
    
    /* Multiselect selected option in dropdown */
    div[data-testid="stMultiSelect"] div[role="option"][aria-selected="true"] {
        background-color: rgba(147, 112, 219, 0.6) !important;
        color: #ffffff !important;
    }
    
    /* Multiselect dropdown arrow icon */
    div[data-testid="stMultiSelect"] svg {
        fill: #9370db !important;
    }
    
    /* All text white */
    p, span, div, label {
        color: #ffffff !important;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def get_aqi_category(aqi):
    """Return AQI category, color, and emoji"""
    try:
        aqi = float(aqi)
        if aqi <= 50:
            return "Good", "#22c55e", "😊"
        elif aqi <= 100:
            return "Moderate", "#ffc107", "😐"
        elif aqi <= 150:
            return "Sensitive", "#fb923c", "😷"
        elif aqi <= 200:
            return "Unhealthy", "#ef4444", "😨"
        elif aqi <= 300:
            return "Very Unhealthy", "#991b1b", "😱"
        else:
            return "Hazardous", "#7f1d1d", "☠️"
    except:
        return "Unknown", "#808080", "❓"

def get_health_message(aqi):
    """Return health advisory based on AQI"""
    try:
        aqi = float(aqi)
        if aqi <= 50:
            return "Excellent air quality! Perfect for outdoor activities."
        elif aqi <= 100:
            return "Acceptable air quality. Enjoy your day with minor precautions."
        elif aqi <= 150:
            return "Sensitive groups should reduce prolonged outdoor activities."
        elif aqi <= 200:
            return "Unhealthy air. Limit outdoor activities."
        elif aqi <= 300:
            return "Very Unhealthy! Health alert for all."
        else:
            return "HAZARDOUS! Emergency conditions. Stay indoors!"
    except:
        return "Unable to determine health advisory."

@st.cache_resource(show_spinner=False)
def load_xgboost_model():
    """Load XGBoost model from local file"""
    try:
        from joblib import load
        
        possible_paths = [
            "models/xgboost.pkl",
            "../models/xgboost.pkl",
            "./models/xgboost.pkl",
            "xgboost.pkl"
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            st.error("❌ XGBoost model file not found")
            return None, None
        
        model = load(model_path)
        
        # Try to load metadata
        metadata = None
        metadata_path = model_path.replace(".pkl", "_metadata.json")
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            # Default features list
            metadata = {
                'features_used': [
                    'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide',
                    'ozone', 'sulphur_dioxide', 'temperature_2m',
                    'relative_humidity_2m', 'wind_speed_10m', 'month',
                    'hour', 'day', 'weekday', 'hour_sin', 'aqi_change_rate',
                    'pm_ratio', 'temp_humidity_ratio', 'wind_effect'
                ]
            }
        
        st.success(f"✅ Loaded XGBoost model from {model_path}")
        return model, metadata
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data_from_hopsworks():
    """Fetch latest data from Hopsworks Feature Store"""
    try:
        import hopsworks
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("HOPSWORKS_API_KEY")
        
        if not api_key:
            st.warning("⚠️ HOPSWORKS_API_KEY not found. Using demo data.")
            return generate_demo_data()
        
        # Connect to Hopsworks
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
        
        # Get feature group
        fg = fs.get_feature_group("aqi_features", version=2)
        df = fg.read()
        
        # Ensure datetime column
        if 'datetime' not in df.columns:
            if 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])
            elif 'datetime_str' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime_str'])
                df.drop(columns=['datetime_str'], inplace=True)
            else:
                df['datetime'] = pd.date_range(end=datetime.now(), periods=len(df), freq='h')
        else:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Ensure AQI column exists
        if 'aqi' not in df.columns:
            if 'pm2_5' in df.columns:
                df['aqi'] = df['pm2_5'] * 1.5
            else:
                df['aqi'] = 100.0
        
        # Required feature columns (pollutants and weather)
        required_features = [
            'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide',
            'ozone', 'sulphur_dioxide', 'temperature_2m',
            'relative_humidity_2m', 'wind_speed_10m'
        ]
        
        # Add missing base features with defaults
        for col in required_features:
            if col not in df.columns:
                st.warning(f"⚠️ Missing feature '{col}'. Adding default values.")
                df[col] = 0.0
        
        # Calculate time-based features if missing
        if 'month' not in df.columns:
            df['month'] = df['datetime'].dt.month
        if 'hour' not in df.columns:
            df['hour'] = df['datetime'].dt.hour
        if 'day' not in df.columns:
            df['day'] = df['datetime'].dt.day
        if 'weekday' not in df.columns:
            df['weekday'] = df['datetime'].dt.weekday
        if 'hour_sin' not in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        
        # Calculate engineered features if missing
        if 'pm_ratio' not in df.columns:
            df['pm_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-5)
        
        if 'temp_humidity_ratio' not in df.columns:
            df['temp_humidity_ratio'] = df['temperature_2m'] / (df['relative_humidity_2m'] + 1e-5)
        
        if 'wind_effect' not in df.columns:
            df['wind_effect'] = -df['wind_speed_10m'] * 0.5
        
        if 'aqi_change_rate' not in df.columns:
            df['aqi_change_rate'] = df['aqi'].diff().fillna(0)
        
        st.success(f"✅ Loaded {len(df)} records from Hopsworks Feature Store")
        return df
        
    except Exception as e:
        st.warning(f"⚠️ Hopsworks error: {str(e)}. Using demo data.")
        return generate_demo_data()

def generate_demo_data():
    """Generate realistic demo data with all required features"""
    dates = pd.date_range(end=datetime.now(), periods=720, freq='h')  # 30 days hourly
    
    # Create base dataframe
    df = pd.DataFrame({
        'datetime': dates,
        'aqi': np.random.uniform(50, 150, 720),
        'pm2_5': np.random.uniform(20, 80, 720),
        'pm10': np.random.uniform(30, 120, 720),
        'carbon_monoxide': np.random.uniform(300, 900, 720),
        'nitrogen_dioxide': np.random.uniform(15, 70, 720),
        'ozone': np.random.uniform(25, 110, 720),
        'sulphur_dioxide': np.random.uniform(8, 35, 720),
        'temperature_2m': np.random.uniform(20, 35, 720),
        'relative_humidity_2m': np.random.uniform(40, 80, 720),
        'wind_speed_10m': np.random.uniform(5, 15, 720),
        'month': dates.month,
        'hour': dates.hour,
        'day': dates.day,
        'weekday': dates.weekday,
        'hour_sin': np.sin(2 * np.pi * dates.hour / 24),
        'aqi_change_rate': np.random.uniform(-10, 10, 720),
        'pm_ratio': np.random.uniform(0.45, 0.75, 720),
        'temp_humidity_ratio': np.random.uniform(0.35, 0.85, 720),
        'wind_effect': np.random.uniform(-12, 12, 720)
    })
    
    return df

def predict_future_aqi(model, df, metadata=None, hours=72):
    """Generate predictions for next 72 hours (3 days) - FIXED VERSION"""
    try:
        last_date = df["datetime"].max()
        future_dates = [last_date + timedelta(hours=i) for i in range(1, hours + 1)]
        
        # Get expected features from model - PRIORITY ORDER MATTERS
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
        elif metadata and 'features_used' in metadata:
            expected_features = metadata['features_used']
        else:
            expected_features = [
                'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide',
                'ozone', 'sulphur_dioxide', 'temperature_2m',
                'relative_humidity_2m', 'wind_speed_10m', 'month',
                'hour', 'day', 'weekday', 'hour_sin', 'aqi_change_rate',
                'pm_ratio', 'temp_humidity_ratio', 'wind_effect'
            ]
        
        # Get the last row as base features
        last_row = df.iloc[-1].copy()
        
        # Create future data by copying the last row and modifying time features
        future_data_list = []
        
        for i, future_date in enumerate(future_dates):
            row = last_row.copy()
            
            # Update time-based features
            row['hour'] = future_date.hour
            row['day'] = future_date.day
            row['month'] = future_date.month
            row['weekday'] = future_date.weekday()
            row['hour_sin'] = np.sin(2 * np.pi * future_date.hour / 24)
            
            # Add small random variation to environmental features (±3%)
            vary_cols = [
                "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                "ozone", "sulphur_dioxide", "temperature_2m", 
                "relative_humidity_2m", "wind_speed_10m"
            ]
            
            for col in vary_cols:
                if col in row.index:
                    noise = np.random.normal(0, 0.03)
                    row[col] = row[col] * (1 + noise)
            
            future_data_list.append(row)
        
        # Create DataFrame from list of Series - ENSURE CORRECT COLUMN ORDER
        future_data = pd.DataFrame(future_data_list)
        
        # Ensure ALL expected features exist and are in CORRECT ORDER
        for feat in expected_features:
            if feat not in future_data.columns:
                if 'sin' in feat or 'cos' in feat:
                    future_data[feat] = 0.0
                elif 'ratio' in feat:
                    future_data[feat] = 0.5
                elif feat in ['hour', 'day', 'month', 'weekday']:
                    future_data[feat] = 1
                else:
                    future_data[feat] = 0.0
        
        # REORDER COLUMNS TO MATCH EXACTLY WHAT MODEL EXPECTS
        future_data = future_data[expected_features]
        
        # Generate predictions with validate_features=False to avoid strict checking
        predictions = model.predict(future_data, validate_features=False)
        predictions = np.clip(predictions, 0, 500)
        
        return pd.DataFrame({
            'datetime': future_dates,
            'predicted_aqi': predictions
        })
        
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def create_forecast_table(predictions):
    """Create formatted forecast table with AQI status"""
    if predictions is None:
        return None
    
    # Group by day
    predictions['date'] = predictions['datetime'].dt.date
    predictions['time'] = predictions['datetime'].dt.strftime('%H:%M')
    predictions['day_name'] = predictions['datetime'].dt.strftime('%A')
    
    # Get AQI status for each prediction
    predictions['status'] = predictions['predicted_aqi'].apply(
        lambda x: get_aqi_category(x)[0]
    )
    predictions['emoji'] = predictions['predicted_aqi'].apply(
        lambda x: get_aqi_category(x)[2]
    )
    
    # Create daily summary
    daily_summary = []
    for date in sorted(predictions['date'].unique()):
        day_data = predictions[predictions['date'] == date]
        avg_aqi = day_data['predicted_aqi'].mean()
        min_aqi = day_data['predicted_aqi'].min()
        max_aqi = day_data['predicted_aqi'].max()
        status, color, emoji = get_aqi_category(avg_aqi)
        
        daily_summary.append({
            'Date': date.strftime('%b %d'),
            'Day': day_data['day_name'].iloc[0],
            'Avg AQI': f"{avg_aqi:.0f}",
            'Min AQI': f"{min_aqi:.0f}",
            'Max AQI': f"{max_aqi:.0f}",
            'Status': f"{emoji} {status}",
            'Color': color
        })
    
    return pd.DataFrame(daily_summary), predictions

# Main Application
def main():
    # Compact Title Banner
    st.markdown("""
    <div style='background: linear-gradient(90deg, rgba(147,112,219,0.2) 0%, rgba(167,139,250,0.2) 100%); 
                border-radius: 8px; padding: 0.5rem; margin-bottom: 0.5rem; text-align: center;'>
        <h1 style='margin:0; padding:0; color: #9370db; font-size: 1.5rem;'>🌍 Karachi AQI Predictor</h1>
        <p style='margin:0; padding:0; color: #a78bfa; font-size: 0.75rem;'>
            Real-time Air Quality Monitoring system using ML Algorithms 
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and data
    with st.spinner("Loading model and data..."):
        model, metadata = load_xgboost_model()
        df = fetch_data_from_hopsworks()
    
    if model is None:
        st.error("❌ Failed to load model. Cannot proceed.")
        return
    
    # Current AQI Dashboard
    current_aqi = float(df['aqi'].iloc[-1]) if 'aqi' in df.columns else 0
    category, color, emoji = get_aqi_category(current_aqi)
    health_msg = get_health_message(current_aqi)
    
    # Ultra-Compact Top Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Current AQI", f"{current_aqi:.0f}")
    
    with col2:
        st.markdown(f"""
        <div style='background: {color}20; border: 1px solid {color}; border-radius: 8px; 
                    padding: 0.5rem; text-align: center; height: 100%;'>
            <div style='font-size: 1.5rem;'>{emoji} {category}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        temp = df['temperature_2m'].iloc[-1] if 'temperature_2m' in df.columns else 0
        st.metric("Temperature", f"{temp:.1f}°C")
    
    with col4:
        humidity = df['relative_humidity_2m'].iloc[-1] if 'relative_humidity_2m' in df.columns else 0
        st.metric("Humidity", f"{humidity:.0f}%")
    
    with col5:
        wind = df['wind_speed_10m'].iloc[-1] if 'wind_speed_10m' in df.columns else 0
        st.metric("Wind Speed", f"{wind:.1f} m/s")
    
    # Health Advisory
    st.info(f"💡 **Health Advisory:** {health_msg}")
    
    # Main Tabs - Compact layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Current Status",
        "🔮 3-Day Forecast",
        "📈 Historical",
        "🧪 Pollutants"
    ])
    
    # TAB 1: Current Status
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Recent AQI Trend (24h)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['datetime'].tail(24),
                y=df['aqi'].tail(24),
                mode='lines',
                name='AQI',
                line=dict(color='#9370db', width=2),
                fill='tozeroy',
                fillcolor='rgba(147, 112, 219, 0.2)'
            ))
            
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(48, 25, 52, 0.3)',
                xaxis_title=None,
                yaxis_title="AQI",
                hovermode='x',
                height=220,
                margin=dict(l=5, r=5, t=5, b=5),
                font=dict(color='#ffffff', size=10)
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            st.markdown("### Current Pollutants")
            pollutants = ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'ozone']
            for pollutant in pollutants:
                if pollutant in df.columns:
                    value = df[pollutant].iloc[-1]
                    st.metric(
                        pollutant.replace('_', ' ').title(),
                        f"{value:.1f} µg/m³"
                    )
    
    # TAB 2: 3-Day Forecast
    with tab2:
        st.markdown("### 72-Hour AQI Forecast")
        
        predictions = predict_future_aqi(model, df, metadata, hours=72)
        
        if predictions is not None:
            # Generate forecast table
            daily_summary_df, hourly_predictions = create_forecast_table(predictions)
            
            # Display daily summary
            st.markdown("##### Daily Summary")
            cols = st.columns(3)
            for idx, (_, row) in enumerate(daily_summary_df.iterrows()):
                with cols[idx]:
                    st.markdown(f"""
                    <div style='background: rgba(147,112,219,0.1); border: 1px solid rgba(147,112,219,0.3); 
                                border-radius: 8px; padding: 0.5rem; text-align: center;'>
                        <div style='font-size: 0.9rem; font-weight: bold;'>{row['Day']}</div>
                        <div style='font-size: 0.75rem; color: #a78bfa;'>{row['Date']}</div>
                        <div style='font-size: 1.5rem; color: {row['Color']}; margin: 0.25rem 0;'>{row['Avg AQI']}</div>
                        <div style='font-size: 0.85rem;'>{row['Status']}</div>
                        <div style='font-size: 0.7rem; color: #a78bfa;'>{row['Min AQI']} - {row['Max AQI']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Hourly predictions table
            st.markdown("##### Hourly Predictions")
            display_hourly = hourly_predictions.copy()
            display_hourly['Time'] = display_hourly['datetime'].dt.strftime('%m/%d %H:%M')
            display_hourly['AQI'] = display_hourly['predicted_aqi'].round(0).astype(int)
            display_hourly = display_hourly[['Time', 'AQI', 'status', 'emoji']]
            display_hourly.columns = ['Time', 'AQI', 'Status', '']
            
            st.dataframe(
                display_hourly,
                use_container_width=True,
                height=250,
                hide_index=True
            )
            
            # Forecast chart
            st.markdown("##### Forecast Trend")
            fig = go.Figure()
            
            # Historical (last 24h)
            if len(df) >= 24:
                fig.add_trace(go.Scatter(
                    x=df['datetime'].tail(24),
                    y=df['aqi'].tail(24),
                    mode='lines',
                    name='Historical',
                    line=dict(color='#5eead4', width=1.5)
                ))
            
            # Predictions
            fig.add_trace(go.Scatter(
                x=hourly_predictions['datetime'],
                y=hourly_predictions['predicted_aqi'],
                mode='lines',
                name='Forecast',
                line=dict(color='#fb923c', width=2, dash='dash')
            ))
            
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(48, 25, 52, 0.3)',
                xaxis_title="Time",
                yaxis_title="AQI",
                hovermode='x',
                height=220,
                margin=dict(l=5, r=5, t=5, b=5),
                legend=dict(font=dict(size=10))
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # TAB 3: Historical Data
    with tab3:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### AQI Historical Trend")
            time_range = st.selectbox(
                "Time Range",
                ["Last 24h", "Last 3d", "Last 7d", "Last 30d"],
                key="history_range"
            )
            
            if time_range == "Last 24h":
                filtered = df.tail(24)
            elif time_range == "Last 3d":
                filtered = df.tail(72)
            elif time_range == "Last 7d":
                filtered = df.tail(168)
            else:
                filtered = df.tail(720)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered['datetime'],
                y=filtered['aqi'],
                mode='lines',
                name='AQI',
                line=dict(color='#9370db', width=2),
                fill='tozeroy',
                fillcolor='rgba(147, 112, 219, 0.2)'
            ))
            
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(48, 25, 52, 0.3)',
                xaxis_title=None,
                yaxis_title="AQI",
                hovermode='x',
                height=270,
                margin=dict(l=5, r=5, t=5, b=5)
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            st.markdown("### Statistics")
            if len(filtered) > 0:
                st.metric("Average", f"{filtered['aqi'].mean():.1f}")
                st.metric("Maximum", f"{filtered['aqi'].max():.0f}")
                st.metric("Minimum", f"{filtered['aqi'].min():.0f}")
                st.metric("Std Dev", f"{filtered['aqi'].std():.1f}")
    
    # TAB 4: Pollutant Analysis
    with tab4:
        pollutants = ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'ozone', 'sulphur_dioxide']
        available = [p for p in pollutants if p in df.columns]
        
        if available:
            selected = st.multiselect(
                "Select Pollutants to Display",
                available,
                default=available[:3] if len(available) >= 3 else available
            )
            
            if selected:
                fig = go.Figure()
                colors = ['#9370db', '#fb923c', '#22c55e', '#fbbf24', '#a78bfa', '#f472b6']
                
                for idx, col in enumerate(selected):
                    fig.add_trace(go.Scatter(
                        x=df['datetime'].tail(168),
                        y=df[col].tail(168),
                        mode='lines',
                        name=col.replace('_', ' ').title(),
                        line=dict(color=colors[idx % len(colors)], width=1.5)
                    ))
                
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(48, 25, 52, 0.3)',
                    xaxis_title="Time",
                    yaxis_title="Concentration (µg/m³)",
                    hovermode='x',
                    height=270,
                    margin=dict(l=5, r=5, t=5, b=5)
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Current levels in columns
                cols = st.columns(len(selected))
                for idx, col in enumerate(selected):
                    with cols[idx]:
                        current_val = df[col].iloc[-1]
                        st.metric(
                            col.replace('_', ' ').title(),
                            f"{current_val:.1f} µg/m³"
                        )
    
    # Compact Footer
    st.markdown(f"""
    <div style='text-align: center; color: #a78bfa; font-size: 0.7rem; 
                margin-top: 0.5rem; padding: 0.5rem; border-top: 1px solid rgba(147,112,219,0.3);'>
        🌍 Karachi AQI Prediction System | Keeping an eye on Karachi's air so you don't have to<br>
        Last updated: {datetime.now().strftime("%b %d, %Y %H:%M")}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()