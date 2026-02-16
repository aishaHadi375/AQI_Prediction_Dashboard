from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from joblib import load
import os
import logging
from functools import lru_cache
from contextlib import asynccontextmanager

# ===========================
# LOGGING CONFIGURATION
# ===========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aqi_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===========================
# GLOBAL STATE (Thread-Safe)
# ===========================
class AppState:
    """Thread-safe application state"""
    def __init__(self):
        self.model = None
        self.metadata = None
        self.feature_store_data = None
        self.last_refresh = None
    
app_state = AppState()

# ===========================
# LIFESPAN CONTEXT MANAGER
# ===========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting Karachi AQI Prediction API...")
    try:
        load_model()
        fetch_feature_store_data()
        app_state.last_refresh = datetime.now()
        logger.info("API started successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        # Continue with demo data
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")

# ===========================
# FASTAPI APP INITIALIZATION
# ===========================
app = FastAPI(
    title="Karachi AQI Prediction API",
    description="Real-time Air Quality Index predictions for Karachi with 72-hour forecasting",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ===========================
# CORS MIDDLEWARE
# ===========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production: ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# RATE LIMITING (Optional)
# ===========================
# Uncomment to enable rate limiting
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.util import get_remote_address
# from slowapi.errors import RateLimitExceeded
#
# limiter = Limiter(key_func=get_remote_address)
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ===========================
# PYDANTIC MODELS
# ===========================
class PredictionRequest(BaseModel):
    hours: int = Field(default=72, ge=1, le=168, description="Number of hours to predict (1-168)")
    model_name: str = Field(default="random_forest", description="Model name to use")
    
    @validator('hours')
    def validate_hours(cls, v):
        if v < 1 or v > 168:
            raise ValueError('Hours must be between 1 and 168 (7 days)')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "hours": 72,
                "model_name": "random_forest"
            }
        }

class PredictionData(BaseModel):
    datetime: str
    predicted_aqi: float

class DailySummary(BaseModel):
    date: str
    avg_aqi: float
    min_aqi: float
    max_aqi: float

class PredictionResponse(BaseModel):
    predictions: List[PredictionData]
    model_used: str
    generated_at: str
    daily_summary: List[DailySummary]
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {"datetime": "2024-02-08T13:00:00", "predicted_aqi": 128.3}
                ],
                "model_used": "random_forest",
                "generated_at": "2024-02-08T12:00:00",
                "daily_summary": [
                    {
                        "date": "2024-02-08",
                        "avg_aqi": 132.5,
                        "min_aqi": 115.2,
                        "max_aqi": 148.7
                    }
                ]
            }
        }

class PollutantData(BaseModel):
    pm2_5: float
    pm10: float
    carbon_monoxide: float
    nitrogen_dioxide: float
    ozone: float
    sulphur_dioxide: float

class CurrentAQIResponse(BaseModel):
    current_aqi: float
    category: str
    health_message: str
    timestamp: str
    pollutants: PollutantData
    
    class Config:
        schema_extra = {
            "example": {
                "current_aqi": 125.5,
                "category": "Unhealthy for Sensitive Groups",
                "health_message": "Sensitive groups should consider reducing prolonged outdoor activities.",
                "timestamp": "2024-02-08T12:00:00",
                "pollutants": {
                    "pm2_5": 62.3,
                    "pm10": 105.7,
                    "carbon_monoxide": 450.2,
                    "nitrogen_dioxide": 42.1,
                    "ozone": 68.5,
                    "sulphur_dioxide": 18.3
                }
            }
        }

class HistoricalDataResponse(BaseModel):
    data: List[Dict]
    total_records: int
    time_range: str

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    data_loaded: bool
    data_freshness: Optional[str]
    timestamp: str

class RefreshResponse(BaseModel):
    status: str
    message: str
    records: int
    timestamp: str

# ===========================
# HELPER FUNCTIONS
# ===========================
def get_aqi_category(aqi: float) -> Tuple[str, str]:
    """
    Return AQI category and color
    
    Args:
        aqi: Air Quality Index value
        
    Returns:
        Tuple of (category_name, hex_color)
    """
    if aqi <= 50:
        return "Good", "#28a745"
    elif aqi <= 100:
        return "Moderate", "#ffc107"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#fd7e14"
    elif aqi <= 200:
        return "Unhealthy", "#e74c3c"
    elif aqi <= 300:
        return "Very Unhealthy", "#8b0000"
    else:
        return "Hazardous", "#4b0082"

def get_health_message(aqi: float) -> str:
    """
    Return health advisory message based on AQI
    
    Args:
        aqi: Air Quality Index value
        
    Returns:
        Health advisory string
    """
    if aqi <= 50:
        return "Air quality is excellent! Perfect for outdoor activities."
    elif aqi <= 100:
        return "Air quality is acceptable. Enjoy your day with minor precautions."
    elif aqi <= 150:
        return "Sensitive groups should consider reducing prolonged outdoor activities."
    elif aqi <= 200:
        return "Unhealthy air. Everyone should limit outdoor activities."
    elif aqi <= 300:
        return "Very Unhealthy! Health alert for all. Avoid outdoor exposure."
    else:
        return "HAZARDOUS! Emergency conditions. Stay indoors immediately!"

@lru_cache(maxsize=1)
def load_model(model_name: str = "random_forest") -> bool:
    """
    Load ML model and metadata with caching
    
    Args:
        model_name: Name of the model file (without .pkl extension)
        
    Returns:
        True if successful
        
    Raises:
        HTTPException if model loading fails
    """
    logger.info(f"Loading model: {model_name}")
    
    try:
        # Check multiple possible paths
        possible_paths = [
            f"models/{model_name}.pkl",
            f"../models/{model_name}.pkl",
            f"./models/{model_name}.pkl",
            os.path.join(os.path.dirname(__file__), "models", f"{model_name}.pkl")
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                logger.info(f"Found model at: {path}")
                break
        
        if not model_path:
            error_msg = f"Model not found: {model_name}.pkl in any of {possible_paths}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Load model
        app_state.model = load(model_path)
        logger.info(f"Model loaded successfully: {type(app_state.model).__name__}")
        
        # Load metadata if available
        metadata_path = os.path.join(os.path.dirname(model_path), "training_metadata.json")
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                app_state.metadata = json.load(f)
            logger.info(f"Metadata loaded: {len(app_state.metadata.get('features_used', []))} features")
        else:
            logger.warning("No metadata file found")
            app_state.metadata = None
        
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_name}.pkl"
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )

def fetch_feature_store_data():
    """
    Fetch data from Hopsworks Feature Store
    Falls back to demo data if connection fails
    """
    logger.info("Fetching data from feature store...")
    
    try:
        import hopsworks
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("HOPSWORKS_API_KEY")
        
        if not api_key:
            logger.warning("No Hopsworks API key found, using demo data")
            app_state.feature_store_data = generate_demo_data()
            return
        
        # Connect to Hopsworks
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
        
        # Get feature group
        feature_group_name = os.getenv("FEATURE_GROUP_NAME", "aqi_features")
        feature_group_version = int(os.getenv("FEATURE_GROUP_VERSION", "2"))
        
        fg = fs.get_feature_group(feature_group_name, version=feature_group_version)
        df = fg.read()
        
        logger.info(f"Fetched {len(df)} records from feature store")
        
        # Process datetime column
        if "datetime_str" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime_str"])
            df.drop(columns=["datetime_str"], inplace=True)
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        
        # Sort by datetime
        app_state.feature_store_data = df.sort_values("datetime").reset_index(drop=True)
        app_state.last_refresh = datetime.now()
        
        logger.info(f"Feature store data loaded successfully. Latest record: {df['datetime'].max()}")
        
    except ImportError:
        logger.warning("Hopsworks library not installed, using demo data")
        app_state.feature_store_data = generate_demo_data()
    except Exception as e:
        logger.error(f"Feature store error: {e}. Using demo data.")
        app_state.feature_store_data = generate_demo_data()

def generate_demo_data() -> pd.DataFrame:
    """
    Generate realistic demo data for testing
    
    Returns:
        DataFrame with synthetic AQI data
    """
    logger.info("Generating demo data...")
    
    dates = pd.date_range(end=datetime.now(), periods=720, freq='H')
    
    # Generate realistic AQI patterns
    base_aqi = 120
    hourly_pattern = np.sin(np.arange(720) * 2 * np.pi / 24) * 25  # Daily cycle
    weekly_pattern = np.sin(np.arange(720) * 2 * np.pi / 168) * 15  # Weekly cycle
    noise = np.random.normal(0, 12, 720)  # Random noise
    
    aqi_values = np.clip(base_aqi + hourly_pattern + weekly_pattern + noise, 30, 350)
    
    df = pd.DataFrame({
        'datetime': dates,
        'aqi': aqi_values,
        'pm2_5': aqi_values * 0.48 + np.random.normal(0, 8, 720),
        'pm10': aqi_values * 0.82 + np.random.normal(0, 12, 720),
        'temperature_2m': 25 + 8 * np.sin(np.arange(720) * 2 * np.pi / 24) + np.random.normal(0, 2, 720),
        'relative_humidity_2m': 60 + 15 * np.sin(np.arange(720) * 2 * np.pi / 24 + np.pi) + np.random.normal(0, 5, 720),
        'wind_speed_10m': np.abs(np.random.normal(8, 4, 720)),
        'carbon_monoxide': np.random.uniform(300, 900, 720),
        'nitrogen_dioxide': np.random.uniform(15, 70, 720),
        'ozone': np.random.uniform(25, 110, 720),
        'sulphur_dioxide': np.random.uniform(8, 35, 720),
        'hour': dates.hour,
        'day': dates.day,
        'month': dates.month,
        'weekday': dates.weekday,
        'hour_sin': np.sin(2 * np.pi * dates.hour / 24),
        'hour_cos': np.cos(2 * np.pi * dates.hour / 24),
        'pm_ratio': np.random.uniform(0.45, 0.75, 720),
        'temp_humidity_ratio': np.random.uniform(0.35, 0.85, 720),
        'wind_effect': np.random.uniform(-12, 12, 720)
    })
    
    logger.info(f"Generated {len(df)} demo records")
    return df

def prepare_features(df: pd.DataFrame, metadata: Optional[Dict] = None) -> pd.DataFrame:
    """
    Prepare features for prediction from the latest row
    
    Args:
        df: DataFrame with historical data
        metadata: Optional model metadata with feature list
        
    Returns:
        DataFrame with single row of features
    """
    last_row = df.iloc[-1].copy()
    
    # Default features
    expected_features = [
        'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'ozone',
        'sulphur_dioxide', 'temperature_2m', 'relative_humidity_2m',
        'wind_speed_10m', 'hour', 'day', 'month', 'weekday',
        'hour_sin', 'hour_cos', 'pm_ratio', 'temp_humidity_ratio', 'wind_effect'
    ]
    
    # Use metadata features if available
    if metadata and 'features_used' in metadata:
        expected_features = metadata['features_used']
    
    # Extract features with fallback to 0
    features = {feat: last_row.get(feat, 0.0) for feat in expected_features}
    
    return pd.DataFrame([features])

# ===========================
# EXCEPTION HANDLERS
# ===========================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail} - Path: {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# ===========================
# API ENDPOINTS
# ===========================
@app.get("/", tags=["Info"])
async def root():
    """
    API root endpoint with basic information
    """
    return {
        "message": "Karachi AQI Prediction API",
        "version": "2.0.0",
        "description": "Real-time Air Quality Index predictions for Karachi",
        "endpoints": {
            "/health": "System health check",
            "/current": "Get current AQI and pollutants",
            "/predict": "Generate AQI forecast (POST)",
            "/historical": "Retrieve historical data",
            "/refresh": "Refresh data from feature store (POST)",
            "/docs": "Interactive API documentation",
            "/redoc": "Alternative API documentation"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check():
    """
    System health check endpoint
    
    Returns current system status and data freshness
    """
    health_status = {
        "status": "healthy",
        "model_loaded": app_state.model is not None,
        "data_loaded": app_state.feature_store_data is not None,
        "data_freshness": None,
        "timestamp": datetime.now().isoformat()
    }
    
    # Check data freshness
    if app_state.feature_store_data is not None:
        try:
            latest_data = app_state.feature_store_data['datetime'].max()
            age = datetime.now() - latest_data
            
            if age > timedelta(hours=3):
                health_status["status"] = "degraded"
                health_status["data_freshness"] = f"Data is {age.total_seconds() / 3600:.1f} hours old (stale)"
            else:
                health_status["data_freshness"] = f"Data is {age.total_seconds() / 3600:.1f} hours old (fresh)"
        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
            health_status["data_freshness"] = "Unable to determine"
    
    # Set overall status
    if not health_status["model_loaded"] or not health_status["data_loaded"]:
        health_status["status"] = "unhealthy"
    
    return health_status

@app.get("/current", response_model=CurrentAQIResponse, tags=["AQI"])
# @limiter.limit("30/minute")  # Uncomment to enable rate limiting
async def get_current_aqi(request: Request):
    """
    Get current AQI and pollutant concentrations
    
    Returns the most recent AQI reading with:
    - Current AQI value
    - Health category
    - Health advisory message
    - Individual pollutant concentrations
    """
    if app_state.feature_store_data is None:
        logger.error("No data available for current AQI request")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data not available. Please try again later or use /refresh endpoint."
        )
    
    try:
        latest = app_state.feature_store_data.iloc[-1]
        current_aqi = float(latest['aqi'])
        category, _ = get_aqi_category(current_aqi)
        health_msg = get_health_message(current_aqi)
        
        # Extract pollutant data
        pollutant_cols = ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'ozone', 'sulphur_dioxide']
        pollutants = {col: float(latest.get(col, 0)) for col in pollutant_cols}
        
        logger.info(f"Current AQI requested: {current_aqi} ({category})")
        
        return CurrentAQIResponse(
            current_aqi=current_aqi,
            category=category,
            health_message=health_msg,
            timestamp=latest['datetime'].isoformat(),
            pollutants=PollutantData(**pollutants)
        )
        
    except KeyError as e:
        logger.error(f"Missing required column: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data format error: Missing column {e}"
        )
    except Exception as e:
        logger.error(f"Error getting current AQI: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving current AQI: {str(e)}"
        )

@app.post("/predict", response_model=PredictionResponse, tags=["AQI"])
# @limiter.limit("10/minute")  # Uncomment to enable rate limiting
async def predict_aqi(request: Request, prediction_request: PredictionRequest):
    """
    Generate AQI forecast for specified number of hours
    
    Request body:
    - hours: Number of hours to predict (1-168)
    - model_name: Model to use (default: random_forest)
    
    Returns:
    - Hourly predictions
    - Daily summary statistics
    - Model metadata
    """
    # Ensure model is loaded
    if app_state.model is None:
        logger.info("Model not loaded, loading now...")
        load_model(prediction_request.model_name)
    
    if app_state.feature_store_data is None:
        logger.error("No data available for prediction")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data not available. Please try again later or use /refresh endpoint."
        )
    
    try:
        logger.info(f"Prediction requested: {prediction_request.hours} hours using {prediction_request.model_name}")
        
        # Get last datetime
        last_date = app_state.feature_store_data['datetime'].max()
        
        # Generate future dates
        future_dates = [last_date + timedelta(hours=i) for i in range(1, prediction_request.hours + 1)]
        
        # Prepare base features
        base_features = prepare_features(app_state.feature_store_data, app_state.metadata)
        
        # Generate predictions iteratively
        future_data_list = []
        for future_date in future_dates:
            row = base_features.iloc[0].copy()
            
            # Update temporal features
            row['hour'] = future_date.hour
            row['day'] = future_date.day
            row['month'] = future_date.month
            row['weekday'] = future_date.weekday()
            row['hour_sin'] = np.sin(2 * np.pi * future_date.hour / 24)
            
            if 'hour_cos' in row.index:
                row['hour_cos'] = np.cos(2 * np.pi * future_date.hour / 24)
            
            # Add small variations to environmental features (±4%)
            env_features = ['pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'ozone', 'sulphur_dioxide']
            for col in env_features:
                if col in row.index:
                    row[col] *= (1 + np.random.normal(0, 0.04))
            
            future_data_list.append(row)
        
        # Create DataFrame and predict
        future_df = pd.DataFrame(future_data_list)
        predictions = app_state.model.predict(future_df)
        
        # Clip predictions to valid AQI range
        predictions = np.clip(predictions, 0, 500)
        
        # Format predictions
        pred_list = [
            PredictionData(
                datetime=future_dates[i].isoformat(),
                predicted_aqi=float(predictions[i])
            )
            for i in range(len(predictions))
        ]
        
        # Calculate daily summaries
        pred_df = pd.DataFrame([p.dict() for p in pred_list])
        pred_df['datetime'] = pd.to_datetime(pred_df['datetime'])
        pred_df['date'] = pred_df['datetime'].dt.date
        
        daily_summary = pred_df.groupby('date')['predicted_aqi'].agg(['mean', 'min', 'max']).reset_index()
        
        daily_list = [
            DailySummary(
                date=str(row['date']),
                avg_aqi=float(row['mean']),
                min_aqi=float(row['min']),
                max_aqi=float(row['max'])
            )
            for _, row in daily_summary.iterrows()
        ]
        
        logger.info(f"Successfully generated {len(predictions)} predictions")
        
        return PredictionResponse(
            predictions=pred_list,
            model_used=prediction_request.model_name,
            generated_at=datetime.now().isoformat(),
            daily_summary=daily_list
        )
        
    except ValueError as e:
        logger.error(f"Validation error in prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/historical", response_model=HistoricalDataResponse, tags=["AQI"])
# @limiter.limit("20/minute")  # Uncomment to enable rate limiting
async def get_historical_data(
    request: Request,
    hours: int = 168
):
    """
    Retrieve historical AQI data
    
    Query parameters:
    - hours: Number of hours to retrieve (default: 168 = 7 days, max: 720 = 30 days)
    
    Returns historical data with timestamps
    """
    # Validate hours parameter
    if hours < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Hours must be at least 1"
        )
    
    if hours > 720:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Hours cannot exceed 720 (30 days)"
        )
    
    if app_state.feature_store_data is None:
        logger.error("No data available for historical request")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data not available. Please try again later or use /refresh endpoint."
        )
    
    try:
        # Get requested number of hours
        data = app_state.feature_store_data.tail(hours).copy()
        
        # Convert datetime to ISO format string
        data['datetime'] = data['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Convert to records
        records = data.to_dict('records')
        
        logger.info(f"Historical data requested: {len(records)} records ({hours} hours)")
        
        return HistoricalDataResponse(
            data=records,
            total_records=len(records),
            time_range=f"Last {hours} hours"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving historical data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving historical data: {str(e)}"
        )

@app.post("/refresh", response_model=RefreshResponse, tags=["System"])
# @limiter.limit("5/hour")  # Uncomment to enable rate limiting
async def refresh_data(request: Request):
    """
    Manually refresh data from feature store
    
    Use this endpoint to force a data refresh if you suspect the data is stale.
    Rate limited to prevent abuse.
    """
    try:
        logger.info("Manual data refresh requested")
        fetch_feature_store_data()
        
        record_count = len(app_state.feature_store_data) if app_state.feature_store_data is not None else 0
        
        return RefreshResponse(
            status="success",
            message="Data refreshed successfully",
            records=record_count,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error during data refresh: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Refresh failed: {str(e)}"
        )

# ===========================
# MAIN ENTRY POINT
# ===========================
if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment or use defaults
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    
    # Run the application
    uvicorn.run(
        "fast_api:app",  # FIXED: Correct module path
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )