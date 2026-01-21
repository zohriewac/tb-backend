"""FastAPI Backend for TurboBulk Progress Prediction

This API provides endpoints to train the progress prediction model and generate
predictions for users based on their workout data.

Endpoints:
  POST /train - Train the model (optionally for a specific user)
  POST /predict - Get predictions for a specific user
  GET /health - Health check endpoint
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import logging
from datetime import datetime
import pandas as pd

# Import functions from train_progress_model
from train_progress_model import (
    fetch_weekly_data,
    build_features,
    train_model,
    predict_next_weeks
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TurboBulk Progress Prediction API",
    description="API for training and predicting workout progress",
    version="1.0.0"
)

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://csxhklwgkaehwmrhuhyq.supabase.co")
# Use ANON key for better security (relies on RLS policies)
# Only use SERVICE_ROLE key if you need to bypass RLS for admin operations
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", os.environ.get("SUPABASE_KEY", ""))
MODEL_PATH = os.environ.get("MODEL_PATH", "models/progress_model.pkl")

# In-memory status tracking
training_status = {
    "is_training": False,
    "last_trained": None,
    "last_error": None
}


# Request/Response Models
class TrainRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional user ID to train model for specific user")
    predict_weeks: int = Field(1, description="Number of weeks ahead to predict", ge=1, le=12)
    max_lag: int = Field(2, description="Number of weekly lags to create", ge=1, le=8)


class PredictRequest(BaseModel):
    user_id: str = Field(..., description="User ID to generate predictions for")


class PredictionResponse(BaseModel):
    user_id: str
    exercise_id: str
    exercise_name: str
    week_start: str
    predicted_e1rm: float
    predict_weeks_ahead: int


class TrainResponse(BaseModel):
    message: str
    status: str
    timestamp: str


class StatusResponse(BaseModel):
    is_training: bool
    last_trained: Optional[str]
    last_error: Optional[str]
    model_exists: bool


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "TurboBulk Progress Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "train": "/train",
            "predict": "/predict",
            "status": "/status"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "supabase_configured": bool(SUPABASE_KEY),
        "model_exists": os.path.exists(MODEL_PATH)
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get training status and model information."""
    return StatusResponse(
        is_training=training_status["is_training"],
        last_trained=training_status["last_trained"],
        last_error=training_status["last_error"],
        model_exists=os.path.exists(MODEL_PATH)
    )


def train_model_background(user_id: Optional[str], predict_weeks: int, max_lag: int):
    """Background task to train the model."""
    try:
        training_status["is_training"] = True
        training_status["last_error"] = None
        
        logger.info("Starting model training...")
        
        # Fetch data
        df = fetch_weekly_data(SUPABASE_URL, SUPABASE_KEY)
        if df.empty:
            raise ValueError("No data fetched from Supabase")
        
        # Filter to specific user if requested
        if user_id:
            user_data = df[df['user_id'] == user_id]
            if user_data.empty:
                raise ValueError(f"No data found for user {user_id}")
            logger.info(f"Training for specific user: {user_id}")
            df = user_data
        
        # Build features
        df_feats = build_features(df, max_lag=max_lag, predict_weeks_ahead=predict_weeks)
        if df_feats.empty:
            raise ValueError("No training data after preprocessing")
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(MODEL_PATH) or '.', exist_ok=True)
        
        # Train model
        train_model(df_feats, MODEL_PATH, predict_weeks_ahead=predict_weeks)
        
        training_status["last_trained"] = datetime.utcnow().isoformat()
        logger.info("Model training completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Training failed: {error_msg}")
        training_status["last_error"] = error_msg
    finally:
        training_status["is_training"] = False


@app.post("/train", response_model=TrainResponse)
async def train(
    background_tasks: BackgroundTasks,
    predict_weeks: int = 1,
    max_lag: int = 2,
    user_id: Optional[str] = None
):
    """
    Train the progress prediction model.
    
    This endpoint triggers model training in the background. The model will be
    trained on all available workout data, or for a specific user if user_id is provided.

    Query params: predict_weeks (default 1), max_lag (default 2), user_id (optional)
    """
    if not SUPABASE_KEY:
        raise HTTPException(
            status_code=500,
            detail="Supabase key not configured. Set SUPABASE_ANON_KEY or SUPABASE_KEY environment variable."
        )
    
    if training_status["is_training"]:
        raise HTTPException(
            status_code=409,
            detail="Model training is already in progress"
        )

    # Start training in background
    background_tasks.add_task(
        train_model_background,
        user_id,
        predict_weeks,
        max_lag
    )
    
    return TrainResponse(
        message=f"Model training started (predict_weeks={predict_weeks}, max_lag={max_lag})",
        status="training",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=List[PredictionResponse])
async def predict(request: PredictRequest):
    """
    Generate progress predictions for a specific user.
    
    Returns predictions for all exercises the user has performed, based on
    their most recent workout data.
    """
    if not SUPABASE_KEY:
        raise HTTPException(
            status_code=500,
            detail="Supabase key not configured. Set SUPABASE_KEY environment variable."
        )
    
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=404,
            detail="Model not found. Please train the model first by calling /train endpoint."
        )
    
    try:
        # Fetch latest data for the user
        logger.info(f"Fetching data for user {request.user_id}")
        df = fetch_weekly_data(SUPABASE_URL, SUPABASE_KEY)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No workout data found")
        
        # Filter to specific user
        user_data = df[df['user_id'] == request.user_id]
        if user_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No workout data found for user {request.user_id}"
            )
        
        # Build features (use same params as model was trained with)
        import joblib
        model_info = joblib.load(MODEL_PATH)
        predict_weeks_ahead = model_info.get('predict_weeks_ahead', 1)
        
        df_feats = build_features(user_data, max_lag=2, predict_weeks_ahead=predict_weeks_ahead)
        
        if df_feats.empty:
            raise HTTPException(
                status_code=404,
                detail="Insufficient data for predictions. User needs at least 2 weeks of workout data."
            )
        
        # Get predictions for most recent week per exercise
        last_weeks = df_feats.sort_values('week_start').groupby(['user_id', 'exercise_id']).tail(1)
        predictions_df = predict_next_weeks(MODEL_PATH, last_weeks, user_id=request.user_id)
        
        # Convert to response format
        pred_col = f'pred_e1rm_{predict_weeks_ahead}wk'
        predictions = [
            PredictionResponse(
                user_id=row['user_id'],
                exercise_id=row['exercise_id'],
                exercise_name=row['exercise_name'],
                week_start=row['week_start'].strftime('%Y-%m-%d'),
                predicted_e1rm=round(row[pred_col], 2),
                predict_weeks_ahead=predict_weeks_ahead
            )
            for _, row in predictions_df.iterrows()
        ]
        
        logger.info(f"Generated {len(predictions)} predictions for user {request.user_id}")
        return predictions
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
