# TurboBulk Backend Deployment Guide

This guide explains how to deploy the TurboBulk progress prediction backend to free hosting platforms.

## Prerequisites

- Git repository with your code
- Supabase service role key

## Files Included

- `api.py` - FastAPI backend with endpoints
- `train_progress_model.py` - ML model training script
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration
- `.env.example` - Environment variables template

## Deployment Options

### Option 1: Deploy to Render (Recommended)

**Steps:**

1. **Push your code to GitHub**
   ```bash
   git add api.py train_progress_model.py requirements.txt Dockerfile .env.example
   git commit -m "Add backend API"
   git push origin main
   ```

2. **Create a Render account**
   - Go to https://render.com
   - Sign up with GitHub

3. **Create a new Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository containing your backend

4. **Configure the service**
   - **Name**: turbobulk-api
   - **Region**: Select closest to you
   - **Branch**: main
   - **Runtime**: Docker
   - **Instance Type**: Free

5. **Add environment variables**
   - Click "Environment" tab
   - Add these variables:
     - `SUPABASE_URL`: `https://csxhklwgkaehwmrhuhyq.supabase.co`
     - `SUPABASE_KEY`: Your Supabase service role key
     - `PORT`: `8000`

6. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)
   - Your API will be available at: `https://turbobulk-api.onrender.com`

**Important Notes:**
- Free tier sleeps after 15 minutes of inactivity
- First request after sleep takes 30-60 seconds
- 750 hours/month free (enough for most projects)

---

### Option 2: Deploy to Railway

**Steps:**

1. **Push your code to GitHub** (same as above)

2. **Create a Railway account**
   - Go to https://railway.app
   - Sign up with GitHub

3. **Create a new project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

4. **Configure environment variables**
   - Click on your service
   - Go to "Variables" tab
   - Add:
     - `SUPABASE_URL`: `https://csxhklwgkaehwmrhuhyq.supabase.co`
     - `SUPABASE_KEY`: Your Supabase service role key
     - `PORT`: `8000`

5. **Deploy**
   - Railway will automatically detect the Dockerfile and deploy
   - Your API will be available at the generated domain

**Important Notes:**
- $5/month free credit
- Automatically sleeps when not in use
- Good performance

---

## Testing Your Deployment

Once deployed, test your API:

### 1. Health Check
```bash
curl https://your-api-url.com/health
```

### 2. Train Model
```bash
curl -X POST https://your-api-url.com/train \
  -H "Content-Type: application/json" \
  -d '{"predict_weeks": 1}'
```

### 3. Get Predictions
```bash
curl -X POST https://your-api-url.com/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": "your-user-id-here"}'
```

### 4. Check Status
```bash
curl https://your-api-url.com/status
```

---

## Local Development

To run the backend locally:

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables**
   ```bash
   # Windows PowerShell
   $env:SUPABASE_URL="https://csxhklwgkaehwmrhuhyq.supabase.co"
   $env:SUPABASE_KEY="your-key-here"
   
   # Or create a .env file and use python-dotenv
   ```

3. **Run the server**
   ```bash
   python api.py
   ```

4. **Access the API**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

---

## Integrating with Frontend

### Example: Fetch Predictions

```javascript
// In your React/TypeScript frontend
async function getPredictions(userId) {
  try {
    const response = await fetch('https://your-api-url.com/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ user_id: userId }),
    });
    
    if (!response.ok) {
      throw new Error('Failed to fetch predictions');
    }
    
    const predictions = await response.json();
    return predictions;
  } catch (error) {
    console.error('Error fetching predictions:', error);
    throw error;
  }
}

// Usage
const predictions = await getPredictions('user-id-123');
console.log(predictions);
```

### Example: Train Model

```javascript
async function trainModel() {
  try {
    const response = await fetch('https://your-api-url.com/train', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        predict_weeks: 1 
      }),
    });
    
    if (!response.ok) {
      throw new Error('Failed to start training');
    }
    
    const result = await response.json();
    console.log('Training started:', result);
    
    // Poll status endpoint to check when training is complete
    const checkStatus = setInterval(async () => {
      const statusResponse = await fetch('https://your-api-url.com/status');
      const status = await statusResponse.json();
      
      if (!status.is_training) {
        clearInterval(checkStatus);
        console.log('Training complete!');
      }
    }, 5000); // Check every 5 seconds
    
  } catch (error) {
    console.error('Error training model:', error);
    throw error;
  }
}
```

---

## API Endpoints Reference

### `GET /health`
Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-21T10:30:00",
  "supabase_configured": true,
  "model_exists": true
}
```

### `POST /train`
Train the prediction model.

**Request:**
```json
{
  "user_id": "optional-user-id",
  "predict_weeks": 1,
  "max_lag": 2
}
```

**Response:**
```json
{
  "message": "Model training started",
  "status": "training",
  "timestamp": "2026-01-21T10:30:00"
}
```

### `POST /predict`
Get predictions for a user.

**Request:**
```json
{
  "user_id": "user-id-here"
}
```

**Response:**
```json
[
  {
    "user_id": "user-id-here",
    "exercise_id": "exercise-123",
    "exercise_name": "Bench Press",
    "week_start": "2026-01-20",
    "predicted_e1rm": 225.5,
    "predict_weeks_ahead": 1
  }
]
```

### `GET /status`
Check training status and model info.

**Response:**
```json
{
  "is_training": false,
  "last_trained": "2026-01-21T10:00:00",
  "last_error": null,
  "model_exists": true
}
```

---

## Troubleshooting

### API returns 500 error
- Check that `SUPABASE_KEY` is set correctly
- Verify Supabase credentials are valid

### Model not found error
- Call `/train` endpoint first to train the model
- Check that training completed successfully via `/status`

### Predictions return 404
- Ensure the user has at least 2 weeks of workout data
- Check that the user_id is correct

### Training takes too long
- Normal for large datasets (can take 5-10 minutes)
- Check `/status` endpoint to monitor progress
- Training runs in background, API remains responsive

---

## Next Steps

1. Deploy to Render or Railway
2. Train the model with your production data
3. Integrate prediction endpoints into your frontend
4. Set up automated retraining (e.g., weekly cron job)

For questions or issues, refer to the platform documentation:
- [Render Docs](https://render.com/docs)
- [Railway Docs](https://docs.railway.app)
