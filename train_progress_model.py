"""train_progress_model.py

Professional training script to predict per-user per-exercise progress (4 weeks ahead)
using weekly aggregated features from the database.

Usage:
  python train_progress_model.py --supabase-url "https://..." --supabase-key "..." --model-out path/to/model.pkl

Dependencies:
  pip install pandas scikit-learn joblib requests
"""

import os
import argparse
import logging
from typing import Optional

import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Default Supabase project URL
DEFAULT_SUPABASE_URL = "https://csxhklwgkaehwmrhuhyq.supabase.co"
DEFAULT_SUPABASE_KEY = ""  # Set via --supabase-key or SUPABASE_KEY env var

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# SQL to fetch weekly aggregated stats per user & exercise
WEEKLY_SQL = """
SELECT
  w.user_id,
  ws.exercise_id,
  COALESCE(e.name, '') AS exercise_name,
  date_trunc('week', ws.completed_at)::date AS week_start,
  MAX(ws.weight) FILTER (WHERE ws.weight IS NOT NULL) AS max_weight,
  SUM(COALESCE(ws.weight,0) * COALESCE(ws.reps,0)) AS total_volume,
  AVG(ws.rpe) AS avg_rpe,
  AVG(ws.reps) AS avg_reps,
  COUNT(DISTINCT ws.workout_id) AS sessions,
  COUNT(*) AS sets
FROM public.workout_sets ws
JOIN public.workouts w ON ws.workout_id = w.id
LEFT JOIN public.exercises e ON ws.exercise_id = e.id
GROUP BY 1,2,3,4
ORDER BY 1,2,4;
"""


def calculate_one_rep_max(weight: float, reps: int, rpe: float) -> float:
    """Calculate estimated 1RM using weight, reps, and RPE.
    
    Uses RPE to determine reps in reserve (RIR = 10 - RPE), then applies
    the Epley formula adjusted for total reps to failure.
    
    Args:
        weight: Weight lifted
        reps: Reps performed
        rpe: Rate of Perceived Exertion (0-10 scale)
    
    Returns:
        Estimated one-rep max
    """
    if pd.isna(weight) or pd.isna(reps) or weight == 0 or reps == 0:
        return 0.0
    
    # Calculate reps in reserve from RPE
    rir = 10 - rpe if not pd.isna(rpe) else 0
    
    # Total reps to failure = performed reps + reps in reserve
    total_reps_to_failure = reps + rir
    
    # Epley formula: 1RM = weight × (1 + reps_to_failure / 30)
    one_rm = weight * (1 + total_reps_to_failure / 30.0)
    
    return one_rm


def fetch_weekly_data(supabase_url: str, supabase_key: str) -> pd.DataFrame:
    """Fetches workout data from Supabase REST API and aggregates weekly stats per user & exercise."""
    logger.info("Fetching workout data from Supabase REST API...")
    
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json"
    }
    
    # Fetch workout_sets with related workouts and exercises
    url = f"{supabase_url}/rest/v1/workout_sets?select=*,workouts(user_id,started_at),exercises(name)"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        logger.error("Failed to fetch data: %s - %s", response.status_code, response.text)
        return pd.DataFrame()
    
    data = response.json()
    if not data:
        logger.warning("No workout data found")
        return pd.DataFrame()
    
    # Flatten the nested structure
    rows = []
    for item in data:
        workout = item.get('workouts', {}) or {}
        exercise = item.get('exercises', {}) or {}
        rows.append({
            'user_id': workout.get('user_id'),
            'exercise_id': item.get('exercise_id'),
            'exercise_name': exercise.get('name', ''),
            'completed_at': item.get('completed_at'),
            'weight': item.get('weight'),
            'reps': item.get('reps'),
            'rpe': item.get('rpe')
        })
    
    df = pd.DataFrame(rows)
    df['completed_at'] = pd.to_datetime(df['completed_at'], format='ISO8601')
    
    # Calculate 1RM for each set
    df['estimated_1rm'] = df.apply(
        lambda row: calculate_one_rep_max(row['weight'], row['reps'], row['rpe']),
        axis=1
    )
    
    # Aggregate by week
    df['week_start'] = df['completed_at'].dt.to_period('W').dt.start_time
    
    weekly = df.groupby(['user_id', 'exercise_id', 'exercise_name', 'week_start']).agg({
        'estimated_1rm': 'max',  # Best estimated 1RM for the week
        'weight': 'max',
        'reps': 'mean',
        'rpe': 'mean',
        'completed_at': 'count'
    }).reset_index()
    
    weekly.columns = ['user_id', 'exercise_id', 'exercise_name', 'week_start', 
                      'max_e1rm', 'max_weight', 'avg_reps', 'avg_rpe', 'sets']
    
    # Calculate total volume
    df['volume'] = df['weight'].fillna(0) * df['reps'].fillna(0)
    volume_df = df.groupby(['user_id', 'exercise_id', 'week_start'])['volume'].sum().reset_index()
    volume_df.columns = ['user_id', 'exercise_id', 'week_start', 'total_volume']
    
    # Merge volume back
    weekly = weekly.merge(volume_df, on=['user_id', 'exercise_id', 'week_start'], how='left')
    
    # Count sessions per week
    session_df = df.groupby(['user_id', 'exercise_id', 'week_start'])['completed_at'].apply(
        lambda x: x.dt.date.nunique()
    ).reset_index()
    session_df.columns = ['user_id', 'exercise_id', 'week_start', 'sessions']
    
    weekly = weekly.merge(session_df, on=['user_id', 'exercise_id', 'week_start'], how='left')
    
    logger.info("Fetched and aggregated %d weekly records", len(weekly))
    return weekly


def build_features(df: pd.DataFrame, max_lag: int = 2, predict_weeks_ahead: int = 1) -> pd.DataFrame:
    """Build lag features and targets (1 week ahead target by default for sparse data).

    For each (user_id, exercise_id) week row we create features:
      - lags of max_e1rm, total_volume, avg_rpe, avg_reps, sessions, sets
      - rolling means and growth rates
    Target: target_e1rm_Xwk = estimated 1RM at week + predict_weeks_ahead
    """

    df = df.copy()
    # Ensure types
    df['week_start'] = pd.to_datetime(df['week_start'])
    df = df.sort_values(['user_id', 'exercise_id', 'week_start'])

    # fill NaNs sensibly
    df['max_e1rm'] = df['max_e1rm'].astype(float)
    df['max_weight'] = df['max_weight'].astype(float)
    for col in ['total_volume', 'avg_rpe', 'avg_reps', 'sessions', 'sets']:
        df[col] = df[col].astype(float)

    # group and create lags
    def make_lags(group):
        group = group.sort_values('week_start')

        # Rest-awareness: days between sessions (using week_start gaps as weekly aggregation)
        group['rest_days'] = group['week_start'].diff().dt.days.fillna(7).clip(lower=0)

        for col in ['max_e1rm', 'max_weight', 'total_volume', 'avg_rpe', 'avg_reps', 'sessions', 'sets', 'rest_days']:
            # Create lag features directly without forcing weekly frequency
            group[f'{col}_lag1'] = group[col].shift(1)
            group[f'{col}_lag2'] = group[col].shift(2)
            # rolling mean of last 2 weeks
            group[f'{col}_roll_mean2'] = group[col].rolling(window=2, min_periods=1).mean()
            group[f'{col}_roll_std2'] = group[col].rolling(window=2, min_periods=1).std().fillna(0)
            # velocity: week-over-week change
            group[f'{col}_velocity'] = group[col].diff().fillna(0)
            # growth rate
            group[f'{col}_growth_rate'] = group[col].pct_change(fill_method=None).fillna(0).replace([np.inf, -np.inf], 0)
        # target: estimated 1RM N weeks ahead
        group[f'target_e1rm_{predict_weeks_ahead}wk'] = group['max_e1rm'].shift(-predict_weeks_ahead)
        return group

    df_expanded = df.groupby(['user_id', 'exercise_id'], group_keys=False).apply(make_lags)

    # Drop rows without target - but if all rows are missing target, use the most recent week's data
    # with their current e1rm as the target (simulating "maintain current level")
    target_col = f'target_e1rm_{predict_weeks_ahead}wk'
    logger.info("Target column: %s, available columns: %s", target_col, df_expanded.columns.tolist())
    logger.info("NaN counts in target: %d / %d", df_expanded[target_col].isna().sum(), len(df_expanded))
    
    rows_with_target = df_expanded.dropna(subset=[target_col])
    if rows_with_target.empty:
        logger.warning("No forward-looking targets found; using recent data with maintenance assumption")
        # Use the most recent entries per exercise, set target to current max_e1rm (maintain level)
        df_expanded[target_col] = df_expanded['max_e1rm']
        df_expanded = df_expanded.dropna(subset=['max_e1rm'])
    else:
        df_expanded = rows_with_target

    # Fill remaining NaNs in lag features with 0 (or better: with small values)
    lag_cols = [c for c in df_expanded.columns if '_lag' in c or '_roll_mean' in c or '_roll_std' in c or '_velocity' in c or '_growth_rate' in c]
    df_expanded[lag_cols] = df_expanded[lag_cols].fillna(0).replace([np.inf, -np.inf], 0)

    # Add comprehensive time-based features for temporal awareness
    df_expanded['week_of_year'] = df_expanded['week_start'].dt.isocalendar().week
    df_expanded['year'] = df_expanded['week_start'].dt.year
    df_expanded['month'] = df_expanded['week_start'].dt.month
    df_expanded['quarter'] = df_expanded['week_start'].dt.quarter
    df_expanded['day_of_week'] = df_expanded['week_start'].dt.dayofweek  # 0=Monday, 6=Sunday
    
    # Seasonality (0=winter, 1=spring, 2=summer, 3=fall)
    df_expanded['season'] = (df_expanded['month'] % 12) // 3
    
    # Days since start of year (for continuity)
    df_expanded['day_of_year'] = df_expanded['week_start'].dt.dayofyear

    logger.info("Built features with %d rows and %d columns", df_expanded.shape[0], df_expanded.shape[1])
    return df_expanded


def train_model(df: pd.DataFrame, model_out: str, predict_weeks_ahead: int = 2):
    """Train a regressor to predict N-week ahead estimated 1RM and save the model pipeline."""
    target_col = f'target_e1rm_{predict_weeks_ahead}wk'
    features = [c for c in df.columns if (
        ('_lag' in c) or ('_roll_mean' in c) or ('_roll_std' in c) or 
        ('_velocity' in c) or ('_growth_rate' in c) or 
        c in ('avg_rpe','avg_reps','sessions','sets','total_volume','rest_days',
              'week_of_year','year','month','quarter','day_of_week','season','day_of_year')
    )]
    # Ensure features exist and handle inf values
    X = df[features].astype(float).replace([np.inf, -np.inf], 0)
    y = df[target_col].astype(float)
    groups = df['user_id']

    # Enhanced pipeline with better hyperparameters
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.8,
            random_state=42
        ))
    ])

    # Group-aware CV to avoid data leakage between users
    n_unique_groups = groups.nunique()
    n_splits = min(5, n_unique_groups)  # Use fewer splits if we have fewer groups
    
    if n_splits < 2:
        logger.warning("Only %d unique user(s) in dataset - skipping cross-validation", n_unique_groups)
        # Just train without CV
        pipeline.fit(X, y)
        logger.info("Trained on full dataset (%d samples)", len(X))
    else:
        gkf = GroupKFold(n_splits=n_splits)
        rmses = []
        r2s = []

        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = pipeline.predict(X.iloc[test_idx])
            rmse = mean_squared_error(y.iloc[test_idx], pred, squared=False)
            r2 = r2_score(y.iloc[test_idx], pred)
            rmses.append(rmse)
            r2s.append(r2)

        logger.info("Cross-val RMSE: %.3f ± %.3f", np.mean(rmses), np.std(rmses))
        logger.info("Cross-val R2: %.3f ± %.3f", np.mean(r2s), np.std(r2s))

        # Retrain on full data
        pipeline.fit(X, y)
    joblib.dump({'pipeline': pipeline, 'features': features, 'predict_weeks_ahead': predict_weeks_ahead}, model_out)
    logger.info("Trained model saved to %s", model_out)
    return pipeline, features


def predict_next_weeks(model_file: str, df_latest: pd.DataFrame, user_id: Optional[str] = None, num_future_weeks: int = 4) -> pd.DataFrame:
    """Iteratively predict current and future estimated 1RM per exercise.

    The model is autoregressive: each future step feeds the previous prediction
    into the lagged features so the line moves instead of staying flat.
    """
    saved = joblib.load(model_file)
    pipeline = saved['pipeline']
    features = saved['features']

    # Columns we evolve step to step
    signal_cols = ['max_e1rm', 'max_weight', 'total_volume', 'avg_rpe', 'avg_reps', 'sessions', 'sets', 'rest_days']

    all_predictions = []

    for _, row in df_latest.iterrows():
        user_ex_predictions = {
            'user_id': row['user_id'],
            'exercise_id': row['exercise_id'],
            'exercise_name': row['exercise_name'],
            'predictions': []
        }

        # Work on a mutable copy
        working = row.copy()

        # Helper to push a prediction row to the list
        def append_pred(week_start_ts: pd.Timestamp, value: float, week_offset: int):
            user_ex_predictions['predictions'].append({
                'week_start': week_start_ts.strftime('%Y-%m-%d'),
                'predicted_e1rm': float(value),
                'week_offset': week_offset
            })

        # Week 0: current expected 1RM (use observed max_e1rm if available)
        week0_value = float(working.get('max_e1rm', 0.0))
        append_pred(working['week_start'], week0_value, 0)

        # Iteratively predict future weeks
        for week_offset in range(1, num_future_weeks + 1):
            # Predict next step using current feature snapshot
            row_features = working[features].astype(float).replace([np.inf, -np.inf], 0).values.reshape(1, -1)
            predicted_1rm = float(pipeline.predict(row_features)[0])

            # Advance week_start
            next_week_start = working['week_start'] + pd.Timedelta(weeks=1)

            # Update base signals for next iteration (autoregressive feed)
            # Shift lags: lag2 <= lag1, lag1 <= current
            for col in signal_cols:
                curr = working.get(col, 0.0)
                lag1 = working.get(f'{col}_lag1', curr)
                working[f'{col}_lag2'] = lag1
                working[f'{col}_lag1'] = curr

                # rolling stats over last two values (curr, lag1)
                working[f'{col}_roll_mean2'] = np.mean([curr, lag1])
                working[f'{col}_roll_std2'] = float(np.std([curr, lag1]))
                working[f'{col}_velocity'] = curr - lag1
                working[f'{col}_growth_rate'] = 0.0 if lag1 == 0 else (curr - lag1) / lag1

            # Set new current values for next step
            working['max_e1rm'] = predicted_1rm
            # keep max_weight/volume stable if unknown; they still get lagged/rolled
            working['rest_days'] = 7  # assume weekly cadence for forecasts

            # Update temporal features
            working['week_start'] = next_week_start
            working['week_of_year'] = next_week_start.isocalendar().week
            working['year'] = next_week_start.year
            working['month'] = next_week_start.month
            working['quarter'] = next_week_start.quarter
            working['day_of_week'] = next_week_start.dayofweek
            working['season'] = (working['month'] % 12) // 3
            working['day_of_year'] = next_week_start.dayofyear

            append_pred(next_week_start, predicted_1rm, week_offset)

        all_predictions.append(user_ex_predictions)

    # Flatten for return
    result_rows = []
    for pred_group in all_predictions:
        for pred in pred_group['predictions']:
            result_rows.append({
                'user_id': pred_group['user_id'],
                'exercise_id': pred_group['exercise_id'],
                'exercise_name': pred_group['exercise_name'],
                'week_start': pred['week_start'],
                'predicted_e1rm': pred['predicted_e1rm'],
                'week_offset': pred['week_offset']
            })

    out = pd.DataFrame(result_rows)
    if user_id:
        out = out[out['user_id'] == user_id]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a model to predict N-week ahead estimated 1RM per user-exercise.")
    parser.add_argument('--supabase-url', default=os.environ.get('SUPABASE_URL', DEFAULT_SUPABASE_URL), 
                        help='Supabase project URL')
    parser.add_argument('--supabase-key', default=os.environ.get('SUPABASE_KEY', DEFAULT_SUPABASE_KEY), 
                        help='Supabase anon/service key')
    parser.add_argument('--model-out', default='models/progress_model.pkl', help='Path to save trained model')
    parser.add_argument('--max-lag', type=int, default=2, help='Number of weekly lags to create')
    parser.add_argument('--predict-weeks', type=int, default=1, help='Number of weeks ahead to predict (default 1 for sparse data)')
    parser.add_argument('--only-user', default=None, help='(optional) user_id to filter and train on')
    args = parser.parse_args(argv)

    if not args.supabase_key:
        logger.error("Supabase key is required. Set via --supabase-key or SUPABASE_KEY env var")
        return

    # Ensure output directory
    os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)

    df = fetch_weekly_data(args.supabase_url, args.supabase_key)
    if df.empty:
        logger.error("No data fetched - aborting")
        return

    # Filter to user early if specified
    if args.only_user:
        user_data = df[df['user_id'] == args.only_user]
        if user_data.empty:
            logger.error("No data found for user %s. Available users: %s", args.only_user, df['user_id'].unique().tolist())
            return
        logger.info("Found %d weekly records for user %s", len(user_data), args.only_user)
        df = user_data

    df_feats = build_features(df, max_lag=args.max_lag, predict_weeks_ahead=args.predict_weeks)

    if df_feats.empty:
        logger.error("No training rows after preprocessing (need at least 2 weeks of data per exercise)")
        return

    train_model(df_feats, args.model_out, predict_weeks_ahead=args.predict_weeks)

    # Example: show predictions for the most recent week per user-exercise
    last_weeks = df_feats.sort_values('week_start').groupby(['user_id','exercise_id']).tail(1)
    preds = predict_next_weeks(args.model_out, last_weeks)
    logger.info("Sample predictions (next %d-week estimated 1RM):\n%s", args.predict_weeks, preds.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
