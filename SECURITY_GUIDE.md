# Security Guide: Using Anon Key Instead of Service Role Key

## The Problem
**Service role keys bypass all Row-Level Security (RLS)** and give full admin access to your database. If leaked, an attacker can read/modify/delete any data.

## The Solution: Use Anon Key + RLS Policies

### What's Different?
- **Service Role Key**: Full admin access, bypasses all security
- **Anon Key**: Respects RLS policies, safe to expose in limited contexts

### How It Works
1. **Backend uses anon key** to make Supabase requests
2. **Supabase RLS policies** control what data can be accessed
3. **Policies check user authentication** to ensure users only access their own data

---

## Setup Steps

### 1. Update Your Environment Variables

**On Render/Railway:**
- Remove or rename `SUPABASE_KEY` → `SUPABASE_SERVICE_KEY` (keep for emergency admin tasks only)
- Add `SUPABASE_ANON_KEY` with your anon key from Supabase dashboard

**Get your anon key:**
- Go to Supabase Dashboard → Settings → API
- Copy the "anon" / "public" key (starts with `eyJhbGc...`)

**Environment variables to set:**
```bash
SUPABASE_URL=https://csxhklwgkaehwmrhuhyq.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here
```

---

### 2. Set Up RLS Policies in Supabase

Your backend needs to read workout data for all users to train the model, so you have two options:

#### Option A: Backend Service Account (Recommended)

Create a special "service" user for the backend:

1. **Create a service user:**
   ```sql
   -- Create a service role in your auth.users table
   INSERT INTO auth.users (id, email, role)
   VALUES ('00000000-0000-0000-0000-000000000001', 'backend-service@yourdomain.com', 'service');
   ```

2. **Create RLS policies that allow service user to read all data:**
   ```sql
   -- Allow service user to read all workout_sets
   CREATE POLICY "Service can read all workout_sets"
   ON workout_sets
   FOR SELECT
   TO authenticated
   USING (auth.uid() = '00000000-0000-0000-0000-000000000001');

   -- Allow service user to read all workouts
   CREATE POLICY "Service can read all workouts"
   ON workouts
   FOR SELECT
   TO authenticated
   USING (auth.uid() = '00000000-0000-0000-0000-000000000001');

   -- Allow service user to read all exercises
   CREATE POLICY "Service can read all exercises"
   ON exercises
   FOR SELECT
   TO authenticated
   USING (true);  -- Exercises are public reference data
   ```

3. **Update your backend to authenticate as the service user:**
   - Generate a JWT token for the service user
   - Pass this token in Authorization header with requests

#### Option B: Public Read Access for Training (Simpler but Less Secure)

If workout data isn't sensitive, allow read access:

```sql
-- Allow authenticated users to read all workout data for ML training
CREATE POLICY "Allow read access for ML training"
ON workout_sets
FOR SELECT
TO authenticated
USING (true);

CREATE POLICY "Allow read access for ML training"
ON workouts
FOR SELECT
TO authenticated
USING (true);
```

⚠️ **Note:** This allows any authenticated user to see all workouts. Only use if workout data isn't sensitive.

#### Option C: User-Specific Predictions (Most Secure)

If you only need predictions for the logged-in user:

```sql
-- Users can only read their own workout data
CREATE POLICY "Users can read own workout_sets"
ON workout_sets
FOR SELECT
TO authenticated
USING (
  workout_id IN (
    SELECT id FROM workouts WHERE user_id = auth.uid()
  )
);

CREATE POLICY "Users can read own workouts"
ON workouts
FOR SELECT
TO authenticated
USING (user_id = auth.uid());
```

**Backend changes needed:**
- Modify `/train` to accept a user token from frontend
- Pass the token to Supabase requests
- Train model only for that specific user

---

### 3. Update Backend to Use User Tokens (Option C)

If you want user-specific access:

```python
from fastapi import Header

@app.post("/predict")
async def predict(
    request: PredictRequest,
    authorization: str = Header(..., description="Bearer token from Supabase auth")
):
    """Get predictions using user's auth token for RLS."""
    
    # Extract token from "Bearer <token>"
    token = authorization.replace("Bearer ", "")
    
    # Pass token to Supabase requests (modify fetch_weekly_data to accept token)
    headers = {
        "apikey": SUPABASE_KEY,  # Anon key
        "Authorization": f"Bearer {token}",  # User token
        "Content-Type": "application/json"
    }
    
    # Now RLS policies will apply based on the user's token
    ...
```

**Frontend sends token:**
```javascript
const { data: { session } } = await supabase.auth.getSession();
const token = session.access_token;

fetch('https://your-api/predict', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ user_id: session.user.id })
});
```

---

## Quick Comparison

| Approach | Security | Complexity | Use Case |
|----------|----------|------------|----------|
| **Option A: Service Account** | ⭐⭐⭐⭐ | Medium | Backend trains on all data, users get predictions |
| **Option B: Public Read** | ⭐⭐ | Low | Non-sensitive data, simple setup |
| **Option C: User Tokens** | ⭐⭐⭐⭐⭐ | High | Maximum security, user-specific models |

---

## Recommended Approach for TurboBulk

**Use Option A (Service Account):**
1. Backend authenticates as service user
2. Can read all workout data to train a global model
3. Users call `/predict` with their user_id
4. Predictions are returned but data stays secure

**Why?**
- Balances security with functionality
- One global model trained on all users (better accuracy)
- Service key never exposed, only anon key + service user token

---

## Testing Your Setup

1. **Remove service key** from environment variables
2. **Add anon key** and restart your backend
3. **Test endpoints:**
   ```bash
   # Should work (uses anon key + RLS)
   curl -X POST https://your-api/train
   
   # Should fail if RLS policies aren't set
   curl -X POST https://your-api/predict -d '{"user_id":"test"}'
   ```

4. **Check Supabase logs** (Dashboard → Logs) to see if requests are blocked by RLS

---

## What If I Need Service Key?

Keep it for **admin-only operations**:
- Database migrations
- Bulk data imports
- Emergency recovery

**Store it separately:**
```bash
SUPABASE_SERVICE_KEY=your_service_key  # Admin only, never used by API
SUPABASE_ANON_KEY=your_anon_key        # Used by API
```

**Create admin-only endpoints:**
```python
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "change-me")

@app.post("/admin/force-retrain")
async def admin_retrain(admin_secret: str = Header(...)):
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(403, "Forbidden")
    
    # Use service key for this operation only
    service_key = os.environ.get("SUPABASE_SERVICE_KEY")
    # ... admin operation
```

---

## Summary

✅ **Do:**
- Use anon key for API operations
- Set up RLS policies to control access
- Pass user tokens from frontend to backend
- Keep service key for emergencies only

❌ **Don't:**
- Expose service key in API or frontend
- Skip RLS policies
- Allow public read access to sensitive data

**Next steps:**
1. Get your anon key from Supabase dashboard
2. Update environment variables on Render
3. Set up RLS policies (Option A recommended)
4. Test your endpoints
5. Remove service key from env vars once working
