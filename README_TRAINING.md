# Training Model Setup Guide

## Issue: Database Connection Problem

Your Windows machine cannot resolve the Supabase PostgreSQL hostname (`db.csxhklwgkaehwmrhuhyq.supabase.co`) due to IPv6-only DNS resolution. This is a common issue on some Windows networks.

## Solutions

### Option 1: Use Supabase REST API (Recommended)

I've updated `train_progress_model.py` to use the Supabase REST API instead of direct PostgreSQL connection.

**You need the Supabase anon key:**

1. Go to your Supabase dashboard: https://app.supabase.com/project/csxhklwgkaehwmrhuhyq/settings/api
2. Copy the "anon" or "public" API key
3. Run the script:

```bash
python train_progress_model.py --supabase-url "https://csxhklwgkaehwmrhuhyq.supabase.co" --supabase-key "YOUR_ANON_KEY_HERE"
```

### Option 2: Use Connection Pooler (IPv4)

Supabase provides a connection pooler that works with IPv4:

```bash
python train_progress_model.py --db-url "postgresql://postgres.csxhklwgkaehwmrhuhyq:P1wXiD4IrG92gj9C@aws-0-us-east-1.pooler.supabase.com:5432/postgres"
```

Note: Replace `aws-0-us-east-1` with your actual pooler region.

### Option 3: Enable IPv6 on Your Network

If you have control over your network:
1. Enable IPv6 in Windows network settings
2. Or use a VPN that supports IPv6
3. Then the original PostgreSQL connection will work

### Option 4: Test with Sample Data

I can create a version that generates synthetic training data for testing the model without needing database access.

## Next Steps

Please provide your Supabase anon key or let me know which option you'd like to pursue.
