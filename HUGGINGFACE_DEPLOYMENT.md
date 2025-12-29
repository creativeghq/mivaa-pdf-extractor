# Hugging Face Remote Embeddings - Deployment Guide

## Overview

This guide explains how to enable remote SigLIP v2 visual embeddings via Hugging Face Inference API.

**Benefits:**
- ✅ GPU-accelerated embeddings without local GPU hardware
- ✅ ~15x faster than CPU (12-17s → <1s per image)
- ✅ Batch processing support (up to 10 images per request)
- ✅ Automatic fallback to local processing if API fails
- ✅ Cost-effective (~$0.0001 per image)

## Configuration

### Mode Selection

The system supports two modes via `VISUAL_EMBEDDING_MODE` environment variable:

| Mode | Description | Use Case |
|------|-------------|----------|
| `local` | Run SigLIP v2 locally on server | Default, no API costs, slower on CPU |
| `remote` | Use Hugging Face Inference API | GPU-accelerated, faster, small API cost |

### Required Environment Variables

Add these to your systemd service file (`/etc/systemd/system/mivaa-pdf-extractor.service`):

```bash
# ⚠️ IMPORTANT: Upgrade to SigLIP v2 (both local and remote)
Environment="VISUAL_EMBEDDING_PRIMARY_MODEL=google/siglip2-so400m-patch14-384"

# Visual Embedding Mode
Environment="VISUAL_EMBEDDING_MODE=remote"

# Hugging Face API Configuration
Environment="HUGGINGFACE_API_KEY=hf_your_api_key_here"
Environment="HUGGINGFACE_API_URL=https://api-inference.huggingface.co"
Environment="HUGGINGFACE_SIGLIP_MODEL=google/siglip2-so400m-patch14-384"
Environment="HUGGINGFACE_BATCH_SIZE=10"
Environment="HUGGINGFACE_TIMEOUT=60"
Environment="HUGGINGFACE_MAX_RETRIES=3"
```

**Why SigLIP v2?**
- Better multilingual support
- Improved accuracy
- Same API as v1 (drop-in replacement)
- Used by both local and remote modes

### Optional: Keep Local as Fallback

The system automatically falls back to local processing if remote API fails, so you can safely enable remote mode.

## Step-by-Step Deployment

### Step 1: Get Hugging Face API Key

1. Go to https://huggingface.co/settings/tokens
2. Click **New token**
3. Name: `mivaa-siglip-embeddings`
4. Type: **Read** (not Write)
5. Copy the token (starts with `hf_...`)

### Step 2: Update Systemd Service

```bash
# SSH into server
ssh root@165.227.31.109

# Edit systemd service file
sudo nano /etc/systemd/system/mivaa-pdf-extractor.service
```

Add these lines in the `[Service]` section (after existing Environment= lines):

```ini
[Service]
# ... existing environment variables ...

# ⚠️ UPGRADE: SigLIP v1 → v2 (for both local and remote)
Environment="VISUAL_EMBEDDING_PRIMARY_MODEL=google/siglip2-so400m-patch14-384"

# Visual Embedding Mode (local or remote)
Environment="VISUAL_EMBEDDING_MODE=remote"

# Hugging Face API
Environment="HUGGINGFACE_API_KEY=hf_YOUR_ACTUAL_KEY_HERE"
Environment="HUGGINGFACE_SIGLIP_MODEL=google/siglip2-so400m-patch14-384"
Environment="HUGGINGFACE_BATCH_SIZE=10"
Environment="HUGGINGFACE_TIMEOUT=60"
```

**Note:** The first time you restart after adding `VISUAL_EMBEDDING_PRIMARY_MODEL`, the service will download SigLIP v2 model (~1.5GB). This is a one-time download.

### Step 3: Reload and Restart Service

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Restart service
sudo systemctl restart mivaa-pdf-extractor

# Check status
sudo systemctl status mivaa-pdf-extractor

# Monitor logs
journalctl -u mivaa-pdf-extractor -f
```

### Step 4: Verify Configuration

Check logs for confirmation:

```bash
journalctl -u mivaa-pdf-extractor -n 50 | grep -E "(Visual embedding mode|HuggingFace)"
```

Expected output:
```
🤗 Hugging Face Visual Embeddings initialized: google/siglip2-so400m-patch14-384
   Batch size: 10, Timeout: 60s
🤗 Visual embedding mode: REMOTE (Hugging Face API)
```

## Performance Monitoring

### Check Latency

```bash
# Monitor embedding generation time
journalctl -u mivaa-pdf-extractor -f | grep "latency="
```

Expected:
- **Remote (GPU)**: `latency=500-1500ms` per image
- **Local (CPU)**: `latency=12000-17000ms` per image

### Check API Calls

```bash
# Monitor Hugging Face API calls
journalctl -u mivaa-pdf-extractor -f | grep "🤗"
```

### Check Costs

API calls are logged to `ai_calls` table with cost tracking:

```sql
SELECT 
  DATE(created_at) as date,
  COUNT(*) as api_calls,
  SUM(cost) as total_cost
FROM ai_calls
WHERE model LIKE 'huggingface/%'
GROUP BY DATE(created_at)
ORDER BY date DESC;
```

## Pricing

**Hugging Face Inference API:**
- Free tier: 30,000 requests/month
- Pro: $9/month for 100,000 requests
- Estimated cost: ~$0.0001 per image

**Example:**
- 194 images per job = ~$0.02 per job
- 100 jobs/month = ~$2/month
- Well within free tier limits

## Troubleshooting

### Issue: "Visual embedding mode: LOCAL" in logs

**Cause**: `VISUAL_EMBEDDING_MODE` not set to `remote` or `HUGGINGFACE_API_KEY` missing

**Solution**:
```bash
# Check environment variables
sudo systemctl show mivaa-pdf-extractor | grep HUGGINGFACE
sudo systemctl show mivaa-pdf-extractor | grep VISUAL_EMBEDDING_MODE
```

### Issue: "HF API failed after 3 attempts"

**Cause**: API key invalid or model not accessible

**Solution**:
1. Verify API key: https://huggingface.co/settings/tokens
2. Check model access: https://huggingface.co/google/siglip2-so400m-patch14-384
3. System will automatically fall back to local processing

### Issue: Still slow after enabling remote mode

**Cause**: Specialized embeddings still use local mode (by design)

**Explanation**: 
- Basic visual embedding: Uses remote (fast)
- Specialized embeddings (color, texture, material, style): Use local (text-guided, not available via API)

**Solution**: This is expected behavior. Remote mode speeds up the main bottleneck.

## Rollback to Local Mode

If you need to disable remote mode:

```bash
# Edit service file
sudo nano /etc/systemd/system/mivaa-pdf-extractor.service

# Change to:
Environment="VISUAL_EMBEDDING_MODE=local"

# Or remove the line entirely (defaults to local)

# Restart
sudo systemctl daemon-reload
sudo systemctl restart mivaa-pdf-extractor
```

## Summary

✅ **What Changed:**
- Added dual-mode support (local/remote)
- Upgraded to SigLIP v2 (both local and remote)
- Added Hugging Face API integration
- Automatic fallback to local on API failure

✅ **What to Do:**
1. Get Hugging Face API key
2. Add environment variables to systemd service
3. Restart service
4. Monitor logs to verify

✅ **Expected Results:**
- ~15x faster visual embeddings
- ~$0.02 per job API costs
- Automatic fallback if API fails

