# Droplet Auto-Scaling for PDF Processing

## Overview

Automatic DigitalOcean droplet resizing to handle CLIP model memory requirements while minimizing costs.

## Problem

- **CLIP/SigLIP models require 2-3GB RAM** when loaded
- **Current 8GB droplet** experiences OOM (Out of Memory) crashes during PDF processing
- **Memory spike during model loading** temporarily doubles memory usage
- **Production crashes** documented in logs (service killed by OOM killer)

## Solution

**Dynamic droplet resizing:**
- **Normal operation:** 4GB droplet ($24/month)
- **During PDF processing:** Automatically resize to 16GB ($96/month)
- **After processing:** Automatically resize back to 4GB
- **Cost:** Only pay for large size during processing hours (~$25-30/month total)

## Architecture

### Components

1. **`droplet_scaler.py`** - Python service for programmatic scaling
2. **`droplet-resize.sh`** - Shell script for manual/automated resizing
3. **Integration points** - Hooks in PDF processing pipeline

### Integration Points

#### 1. Scale UP (Before CLIP Model Loading)
**File:** `mivaa-pdf-extractor/app/api/pdf_processing/stage_3_images.py`
**Line:** ~520

```python
# AUTO-SCALE: Scale up droplet before loading CLIP models (first batch only)
if batch_num == 0:
    logger.info("   🔄 Scaling up droplet for CLIP model loading...")
    scale_success = await droplet_scaler.scale_up_for_processing()
```

**Trigger:** Before first batch of images processes (before CLIP models load)

#### 2. Model Unloading (After Stage 3)
**File:** `mivaa-pdf-extractor/app/api/pdf_processing/stage_3_images.py`
**Line:** ~630

```python
# AUTO-SCALE: Unload CLIP models to free memory
embedding_service._models_loaded = False
embedding_service._siglip_model = None
embedding_service._clip_model = None
gc.collect()
```

**Trigger:** After all images processed in Stage 3

#### 3. Scale DOWN (After Pipeline Completion)
**File:** `mivaa-pdf-extractor/app/api/pdf_processing/stage_5_quality.py`
**Line:** ~145

```python
# AUTO-SCALE: Scale down droplet after PDF processing completes
scale_down_success = await droplet_scaler.scale_down_after_processing(force=False)
```

**Trigger:** After entire PDF processing pipeline completes

## Configuration

### Environment Variables

Add to `.env` or server environment:

```bash
# Enable auto-scaling
DROPLET_AUTO_SCALE=true

# Your droplet ID (find with: doctl compute droplet list)
DROPLET_ID=your-droplet-id-here
```

### Prerequisites

1. **Install doctl CLI** on the droplet:
```bash
snap install doctl
doctl auth init
```

2. **Make script executable:**
```bash
chmod +x /var/www/mivaa-pdf-extractor/scripts/droplet-resize.sh
```

3. **Test manual scaling:**
```bash
export DROPLET_ID=your-droplet-id
/var/www/mivaa-pdf-extractor/scripts/droplet-resize.sh status
/var/www/mivaa-pdf-extractor/scripts/droplet-resize.sh up
/var/www/mivaa-pdf-extractor/scripts/droplet-resize.sh down
```

## Usage

### Automatic (Recommended)

Set environment variables and the system handles scaling automatically:

1. User uploads PDF
2. System scales to 16GB before loading CLIP models
3. PDF processes with full memory available
4. System scales back to 4GB after completion

### Manual

Use the shell script for manual control:

```bash
# Check current status
./scripts/droplet-resize.sh status

# Scale up before processing
./scripts/droplet-resize.sh up

# Scale down after processing
./scripts/droplet-resize.sh down
```

## Cost Analysis

### Without Auto-Scaling

- **Option 1:** Keep 8GB droplet → OOM crashes ❌
- **Option 2:** Upgrade to 16GB permanently → $96/month ❌

### With Auto-Scaling

- **Base:** 4GB droplet @ $24/month (720 hours)
- **Processing:** 16GB upgrade @ $96/month × (10 hours / 720 hours) = $1.33
- **Total:** ~$25-30/month ✅

**Savings:** $66-71/month vs permanent 16GB upgrade

## Safety Features

1. **Lock mechanism** - Prevents concurrent scaling operations
2. **Active job check** - Won't scale down if PDFs are processing
3. **Graceful fallback** - Continues processing if scaling fails
4. **Service restart** - Automatically restarts MIVAA after resize
5. **Memory monitoring** - Logs memory usage before/after scaling

## Monitoring

Check logs for scaling events:

```bash
# View scaling logs
journalctl -u mivaa-pdf-extractor | grep -i "scaling\|droplet"

# Check current memory
ssh root@165.227.31.109 "free -h"

# Check droplet size
doctl compute droplet get $DROPLET_ID --format Size
```

## Troubleshooting

### Scaling fails

```bash
# Check doctl authentication
doctl auth list

# Verify droplet ID
doctl compute droplet list

# Check script permissions
ls -la /var/www/mivaa-pdf-extractor/scripts/droplet-resize.sh
```

### Service doesn't restart after resize

```bash
# Manually restart
systemctl restart mivaa-pdf-extractor

# Check status
systemctl status mivaa-pdf-extractor
```

### Still getting OOM errors

1. Check if auto-scaling is enabled: `echo $DROPLET_AUTO_SCALE`
2. Verify droplet actually resized: `doctl compute droplet get $DROPLET_ID`
3. Check logs for scaling errors: `journalctl -u mivaa-pdf-extractor -n 100`

## Future Enhancements

- [ ] Add Sentry alerts for scaling events
- [ ] Implement scheduled scale-down (e.g., nightly)
- [ ] Add API endpoint for manual scaling control
- [ ] Track cost savings in analytics dashboard
- [ ] Support GPU droplet upgrades for 10x faster processing

