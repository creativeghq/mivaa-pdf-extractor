# Job Recovery and Signal Handling Guide

## Overview

This document explains the job recovery system and signal handling mechanisms implemented in the MIVAA PDF Extractor service to ensure reliable processing and recovery from interruptions.

---

## 🔄 Job Recovery System

### Purpose

The job recovery system ensures that background PDF processing jobs can be tracked, monitored, and recovered after service interruptions or restarts.

### Key Features

1. **Database Persistence**: All jobs are persisted to the `background_jobs` table
2. **Automatic Recovery**: Interrupted jobs are automatically detected on startup
3. **Status Tracking**: Complete lifecycle tracking (pending → processing → completed/failed/interrupted)
4. **Job Statistics**: Real-time monitoring of job counts and status
5. **Auto-Cleanup**: Automatic removal of old completed/failed jobs (7+ days)

### Database Schema

```sql
CREATE TABLE background_jobs (
  id UUID PRIMARY KEY,
  document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
  filename TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'interrupted')),
  progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
  metadata JSONB DEFAULT '{}'::jsonb,
  error TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  failed_at TIMESTAMPTZ,
  interrupted_at TIMESTAMPTZ
);
```

### Job Lifecycle

```
┌─────────┐
│ PENDING │ ──────────────────────────────────┐
└─────────┘                                    │
     │                                         │
     │ Start Processing                        │
     ▼                                         │
┌────────────┐                                 │
│ PROCESSING │                                 │
└────────────┘                                 │
     │                                         │
     ├──────────────┬──────────────┬───────────┤
     │              │              │           │
     ▼              ▼              ▼           ▼
┌───────────┐  ┌────────┐  ┌─────────────┐  ┌────────────┐
│ COMPLETED │  │ FAILED │  │ INTERRUPTED │  │ INTERRUPTED│
└───────────┘  └────────┘  └─────────────┘  └────────────┘
                                (Shutdown)    (Cancellation)
```

### Usage

#### Initialize Job Recovery (Automatic on Startup)

```python
from app.api.rag_routes import initialize_job_recovery

# Called automatically in main.py lifespan
await initialize_job_recovery()
```

#### Persist Job Status

```python
from app.services.job_recovery_service import JobRecoveryService

job_recovery = JobRecoveryService(supabase_client)

# Persist job
await job_recovery.persist_job(
    job_id="abc-123",
    document_id="doc-456",
    filename="document.pdf",
    status="processing",
    progress=50,
    metadata={"title": "My Document"}
)
```

#### Get Interrupted Jobs

```python
interrupted_jobs = await job_recovery.get_interrupted_jobs()
for job in interrupted_jobs:
    print(f"Job {job['id']} was interrupted: {job['error']}")
```

#### Get Job Statistics

```python
stats = await job_recovery.get_job_statistics()
# Returns: {'total': 10, 'pending': 2, 'processing': 1, 'completed': 5, 'failed': 1, 'interrupted': 1}
```

---

## 🛑 Signal Handling

### Purpose

Signal handling ensures that service interruptions are properly logged and jobs are marked as interrupted for later recovery.

### Supported Signals

| Signal   | Description                | Can Be Caught? | Logged? |
|----------|----------------------------|----------------|---------|
| SIGTERM  | Graceful shutdown          | ✅ Yes         | ✅ Yes  |
| SIGINT   | Ctrl+C / Keyboard interrupt| ✅ Yes         | ✅ Yes  |
| SIGHUP   | Terminal closed            | ✅ Yes         | ✅ Yes  |
| SIGKILL  | Force kill                 | ❌ No          | ❌ No   |

### Important Notes

#### 1. SIGKILL Cannot Be Caught

**Problem**: When using `pkill -9` or `SIGKILL`, the process is terminated immediately without any cleanup.

**Why**: SIGKILL is a kernel-level signal that cannot be intercepted by the application.

**Impact**:
- No shutdown logs are written
- Jobs remain in "processing" state in database
- No graceful cleanup occurs

**Solution**: Use graceful shutdown instead:
```bash
# ✅ GOOD - Graceful shutdown
sudo systemctl stop mivaa-pdf-extractor
sudo systemctl restart mivaa-pdf-extractor

# ❌ BAD - Force kill (use only as last resort)
sudo pkill -9 uvicorn
```

**When to Use SIGKILL**:
- Only when service is completely unresponsive
- As a last resort after graceful shutdown fails
- When service is stuck in "deactivating" state for >30 seconds

#### 2. Service Restart Behavior

**Problem**: When `systemctl restart` is used while a job is processing, the service enters "deactivating" state and waits for jobs to complete.

**Why**: SystemD tries to gracefully shutdown the service by sending SIGTERM and waiting for the process to exit.

**Impact**:
- Service appears "stuck" in deactivating state
- Can take several minutes if PDF processing is ongoing
- May timeout and force kill after 90 seconds (default)

**Solution Options**:

**Option A: Wait for Graceful Shutdown (Recommended)**
```bash
# Restart and wait
sudo systemctl restart mivaa-pdf-extractor

# Monitor status
sudo systemctl status mivaa-pdf-extractor

# Check logs
sudo journalctl -u mivaa-pdf-extractor -f
```

**Option B: Stop First, Then Start**
```bash
# Stop service (waits for jobs)
sudo systemctl stop mivaa-pdf-extractor

# Wait for complete shutdown
sleep 5

# Start service
sudo systemctl start mivaa-pdf-extractor
```

**Option C: Force Kill (Last Resort)**
```bash
# Only if service is stuck >30 seconds
sudo pkill -9 uvicorn
sudo systemctl start mivaa-pdf-extractor
```

#### 3. Job Recovery After Interruption

**Current Behavior**: Jobs are marked as "interrupted" but NOT automatically resumed.

**Why**: Prevents duplicate processing and allows manual review of interrupted jobs.

**Workflow**:

1. **Service Starts**: Job recovery service initializes
2. **Detection**: All "processing" or "pending" jobs are found
3. **Marking**: Jobs are marked as "interrupted" with reason
4. **Logging**: Interrupted jobs are logged for visibility
5. **Manual Review**: Admin can review and decide whether to reprocess

**Example Logs**:
```
🔄 Initializing job recovery service...
🛑 Marked 2 jobs as interrupted due to: Service restart detected
   - Job abc-123
   - Job def-456
📊 Job statistics: {'total': 10, 'interrupted': 2, 'completed': 8}
```

**Future Enhancement**: Automatic job resumption could be added with:
- Checkpoint/resume capability
- Idempotency checks
- Configurable retry policies

---

## 📊 Monitoring and Logging

### Startup Logs

```
🔄 Initializing job recovery service...
✅ No interrupted jobs found
📊 Job statistics: {'total': 5, 'pending': 0, 'processing': 0, 'completed': 5, 'failed': 0, 'interrupted': 0}
🧹 Cleaned up 3 old jobs
✅ Job recovery service initialized successfully
```

### Shutdown Logs

```
================================================================================
🛑 SHUTDOWN INITIATED
🛑 Shutdown time: 2025-10-21T14:00:00
================================================================================
🛑 SHUTDOWN WARNING: 1 jobs still processing:
   - Job abc-123: Document doc-456, Started: 2025-10-21T13:58:00
✅ No active jobs during shutdown
================================================================================
🛑 SHUTDOWN COMPLETE
================================================================================
```

### Signal Handling Logs

```
🛑 SERVICE INTERRUPTION DETECTED: SIGTERM (graceful shutdown)
🛑 Received signal 15 at 2025-10-21T14:00:00
🛑 Initiating graceful shutdown...
🛑 Any ongoing PDF processing jobs will be interrupted!
✅ No active jobs to interrupt
```

### Job Processing Logs

```
📋 BACKGROUND JOB STARTED: abc-123
   Document ID: doc-456
   Filename: document.pdf
   Started at: 2025-10-21T14:00:00

📋 BACKGROUND JOB FINISHED: abc-123
   Final status: completed
   Total duration: 45.23s
   Ended at: 2025-10-21T14:00:45
```

### Job Interruption Logs

```
🛑 JOB INTERRUPTED: abc-123
   Document ID: doc-456
   Filename: document.pdf
   Reason: Service shutdown or task cancellation
   Duration before interruption: 30.15s
```

---

## 🔧 Best Practices

### 1. Service Restarts

**DO**:
- Use `systemctl restart` for normal restarts
- Wait for graceful shutdown (check logs)
- Monitor job status before/after restart
- Review interrupted jobs after restart

**DON'T**:
- Use `pkill -9` unless absolutely necessary
- Restart during active processing (if avoidable)
- Ignore interrupted job warnings

### 2. Deployment

**Recommended Deployment Process**:
```bash
# 1. Pull latest code
cd /var/www/mivaa-pdf-extractor
git pull origin main

# 2. Check for active jobs
curl -s "https://v1api.materialshub.gr/api/rag/admin/jobs" | jq '.active_jobs'

# 3. Wait for jobs to complete OR proceed with restart
sudo systemctl restart mivaa-pdf-extractor

# 4. Monitor startup
sudo journalctl -u mivaa-pdf-extractor -f

# 5. Verify service is active
sudo systemctl is-active mivaa-pdf-extractor

# 6. Check job recovery logs
sudo journalctl -u mivaa-pdf-extractor --since "1 minute ago" | grep "job recovery"
```

### 3. Monitoring

**Key Metrics to Monitor**:
- Number of interrupted jobs on startup
- Job completion rate
- Average processing time
- Failed job count
- Service restart frequency

**Monitoring Queries**:
```sql
-- Get interrupted jobs
SELECT * FROM background_jobs WHERE status = 'interrupted' ORDER BY created_at DESC;

-- Get job statistics
SELECT status, COUNT(*) FROM background_jobs GROUP BY status;

-- Get recent failures
SELECT * FROM background_jobs WHERE status = 'failed' AND created_at > NOW() - INTERVAL '24 hours';
```

---

## 🚨 Troubleshooting

### Service Won't Stop

**Symptom**: Service stuck in "deactivating" state

**Cause**: Long-running job in progress

**Solution**:
```bash
# Check what's running
sudo journalctl -u mivaa-pdf-extractor -f

# Wait 30 seconds, then force kill if necessary
sudo pkill -9 uvicorn
sudo systemctl start mivaa-pdf-extractor
```

### Jobs Not Being Recovered

**Symptom**: Interrupted jobs not showing in logs

**Cause**: Job recovery service failed to initialize

**Solution**:
```bash
# Check initialization logs
sudo journalctl -u mivaa-pdf-extractor --since "5 minutes ago" | grep "job recovery"

# Verify database table exists
psql -h db.bgbavxtjlbvgplozizxu.supabase.co -U postgres -d postgres -c "\d background_jobs"

# Restart service
sudo systemctl restart mivaa-pdf-extractor
```

### Signal Handler Not Logging

**Symptom**: No signal logs during shutdown

**Cause**: Using SIGKILL instead of SIGTERM

**Solution**: Use graceful shutdown commands (systemctl stop/restart)

---

## 📚 Related Documentation

- [PDF Processing Guide](./PDF_PROCESSING_GUIDE.md)
- [Deployment Guide](./DEPLOYMENT_GUIDE.md)
- [Monitoring Guide](./MONITORING_GUIDE.md)

