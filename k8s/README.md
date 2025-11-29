# Kubernetes Deployment - MIVAA PDF Extractor

## Architecture Overview

### **Cost-Optimized Auto-Scaling Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                     Ingress (nginx)                         │
│              v1api.materialshub.gr                          │
│              (Single Load Balancer)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                Service (ClusterIP)                          │
│              mivaa-pdf-extractor:80                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  KEDA ScaledObject                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Trigger: Supabase background_jobs table              │   │
│  │ Query: COUNT(*) WHERE status IN (pending, running)   │   │
│  │ Scale to 0 when: No jobs in queue                    │   │
│  │ Scale up when: Jobs detected                         │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Deployment (0-5 replicas)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Pod 0      │  │   Pod 1      │  │   Pod 2-5    │      │
│  │  (idle=off)  │  │  (on demand) │  │  (on demand) │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Key Changes from Previous Setup

### ❌ **BEFORE (Wasteful)**
- **2 Load Balancers**: Service (LoadBalancer) + Ingress = 2 LBs
- **Always 2 pods running**: minReplicas=2 (wasting resources 24/7)
- **No scale-to-zero**: Pods running even when idle
- **Cost**: ~$40/month for 2 pods + 2 load balancers

### ✅ **AFTER (Optimized)**
- **1 Load Balancer**: Ingress only (Service is ClusterIP)
- **Scale to 0**: KEDA scales to 0 pods when no jobs
- **Auto-scale on demand**: 0→5 pods based on job queue
- **Cost**: ~$5/month (only when processing jobs)

## Components

### 1. **Service (ClusterIP)**
- **Type**: ClusterIP (internal only)
- **Port**: 80 → 8000
- **Purpose**: Internal routing to pods
- **No external IP**: Accessed via Ingress only

### 2. **Ingress (nginx)**
- **Domain**: v1api.materialshub.gr
- **TLS**: Let's Encrypt (automatic)
- **Load Balancer**: Single DigitalOcean LB
- **Purpose**: External access + SSL termination

### 3. **KEDA ScaledObject**
- **Min replicas**: 0 (scale to zero!)
- **Max replicas**: 5
- **Trigger**: PostgreSQL query on background_jobs table
- **Scale up when**: Jobs with status IN ('pending', 'running', 'processing')
- **Scale down when**: No jobs for 5 minutes (cooldown)
- **Polling**: Every 30 seconds

### 4. **HPA (Fallback)**
- **Min replicas**: 1
- **Max replicas**: 5
- **Triggers**: Memory (60%) + CPU (70%)
- **Purpose**: Backup if KEDA fails

### 5. **Deployment**
- **Replicas**: Managed by KEDA (not hardcoded)
- **Resources**: 2Gi-3.5Gi memory, 800m-1800m CPU
- **Strategy**: RollingUpdate (zero downtime)

## Scaling Behavior

### **Idle State (No Jobs)**
```
Jobs in queue: 0
KEDA decision: Scale to 0
Pods running: 0
Cost: $0/hour
```

### **Job Submitted**
```
Jobs in queue: 1
KEDA decision: Scale to 1 pod
Time to scale up: ~30 seconds
Pod starts processing
```

### **Heavy Load (Multiple Jobs)**
```
Jobs in queue: 10
Memory usage: 70%
KEDA + HPA decision: Scale to 5 pods
Pods process jobs in parallel
```

### **Jobs Complete**
```
Jobs in queue: 0
Cooldown: Wait 5 minutes
KEDA decision: Scale to 0
Pods terminate gracefully
```

## Deployment

### **Automatic (GitHub Actions)**
```bash
git push origin main
# Triggers: .github/workflows/k8s-deploy.yml
# 1. Build Docker image → ghcr.io
# 2. Install KEDA (if needed)
# 3. Deploy all K8s resources
# 4. Wait for rollout
```

### **Manual**
```bash
# Install KEDA
kubectl apply -f https://github.com/kedacore/keda/releases/download/v2.15.1/keda-2.15.1.yaml

# Deploy resources
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/keda-scaledobject.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/pdb.yaml
kubectl apply -f k8s/ingress.yaml
```

## Monitoring

### **Check Scaling Status**
```bash
# KEDA ScaledObject
kubectl get scaledobject -n default
kubectl describe scaledobject mivaa-pdf-extractor-scaler

# HPA (fallback)
kubectl get hpa -n default

# Pods
kubectl get pods -n default -l app=mivaa-pdf-extractor -w

# Events
kubectl get events -n default --sort-by='.lastTimestamp'
```

### **Check Job Queue**
```bash
# Connect to Supabase
psql "postgresql://postgres.bgbavxtjlbvgplozizxu:PASSWORD@aws-0-eu-west-3.pooler.supabase.com:6543/postgres"

# Check pending jobs
SELECT COUNT(*) FROM background_jobs WHERE status IN ('pending', 'running', 'processing');
```

## Troubleshooting

### **Pods not scaling up**
```bash
# Check KEDA logs
kubectl logs -n keda-system -l app=keda-operator

# Check ScaledObject status
kubectl describe scaledobject mivaa-pdf-extractor-scaler

# Verify Supabase connection
kubectl get secret keda-supabase-secret -o yaml
```

### **Pods not scaling down**
```bash
# Check cooldown period (5 minutes)
kubectl describe scaledobject mivaa-pdf-extractor-scaler | grep -A 10 "Cooldown"

# Force scale down (testing only)
kubectl scale deployment mivaa-pdf-extractor --replicas=0
```

### **Service not accessible**
```bash
# Check Ingress
kubectl get ingress -n default
kubectl describe ingress mivaa-pdf-extractor-ingress

# Check Service
kubectl get svc mivaa-pdf-extractor

# Check DNS
nslookup v1api.materialshub.gr
```

## Cost Savings

### **Before**
- 2 pods × 24 hours × 30 days = 1,440 pod-hours/month
- 2 load balancers × $10/month = $20/month
- **Total**: ~$40/month

### **After**
- Average 2 hours/day processing = 60 pod-hours/month
- 1 load balancer × $10/month = $10/month
- **Total**: ~$15/month

**Savings**: ~$25/month (62% reduction)

