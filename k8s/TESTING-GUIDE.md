# Kubernetes Autoscaling Testing Guide

## 🎯 Overview

This guide walks you through testing the complete Kubernetes autoscaling setup:
- **Cluster Autoscaler (CA)**: Scales nodes 1-4 based on pod scheduling needs
- **HPA**: Scales pods 1-8 based on CPU (60%) and memory (70%)
- **KEDA**: Temporarily disabled (will re-enable later for scale-to-zero)

---

## 📋 Prerequisites

1. ✅ Deployment run #40 succeeded
2. ✅ Service changed to ClusterIP
3. ✅ Old LoadBalancer deleted
4. ⚠️  Need to run complete setup script

---

## 🚀 Step 1: Run Complete Setup Script

This script will:
- Delete old LoadBalancer service (if exists)
- Scale down to 1 node
- Enable node autoscaling (1-4 nodes)
- Verify final state

```bash
cd mivaa-pdf-extractor/k8s
chmod +x complete-setup.sh
./complete-setup.sh
```

**Expected Output:**
```
✅ Deployment: Run #40 succeeded
✅ Service: Changed to ClusterIP
✅ Old LoadBalancer: Deleted
✅ Nodes: Scaled to 1
✅ Node Autoscaling: Enabled (1-4 nodes)
✅ Pod Autoscaling: HPA enabled (1-8 pods)
```

---

## 🧪 Step 2: Test Pod Autoscaling (HPA)

### 2.1 Monitor Initial State

Open 3 terminal windows:

**Terminal 1 - Watch Pods:**
```bash
watch kubectl get pods -n default -o wide
```

**Terminal 2 - Watch Nodes:**
```bash
watch kubectl get nodes -o wide
```

**Terminal 3 - Watch HPA:**
```bash
watch kubectl get hpa -n default
```

**Expected Initial State:**
- Nodes: 1 node running
- Pods: 1 pod running (HPA minReplicas: 1)
- HPA: CPU/Memory at low %

### 2.2 Submit Test PDF

Submit a PDF for processing (use NOVA or Harmony PDF):

```bash
# Via API
curl -X POST https://v1api.materialshub.gr/api/v1/rag/upload-pdf \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@/path/to/test.pdf"

# Or via frontend
# Upload PDF through the admin panel
```

### 2.3 Observe Pod Scaling

**What Should Happen:**
1. **CPU/Memory increases** as PDF processing starts
2. **HPA triggers** when CPU > 60% or Memory > 70%
3. **Pods scale up** from 1 → 2-8 based on load
4. **Processing distributes** across multiple pods

**Timeline:**
- 0-30s: Initial processing on 1 pod
- 30-60s: HPA detects high CPU/memory
- 60-90s: New pods start (Pending → Running)
- 90s+: Processing distributed across pods

---

## 🧪 Step 3: Test Node Autoscaling (CA)

### 3.1 Trigger Node Scaling

If pods can't be scheduled on existing nodes (Pending state), CA will add nodes.

**Force Node Scaling (Optional):**
```bash
# Temporarily increase HPA max to force more pods
kubectl patch hpa mivaa-pdf-extractor -n default -p '{"spec":{"maxReplicas":12}}'

# Submit multiple PDFs simultaneously
# This will create many pods that can't fit on 1 node
```

### 3.2 Observe Node Scaling

**What Should Happen:**
1. **Pods go Pending** (can't be scheduled on existing nodes)
2. **CA detects** pending pods
3. **New nodes added** (1 → 2-4 nodes)
4. **Pods scheduled** on new nodes
5. **Processing continues**

**Timeline:**
- 0-60s: Pods pending (waiting for nodes)
- 60-180s: New nodes provisioning (DigitalOcean)
- 180-240s: Nodes ready, pods scheduled
- 240s+: Processing on multiple nodes

**Check CA Status:**
```bash
kubectl get configmap cluster-autoscaler-status -n kube-system -oyaml
kubectl describe nodes
```

---

## 🧪 Step 4: Test Scale Down

### 4.1 Wait for Processing to Complete

After PDF processing completes:

**What Should Happen:**
1. **CPU/Memory drops** to low levels
2. **HPA cooldown** (5 minutes)
3. **Pods scale down** from 8 → 1
4. **CA cooldown** (10 minutes)
5. **Nodes scale down** from 4 → 1

**Timeline:**
- 0-5min: HPA cooldown period
- 5-10min: Pods scale down to 1
- 10-20min: CA cooldown period
- 20-30min: Nodes scale down to 1

### 4.2 Monitor Scale Down

```bash
# Watch pods scale down
watch kubectl get pods -n default

# Watch nodes scale down
watch kubectl get nodes

# Check HPA metrics
kubectl get hpa -n default
```

---

## 📊 Step 5: Verify Final State

After scale down completes:

```bash
# Should see 1 node
kubectl get nodes

# Should see 1 pod
kubectl get pods -n default

# Should see only 1 LoadBalancer (from Ingress)
doctl compute load-balancer list

# Check HPA is ready for next scale up
kubectl get hpa -n default
```

**Expected Final State:**
- ✅ 1 node running
- ✅ 1 pod running
- ✅ 1 LoadBalancer (Ingress only)
- ✅ HPA ready (minReplicas: 1, maxReplicas: 8)
- ✅ CA ready (minNodes: 1, maxNodes: 4)

---

## 🐛 Troubleshooting

### Pods Not Scaling Up

```bash
# Check HPA status
kubectl describe hpa mivaa-pdf-extractor -n default

# Check metrics server
kubectl top pods -n default
kubectl top nodes

# Check HPA events
kubectl get events -n default --sort-by='.lastTimestamp'
```

### Nodes Not Scaling Up

```bash
# Check CA status
kubectl get configmap cluster-autoscaler-status -n kube-system -oyaml

# Check pending pods
kubectl get pods -n default | grep Pending

# Check node pool autoscaling
doctl kubernetes cluster node-pool get e56b1987-f9d0-4e4d-8e50-b27e12592f19 <NODE_POOL_ID>
```

### Pods Stuck in Pending

```bash
# Check pod events
kubectl describe pod <POD_NAME> -n default

# Check node resources
kubectl describe nodes

# Check if nodes are being added
kubectl get nodes -w
```

---

## 📈 Success Criteria

- ✅ Pods scale from 1 → 8 based on CPU/memory load
- ✅ Nodes scale from 1 → 4 when pods can't be scheduled
- ✅ Pods scale back to 1 after processing completes
- ✅ Nodes scale back to 1 after pods are removed
- ✅ Only 1 LoadBalancer exists (from Ingress)
- ✅ No old LoadBalancer services
- ✅ Cost optimized: ~$24/month idle, ~$72/month peak

---

## 🔄 Next Steps

After successful testing:

1. **Re-enable KEDA** for scale-to-zero capability
2. **Monitor production** workloads
3. **Adjust thresholds** if needed (HPA CPU/memory %)
4. **Document** any issues or optimizations

---

## 📞 Support

If you encounter issues:
1. Check logs: `kubectl logs -n default <POD_NAME>`
2. Check events: `kubectl get events -n default`
3. Check Sentry for errors
4. Review this guide's troubleshooting section

