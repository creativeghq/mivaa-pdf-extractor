# Kubernetes Deployment Guide - MIVAA PDF Extractor

Complete guide for deploying MIVAA PDF Extractor to DigitalOcean Kubernetes (DOKS).

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Initial Setup](#initial-setup)
4. [Deployment](#deployment)
5. [Monitoring & Maintenance](#monitoring--maintenance)
6. [Troubleshooting](#troubleshooting)
7. [Rollback](#rollback)
8. [Cost Estimation](#cost-estimation)

---

## 🔧 Prerequisites

### Required Tools

1. **doctl** (DigitalOcean CLI)
   ```bash
   # macOS
   brew install doctl
   
   # Linux
   cd ~
   wget https://github.com/digitalocean/doctl/releases/download/v1.98.1/doctl-1.98.1-linux-amd64.tar.gz
   tar xf doctl-1.98.1-linux-amd64.tar.gz
   sudo mv doctl /usr/local/bin
   ```

2. **kubectl** (Kubernetes CLI)
   ```bash
   # macOS
   brew install kubectl
   
   # Linux
   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
   sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
   ```

3. **Docker** (for local testing)
   ```bash
   # Install from: https://docs.docker.com/get-docker/
   ```

### DigitalOcean Account Setup

1. **Create API Token**
   - Go to: https://cloud.digitalocean.com/account/api/tokens
   - Click "Generate New Token"
   - Name: "MIVAA K8s Deployment"
   - Scopes: Read & Write
   - Copy the token (you'll only see it once!)

2. **Authenticate doctl**
   ```bash
   doctl auth init
   # Paste your API token when prompted
   ```

3. **Verify authentication**
   ```bash
   doctl account get
   ```

### GitHub Secrets Setup

Add these secrets to your GitHub repository:

1. Go to: `Settings` → `Secrets and variables` → `Actions`
2. Add the following secrets:

| Secret Name | Value | Description |
|------------|-------|-------------|
| `DIGITALOCEAN_ACCESS_TOKEN` | Your DO API token | For cluster access |
| `SUPABASE_ANON_KEY` | Your Supabase anon key | Database access |
| `SUPABASE_SERVICE_ROLE_KEY` | Your Supabase service key | Admin database access |
| `ANTHROPIC_API_KEY` | Your Anthropic API key | Claude AI |
| `OPENAI_API_KEY` | Your OpenAI API key | GPT models |
| `TOGETHER_API_KEY` | Your Together API key | Llama models |
| `HUGGINGFACE_API_KEY` | Your HuggingFace key | Model downloads |
| `JWT_SECRET_KEY` | Random secure string | JWT signing |
| `ENCRYPTION_KEY` | Random secure string | Data encryption |

---

## 🏗️ Architecture Overview

### Cluster Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DigitalOcean Load Balancer               │
│                    (v1api.materialshub.gr)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Nginx Ingress Controller                 │
│                    (SSL/TLS Termination)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Service                       │
│                    (Load Balancing)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌────────┐      ┌────────┐      ┌────────┐
    │  Pod 1 │      │  Pod 2 │      │  Pod N │
    │ 2GB RAM│      │ 2GB RAM│      │ 2GB RAM│
    │ 0.5 CPU│      │ 0.5 CPU│      │ 0.5 CPU│
    └────────┘      └────────┘      └────────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
              ┌──────────────────────┐
              │  Supabase Database   │
              │  (External Service)  │
              └──────────────────────┘
```

### Resource Allocation

| Component | Min Replicas | Max Replicas | CPU Request | Memory Request | CPU Limit | Memory Limit |
|-----------|--------------|--------------|-------------|----------------|-----------|--------------|
| MIVAA Pod | 2 | 10 | 500m | 2Gi | 2000m | 4Gi |

### Auto-Scaling Triggers

- **Scale UP** when:
  - CPU > 70%
  - Memory > 80%
  
- **Scale DOWN** when:
  - CPU < 50% for 5 minutes
  - Memory < 60% for 5 minutes

---

## 🔍 Monitoring & Maintenance

### Check Deployment Status

```bash
# Get all resources
kubectl get all -n default

# Get pods
kubectl get pods -n default -l app=mivaa-pdf-extractor

# Get services
kubectl get svc -n default

# Get ingress
kubectl get ingress -n default

# Get HPA status
kubectl get hpa -n default
```

### View Logs

```bash
# Follow logs from all pods
kubectl logs -f deployment/mivaa-pdf-extractor -n default

# Logs from specific pod
kubectl logs -f <pod-name> -n default

# Previous logs (if pod crashed)
kubectl logs --previous <pod-name> -n default

# Logs from last hour
kubectl logs --since=1h deployment/mivaa-pdf-extractor -n default
```

### Health Checks

```bash
# Get LoadBalancer IP
EXTERNAL_IP=$(kubectl get svc mivaa-pdf-extractor -n default -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "External IP: $EXTERNAL_IP"

# Test health endpoint
curl http://$EXTERNAL_IP/health

# Test via domain (after DNS update)
curl https://v1api.materialshub.gr/health
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment mivaa-pdf-extractor --replicas=5 -n default

# Check HPA status
kubectl get hpa -n default

# Describe HPA for details
kubectl describe hpa mivaa-pdf-extractor-hpa -n default
```

### Update Secrets

```bash
# Update a single secret
kubectl create secret generic mivaa-secrets \
  --from-literal=OPENAI_API_KEY=new_key_here \
  --namespace=default \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up new secrets
kubectl rollout restart deployment/mivaa-pdf-extractor -n default
```

### Update ConfigMap

```bash
# Edit ConfigMap
kubectl edit configmap mivaa-config -n default

# Or apply updated file
kubectl apply -f k8s-configmap.yaml

# Restart pods to pick up changes
kubectl rollout restart deployment/mivaa-pdf-extractor -n default
```

---

## 🐛 Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl get pods -n default

# Describe pod for events
kubectl describe pod <pod-name> -n default

# Check logs
kubectl logs <pod-name> -n default

# Common issues:
# 1. Image pull errors → Check registry credentials
# 2. CrashLoopBackOff → Check logs for application errors
# 3. Pending → Check resource availability
```

### Image Pull Errors

```bash
# Recreate registry secret
kubectl delete secret docr-secret -n default
kubectl create secret docker-registry docr-secret \
  --docker-server=registry.digitalocean.com \
  --docker-username=$DIGITALOCEAN_ACCESS_TOKEN \
  --docker-password=$DIGITALOCEAN_ACCESS_TOKEN \
  --namespace=default
```

### Service Not Accessible

```bash
# Check service
kubectl get svc mivaa-pdf-extractor -n default

# Check endpoints
kubectl get endpoints mivaa-pdf-extractor -n default

# Check ingress
kubectl describe ingress mivaa-pdf-extractor-ingress -n default

# Check nginx-ingress logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller
```

### High Memory Usage

```bash
# Check pod resource usage
kubectl top pods -n default

# Check HPA metrics
kubectl get hpa -n default

# If pods are OOMKilled, increase memory limits in k8s-deployment.yaml
```

### SSL Certificate Issues

```bash
# Check certificate status
kubectl get certificate -n default

# Describe certificate
kubectl describe certificate mivaa-tls-cert -n default

# Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager
```

---

## ⏪ Rollback

### Quick Rollback

```bash
# Use the automated rollback script
chmod +x k8s-rollback.sh
./k8s-rollback.sh
```

### Manual Rollback

```bash
# View rollout history
kubectl rollout history deployment/mivaa-pdf-extractor -n default

# Rollback to previous version
kubectl rollout undo deployment/mivaa-pdf-extractor -n default

# Rollback to specific revision
kubectl rollout undo deployment/mivaa-pdf-extractor --to-revision=2 -n default

# Check rollback status
kubectl rollout status deployment/mivaa-pdf-extractor -n default
```

---

## 💰 Cost Estimation

### Monthly Costs (DigitalOcean)

| Component | Configuration | Cost/Month |
|-----------|--------------|------------|
| **Kubernetes Cluster** | Control Plane | $12 |
| **Worker Nodes (Min)** | 2x s-2vcpu-4gb | $48 ($24 each) |
| **Worker Nodes (Avg)** | 3-4x s-2vcpu-4gb | $72-96 |
| **Worker Nodes (Max)** | 10x s-2vcpu-4gb | $240 |
| **Load Balancer** | 1x LB | $12 |
| **Container Registry** | Basic tier | $5 |
| **Bandwidth** | ~1TB/month | Included |

**Total Estimated Costs:**
- **Minimum (2 nodes):** $77/month
- **Average (3-4 nodes):** $101-125/month
- **Maximum (10 nodes):** $269/month

### Cost Optimization Tips

1. **Use HPA effectively** - Let auto-scaling handle traffic spikes
2. **Set appropriate limits** - Prevent runaway scaling
3. **Monitor usage** - Review metrics weekly
4. **Scale down during low traffic** - Adjust min replicas if needed
5. **Use spot instances** - Not available on DOKS yet, but coming soon

---

## 🔗 DNS Configuration

After deployment, update your DNS:

1. Get LoadBalancer IP:
   ```bash
   kubectl get svc mivaa-pdf-extractor -n default
   ```

2. Update DNS A record:
   - **Domain:** v1api.materialshub.gr
   - **Type:** A
   - **Value:** LoadBalancer IP
   - **TTL:** 300 (5 minutes)

3. Wait for DNS propagation (5-30 minutes)

4. Verify:
   ```bash
   dig v1api.materialshub.gr
   curl https://v1api.materialshub.gr/health
   ```

---

## 📚 Additional Resources

- [DigitalOcean Kubernetes Documentation](https://docs.digitalocean.com/products/kubernetes/)
- [Kubernetes Official Documentation](https://kubernetes.io/docs/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [MIVAA Platform Documentation](../docs/)

---

## 🆘 Support

If you encounter issues:

1. Check logs: `kubectl logs -f deployment/mivaa-pdf-extractor -n default`
2. Check Sentry: https://sentry.io
3. Review GitHub Actions logs
4. Contact: basiliskan@gmail.com

---

**Last Updated:** 2025-11-28
**Version:** 1.0.0
**Cluster:** mivaa-k8s-cluster
**Region:** fra1 (Frankfurt)
## 🚀 Initial Setup

### Step 1: Create Kubernetes Cluster

Run the automated setup script:

```bash
cd mivaa-pdf-extractor
chmod +x k8s-setup-cluster.sh
./k8s-setup-cluster.sh
```

This script will:
1. ✅ Create DigitalOcean Container Registry
2. ✅ Create Kubernetes cluster (2 nodes, 2vCPU/4GB each)
3. ✅ Install nginx-ingress controller
4. ✅ Install cert-manager for SSL
5. ✅ Create Let's Encrypt ClusterIssuer
6. ✅ Connect registry to cluster

**Estimated time:** 5-10 minutes

### Step 2: Create Secrets

```bash
chmod +x k8s-setup-secrets.sh
./k8s-setup-secrets.sh
```

This will create Kubernetes secrets from your environment variables.

**Option A: Using .env file**
```bash
# Create .env file with your secrets
cat > .env << EOF
SUPABASE_ANON_KEY=your_key_here
SUPABASE_SERVICE_ROLE_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
JWT_SECRET_KEY=your_secret_here
EOF

./k8s-setup-secrets.sh
```

**Option B: Using environment variables**
```bash
export SUPABASE_ANON_KEY=your_key_here
export SUPABASE_SERVICE_ROLE_KEY=your_key_here
# ... export other variables
./k8s-setup-secrets.sh
```

---

## 📦 Deployment

### Automated Deployment (Recommended)

Push to GitHub to trigger automated deployment:

```bash
git add .
git commit -m "k8s: Deploy to Kubernetes"
git push origin main
```

GitHub Actions will:
1. Build Docker image
2. Push to DigitalOcean Container Registry
3. Deploy to Kubernetes cluster
4. Run health checks

Monitor deployment:
```bash
# Watch GitHub Actions
# Go to: https://github.com/your-repo/actions

# Or watch locally
kubectl get pods -n default -w
```

### Manual Deployment

If you prefer manual deployment:

```bash
# 1. Build and push Docker image
docker build -f k8s-Dockerfile -t registry.digitalocean.com/mivaa-registry/mivaa-pdf-extractor:latest .
docker push registry.digitalocean.com/mivaa-registry/mivaa-pdf-extractor:latest

# 2. Apply Kubernetes manifests
kubectl apply -f k8s-configmap.yaml
kubectl apply -f k8s-service.yaml
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-hpa.yaml
kubectl apply -f k8s-pdb.yaml
kubectl apply -f k8s-ingress.yaml

# 3. Wait for rollout
kubectl rollout status deployment/mivaa-pdf-extractor -n default
```

---


