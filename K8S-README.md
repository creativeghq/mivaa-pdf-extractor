# Kubernetes Files - Quick Reference

All Kubernetes-related files are prefixed with `k8s-` for easy identification and rollback.

---

## 📁 File Structure

```
mivaa-pdf-extractor/
├── k8s-Dockerfile              # Multi-stage Docker image (CPU-only PyTorch)
├── k8s-deployment.yaml         # Kubernetes Deployment (pods, replicas, resources)
├── k8s-service.yaml            # Kubernetes Service (LoadBalancer)
├── k8s-hpa.yaml                # Horizontal Pod Autoscaler (2-10 replicas)
├── k8s-configmap.yaml          # ConfigMap (non-sensitive config)
├── k8s-secrets.yaml.template   # Secrets template (DO NOT commit with real values!)
├── k8s-ingress.yaml            # Ingress (SSL, domain routing)
├── k8s-pdb.yaml                # Pod Disruption Budget (high availability)
├── k8s-setup-cluster.sh        # Script: Create cluster + infrastructure
├── k8s-setup-secrets.sh        # Script: Create secrets from .env
├── k8s-rollback.sh             # Script: Rollback to previous version
├── k8s-delete-cluster.sh       # Script: Delete entire cluster (DANGER!)
├── K8S-DEPLOYMENT-GUIDE.md     # Complete deployment documentation
└── K8S-README.md               # This file
```

---

## 🚀 Quick Start

### 1. Initial Setup (One-time)

```bash
# Install prerequisites
brew install doctl kubectl  # macOS
# or apt-get install doctl kubectl  # Linux

# Authenticate with DigitalOcean
doctl auth init

# Create cluster and infrastructure
chmod +x k8s-setup-cluster.sh
./k8s-setup-cluster.sh

# Create secrets
chmod +x k8s-setup-secrets.sh
./k8s-setup-secrets.sh
```

### 2. Deploy Application

**Option A: Automated (GitHub Actions)**
```bash
git add .
git commit -m "k8s: Deploy to Kubernetes"
git push origin main
```

**Option B: Manual**
```bash
# Build and push image
docker build -f k8s-Dockerfile -t registry.digitalocean.com/mivaa-registry/mivaa-pdf-extractor:latest .
docker push registry.digitalocean.com/mivaa-registry/mivaa-pdf-extractor:latest

# Deploy to K8s
kubectl apply -f k8s-configmap.yaml
kubectl apply -f k8s-service.yaml
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-hpa.yaml
kubectl apply -f k8s-pdb.yaml
kubectl apply -f k8s-ingress.yaml
```

### 3. Monitor Deployment

```bash
# Watch pods
kubectl get pods -n default -w

# Check logs
kubectl logs -f deployment/mivaa-pdf-extractor -n default

# Get LoadBalancer IP
kubectl get svc mivaa-pdf-extractor -n default
```

---

## 📝 Common Commands

### Deployment Management

```bash
# Get deployment status
kubectl get deployment mivaa-pdf-extractor -n default

# Scale manually
kubectl scale deployment mivaa-pdf-extractor --replicas=5 -n default

# Restart deployment
kubectl rollout restart deployment/mivaa-pdf-extractor -n default

# Check rollout status
kubectl rollout status deployment/mivaa-pdf-extractor -n default
```

### Logs & Debugging

```bash
# Follow logs
kubectl logs -f deployment/mivaa-pdf-extractor -n default

# Logs from specific pod
kubectl logs <pod-name> -n default

# Previous logs (if crashed)
kubectl logs --previous <pod-name> -n default

# Describe pod
kubectl describe pod <pod-name> -n default
```

### Secrets & Config

```bash
# Update secret
kubectl create secret generic mivaa-secrets \
  --from-literal=KEY=value \
  --namespace=default \
  --dry-run=client -o yaml | kubectl apply -f -

# Edit ConfigMap
kubectl edit configmap mivaa-config -n default

# Restart to pick up changes
kubectl rollout restart deployment/mivaa-pdf-extractor -n default
```

### Auto-Scaling

```bash
# Get HPA status
kubectl get hpa -n default

# Describe HPA
kubectl describe hpa mivaa-pdf-extractor-hpa -n default

# Watch HPA
kubectl get hpa -n default -w
```

---

## ⏪ Rollback

```bash
# Quick rollback (automated)
./k8s-rollback.sh

# Manual rollback
kubectl rollout undo deployment/mivaa-pdf-extractor -n default

# Rollback to specific revision
kubectl rollout undo deployment/mivaa-pdf-extractor --to-revision=2 -n default
```

---

## 🗑️ Cleanup

```bash
# Delete deployment only
kubectl delete deployment mivaa-pdf-extractor -n default

# Delete all K8s resources
kubectl delete -f k8s-deployment.yaml
kubectl delete -f k8s-service.yaml
kubectl delete -f k8s-hpa.yaml
kubectl delete -f k8s-pdb.yaml
kubectl delete -f k8s-ingress.yaml

# Delete entire cluster (DANGER!)
./k8s-delete-cluster.sh
```

---

## 📊 Resource Specifications

| Resource | Min | Max | Request | Limit |
|----------|-----|-----|---------|-------|
| **Replicas** | 2 | 10 | - | - |
| **CPU** | - | - | 500m | 2000m |
| **Memory** | - | - | 2Gi | 4Gi |

**Auto-scaling triggers:**
- Scale UP: CPU > 70% OR Memory > 80%
- Scale DOWN: CPU < 50% AND Memory < 60% (after 5 min)

---

## 🔗 Important Links

- **Full Documentation:** [K8S-DEPLOYMENT-GUIDE.md](./K8S-DEPLOYMENT-GUIDE.md)
- **GitHub Actions:** `.github/workflows/k8s-deploy.yml`
- **DigitalOcean Console:** https://cloud.digitalocean.com/kubernetes
- **Container Registry:** https://cloud.digitalocean.com/registry

---

## ⚠️ Important Notes

1. **Never commit secrets!** Use `k8s-secrets.yaml.template` only
2. **All K8s files prefixed with `k8s-`** for easy rollback
3. **GitHub Actions auto-deploys** on push to main
4. **DNS must be updated** after first deployment
5. **SSL certificates** auto-generated by cert-manager

---

**Need help?** See [K8S-DEPLOYMENT-GUIDE.md](./K8S-DEPLOYMENT-GUIDE.md) for complete documentation.

