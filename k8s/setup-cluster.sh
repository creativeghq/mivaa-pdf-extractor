#!/bin/bash
# Kubernetes Cluster Setup Script for MIVAA PDF Extractor
# This script creates a DigitalOcean Kubernetes cluster and sets up all required resources

set -e  # Exit on error

echo "🚀 MIVAA Kubernetes Cluster Setup"
echo "=================================="
echo ""

# Configuration
CLUSTER_NAME="mivaa-k8s-cluster"
REGION="fra1"  # Frankfurt (closest to EU users)
NODE_SIZE="s-2vcpu-4gb"  # 2 vCPU, 4GB RAM
NODE_COUNT=2
REGISTRY_NAME="mivaa-registry"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if doctl is installed
if ! command -v doctl &> /dev/null; then
    echo -e "${RED}❌ doctl is not installed${NC}"
    echo "Install it from: https://docs.digitalocean.com/reference/doctl/how-to/install/"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl is not installed${NC}"
    echo "Install it from: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi

echo -e "${YELLOW}📋 Configuration:${NC}"
echo "  Cluster Name: $CLUSTER_NAME"
echo "  Region: $REGION"
echo "  Node Size: $NODE_SIZE"
echo "  Node Count: $NODE_COUNT"
echo ""

read -p "Continue with cluster creation? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Step 1: Create Container Registry
echo -e "${YELLOW}📦 Step 1: Creating Container Registry...${NC}"
if doctl registry get $REGISTRY_NAME &> /dev/null; then
    echo -e "${GREEN}✅ Registry already exists${NC}"
else
    doctl registry create $REGISTRY_NAME --subscription-tier basic
    echo -e "${GREEN}✅ Registry created${NC}"
fi

# Step 2: Create Kubernetes Cluster
echo -e "${YELLOW}☸️  Step 2: Creating Kubernetes Cluster...${NC}"
if doctl kubernetes cluster get $CLUSTER_NAME &> /dev/null; then
    echo -e "${GREEN}✅ Cluster already exists${NC}"
else
    doctl kubernetes cluster create $CLUSTER_NAME \
        --region $REGION \
        --version latest \
        --size $NODE_SIZE \
        --count $NODE_COUNT \
        --auto-upgrade=true \
        --surge-upgrade=true \
        --maintenance-window "saturday=02:00" \
        --tag mivaa,production
    echo -e "${GREEN}✅ Cluster created${NC}"
fi

# Step 3: Get kubeconfig
echo -e "${YELLOW}🔧 Step 3: Configuring kubectl...${NC}"
doctl kubernetes cluster kubeconfig save $CLUSTER_NAME
echo -e "${GREEN}✅ kubectl configured${NC}"

# Step 4: Verify cluster
echo -e "${YELLOW}✓ Step 4: Verifying cluster...${NC}"
kubectl cluster-info
kubectl get nodes
echo -e "${GREEN}✅ Cluster verified${NC}"

# Step 5: Install nginx-ingress controller
echo -e "${YELLOW}🌐 Step 5: Installing nginx-ingress controller...${NC}"
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/do/deploy.yaml
echo -e "${GREEN}✅ nginx-ingress installed${NC}"

# Step 6: Install cert-manager for SSL
echo -e "${YELLOW}🔒 Step 6: Installing cert-manager...${NC}"
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.2/cert-manager.yaml
echo -e "${GREEN}✅ cert-manager installed${NC}"

# Step 7: Wait for cert-manager to be ready
echo -e "${YELLOW}⏳ Waiting for cert-manager to be ready...${NC}"
kubectl wait --for=condition=Available --timeout=300s deployment/cert-manager -n cert-manager
kubectl wait --for=condition=Available --timeout=300s deployment/cert-manager-webhook -n cert-manager
echo -e "${GREEN}✅ cert-manager ready${NC}"

# Step 8: Create Let's Encrypt ClusterIssuer
echo -e "${YELLOW}📜 Step 8: Creating Let's Encrypt ClusterIssuer...${NC}"
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: basiliskan@gmail.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
echo -e "${GREEN}✅ ClusterIssuer created${NC}"

# Step 9: Connect registry to cluster
echo -e "${YELLOW}🔗 Step 9: Connecting registry to cluster...${NC}"
doctl registry kubernetes-manifest | kubectl apply -f -
echo -e "${GREEN}✅ Registry connected${NC}"

echo ""
echo -e "${GREEN}🎉 Cluster setup complete!${NC}"
echo ""
echo -e "${YELLOW}📊 Next steps:${NC}"
echo "1. Run: ./k8s-setup-secrets.sh (to create secrets)"
echo "2. Push code to trigger GitHub Actions deployment"
echo "3. Update DNS: v1api.materialshub.gr -> LoadBalancer IP"
echo ""
echo -e "${YELLOW}📝 Useful commands:${NC}"
echo "  kubectl get nodes"
echo "  kubectl get pods -n default"
echo "  kubectl get svc -n default"
echo "  kubectl logs -f deployment/mivaa-pdf-extractor -n default"

