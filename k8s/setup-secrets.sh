#!/bin/bash
# Kubernetes Secrets Setup Script
# This script creates Kubernetes secrets from environment variables or .env file

set -e  # Exit on error

echo "🔐 MIVAA Kubernetes Secrets Setup"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if kubectl is configured
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}❌ kubectl is not configured${NC}"
    echo "Run: doctl kubernetes cluster kubeconfig save mivaa-k8s-cluster"
    exit 1
fi

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo -e "${YELLOW}📄 Loading environment variables from .env file...${NC}"
    export $(cat .env | grep -v '^#' | xargs)
    echo -e "${GREEN}✅ Environment variables loaded${NC}"
else
    echo -e "${YELLOW}⚠️  No .env file found, using environment variables${NC}"
fi

# Check required environment variables
REQUIRED_VARS=(
    "SUPABASE_ANON_KEY"
    "SUPABASE_SERVICE_ROLE_KEY"
    "ANTHROPIC_API_KEY"
    "OPENAI_API_KEY"
    "JWT_SECRET_KEY"
)

echo -e "${YELLOW}🔍 Checking required environment variables...${NC}"
MISSING_VARS=()
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo -e "${RED}❌ Missing required environment variables:${NC}"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Set them in .env file or export them before running this script"
    exit 1
fi
echo -e "${GREEN}✅ All required variables present${NC}"

# Create namespace if not exists
echo -e "${YELLOW}📦 Creating namespace...${NC}"
kubectl create namespace default --dry-run=client -o yaml | kubectl apply -f -
echo -e "${GREEN}✅ Namespace ready${NC}"

# Create Docker registry secret
echo -e "${YELLOW}🐳 Creating Docker registry secret...${NC}"
if [ -z "$DIGITALOCEAN_ACCESS_TOKEN" ]; then
    echo -e "${YELLOW}⚠️  DIGITALOCEAN_ACCESS_TOKEN not set, skipping registry secret${NC}"
else
    kubectl create secret docker-registry docr-secret \
        --docker-server=registry.digitalocean.com \
        --docker-username=$DIGITALOCEAN_ACCESS_TOKEN \
        --docker-password=$DIGITALOCEAN_ACCESS_TOKEN \
        --namespace=default \
        --dry-run=client -o yaml | kubectl apply -f -
    echo -e "${GREEN}✅ Docker registry secret created${NC}"
fi

# Create application secrets
echo -e "${YELLOW}🔑 Creating application secrets...${NC}"
kubectl create secret generic mivaa-secrets \
    --from-literal=SUPABASE_ANON_KEY="${SUPABASE_ANON_KEY}" \
    --from-literal=SUPABASE_SERVICE_ROLE_KEY="${SUPABASE_SERVICE_ROLE_KEY}" \
    --from-literal=SUPABASE_DB_PASSWORD="${SUPABASE_DB_PASSWORD:-}" \
    --from-literal=SUPABASE_PROJECT_ID="${SUPABASE_PROJECT_ID:-bgbavxtjlbvgplozizxu}" \
    --from-literal=ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
    --from-literal=OPENAI_API_KEY="${OPENAI_API_KEY}" \
    --from-literal=TOGETHER_API_KEY="${TOGETHER_API_KEY:-}" \
    --from-literal=HUGGINGFACE_API_KEY="${HUGGINGFACE_API_KEY:-}" \
    --from-literal=HUGGING_FACE_ACCESS_TOKEN="${HUGGING_FACE_ACCESS_TOKEN:-}" \
    --from-literal=REPLICATE_API_TOKEN="${REPLICATE_API_TOKEN:-}" \
    --from-literal=JINA_API_KEY="${JINA_API_KEY:-}" \
    --from-literal=FIRECRAWL_API_KEY="${FIRECRAWL_API_KEY:-}" \
    --from-literal=JWT_SECRET_KEY="${JWT_SECRET_KEY}" \
    --from-literal=ENCRYPTION_KEY="${ENCRYPTION_KEY:-}" \
    --from-literal=MATERIAL_KAI_API_KEY="${MATERIAL_KAI_API_KEY:-}" \
    --from-literal=MATERIAL_KAI_API_URL="${MATERIAL_KAI_API_URL:-}" \
    --from-literal=MATERIAL_KAI_WORKSPACE_ID="${MATERIAL_KAI_WORKSPACE_ID:-}" \
    --from-literal=MATERIAL_KAI_CLIENT_ID="${MATERIAL_KAI_CLIENT_ID:-}" \
    --from-literal=SENTRY_DSN="https://73f48f6581b882c707ded429e384fb8a@o4509716458045440.ingest.de.sentry.io/4510132019658832" \
    --namespace=default \
    --dry-run=client -o yaml | kubectl apply -f -
echo -e "${GREEN}✅ Application secrets created${NC}"

# Verify secrets
echo -e "${YELLOW}✓ Verifying secrets...${NC}"
kubectl get secrets -n default
echo -e "${GREEN}✅ Secrets verified${NC}"

echo ""
echo -e "${GREEN}🎉 Secrets setup complete!${NC}"
echo ""
echo -e "${YELLOW}📊 Next steps:${NC}"
echo "1. Deploy application: kubectl apply -f k8s-*.yaml"
echo "2. Or push to GitHub to trigger automated deployment"
echo ""
echo -e "${YELLOW}📝 Useful commands:${NC}"
echo "  kubectl get secrets -n default"
echo "  kubectl describe secret mivaa-secrets -n default"

