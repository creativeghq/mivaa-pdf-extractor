#!/bin/bash
# Kubernetes Rollback Script
# This script rolls back the MIVAA deployment to the previous version

set -e  # Exit on error

echo "⏪ MIVAA Kubernetes Rollback"
echo "============================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

DEPLOYMENT_NAME="mivaa-pdf-extractor"
NAMESPACE="default"

# Check if kubectl is configured
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}❌ kubectl is not configured${NC}"
    exit 1
fi

# Show current deployment status
echo -e "${YELLOW}📊 Current deployment status:${NC}"
kubectl get deployment $DEPLOYMENT_NAME -n $NAMESPACE
echo ""

# Show rollout history
echo -e "${YELLOW}📜 Rollout history:${NC}"
kubectl rollout history deployment/$DEPLOYMENT_NAME -n $NAMESPACE
echo ""

# Confirm rollback
read -p "Do you want to rollback to the previous version? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Rollback cancelled."
    exit 0
fi

# Perform rollback
echo -e "${YELLOW}⏪ Rolling back deployment...${NC}"
kubectl rollout undo deployment/$DEPLOYMENT_NAME -n $NAMESPACE

# Wait for rollback to complete
echo -e "${YELLOW}⏳ Waiting for rollback to complete...${NC}"
kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=5m

# Show new status
echo ""
echo -e "${GREEN}✅ Rollback complete!${NC}"
echo ""
echo -e "${YELLOW}📊 New deployment status:${NC}"
kubectl get deployment $DEPLOYMENT_NAME -n $NAMESPACE
echo ""
kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT_NAME
echo ""

# Run health check
echo -e "${YELLOW}🏥 Running health check...${NC}"
EXTERNAL_IP=$(kubectl get svc $DEPLOYMENT_NAME -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -n "$EXTERNAL_IP" ]; then
    echo "Testing: http://$EXTERNAL_IP/health"
    curl -f http://$EXTERNAL_IP/health && echo -e "${GREEN}✅ Health check passed${NC}" || echo -e "${RED}❌ Health check failed${NC}"
else
    echo -e "${YELLOW}⚠️  LoadBalancer IP not available${NC}"
fi

echo ""
echo -e "${YELLOW}📝 Useful commands:${NC}"
echo "  kubectl logs -f deployment/$DEPLOYMENT_NAME -n $NAMESPACE"
echo "  kubectl describe deployment $DEPLOYMENT_NAME -n $NAMESPACE"
echo "  kubectl rollout history deployment/$DEPLOYMENT_NAME -n $NAMESPACE"

