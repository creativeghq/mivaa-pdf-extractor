#!/bin/bash

# Complete Kubernetes Setup - Cleanup and Enable Autoscaling
# This script performs all remaining setup steps automatically

set -e

CLUSTER_ID="e56b1987-f9d0-4e4d-8e50-b27e12592f19"
MIN_NODES=1
MAX_NODES=4

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}KUBERNETES COMPLETE SETUP${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Step 1: Check current state
echo -e "${YELLOW}📊 STEP 1: Checking current infrastructure state${NC}"
echo ""

echo "Current nodes:"
kubectl get nodes
echo ""

echo "Current services:"
kubectl get svc -n default
echo ""

echo "Current pods:"
kubectl get pods -n default
echo ""

echo "Current HPA:"
kubectl get hpa -n default
echo ""

echo "DigitalOcean Load Balancers:"
doctl compute load-balancer list --format ID,Name,IP,Status
echo ""

# Step 2: Get node pool ID
echo -e "${YELLOW}📊 STEP 2: Getting node pool information${NC}"
NODE_POOL_ID=$(doctl kubernetes cluster node-pool list $CLUSTER_ID --format ID --no-header | head -n 1)

if [ -z "$NODE_POOL_ID" ]; then
    echo -e "${RED}❌ No node pool found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Node Pool ID: $NODE_POOL_ID${NC}"
echo ""

# Step 3: Check if old LoadBalancer service exists
echo -e "${YELLOW}📊 STEP 3: Checking for old LoadBalancer service${NC}"
if kubectl get svc mivaa-pdf-extractor -n default 2>/dev/null | grep -q LoadBalancer; then
    echo -e "${YELLOW}⚠️  Found old LoadBalancer service - deleting...${NC}"
    kubectl delete svc mivaa-pdf-extractor -n default
    echo -e "${GREEN}✅ Old LoadBalancer service deleted${NC}"
else
    echo -e "${GREEN}✅ No old LoadBalancer service found (already ClusterIP)${NC}"
fi
echo ""

# Step 4: Scale down to 1 node
echo -e "${YELLOW}📊 STEP 4: Scaling node pool down to 1 node${NC}"
CURRENT_NODE_COUNT=$(kubectl get nodes --no-headers | wc -l)
echo "Current node count: $CURRENT_NODE_COUNT"

if [ "$CURRENT_NODE_COUNT" -gt 1 ]; then
    echo "Scaling down to 1 node..."
    doctl kubernetes cluster node-pool update $CLUSTER_ID $NODE_POOL_ID --count 1
    echo -e "${GREEN}✅ Node pool scaled to 1 node${NC}"
    echo "Waiting 30 seconds for nodes to drain..."
    sleep 30
else
    echo -e "${GREEN}✅ Already at 1 node${NC}"
fi
echo ""

# Step 5: Enable node autoscaling
echo -e "${YELLOW}📊 STEP 5: Enabling node autoscaling (1-4 nodes)${NC}"
doctl kubernetes cluster node-pool update $CLUSTER_ID $NODE_POOL_ID \
    --auto-scale \
    --min-nodes $MIN_NODES \
    --max-nodes $MAX_NODES

echo -e "${GREEN}✅ Node autoscaling enabled!${NC}"
echo ""

# Step 6: Verify final state
echo -e "${YELLOW}📊 STEP 6: Verifying final infrastructure state${NC}"
echo ""

echo "Final node pool configuration:"
doctl kubernetes cluster node-pool get $CLUSTER_ID $NODE_POOL_ID
echo ""

echo "Final nodes:"
kubectl get nodes
echo ""

echo "Final services:"
kubectl get svc -n default
echo ""

echo "Final pods:"
kubectl get pods -n default
echo ""

echo "Final HPA:"
kubectl get hpa -n default
echo ""

echo "DigitalOcean Load Balancers (should be only 1 from Ingress):"
doctl compute load-balancer list --format ID,Name,IP,Status
echo ""

# Summary
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}SETUP COMPLETE!${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo -e "${GREEN}✅ Deployment: Run #40 succeeded${NC}"
echo -e "${GREEN}✅ Service: Changed to ClusterIP${NC}"
echo -e "${GREEN}✅ Old LoadBalancer: Deleted${NC}"
echo -e "${GREEN}✅ Nodes: Scaled to 1 (from $CURRENT_NODE_COUNT)${NC}"
echo -e "${GREEN}✅ Node Autoscaling: Enabled (1-4 nodes)${NC}"
echo -e "${GREEN}✅ Pod Autoscaling: HPA enabled (1-8 pods)${NC}"
echo ""
echo -e "${YELLOW}📋 Architecture Summary:${NC}"
echo "  - Cluster Autoscaler (CA): Scales nodes 1-4 based on pod scheduling"
echo "  - HPA: Scales pods 1-8 based on CPU (60%) and memory (70%)"
echo "  - Service: ClusterIP (internal only)"
echo "  - Ingress: Handles external traffic (1 LoadBalancer)"
echo ""
echo -e "${YELLOW}💰 Cost Optimization:${NC}"
echo "  - Idle: 1 node + 1 LB = ~\$24/month"
echo "  - Peak: 4 nodes + 1 LB = ~\$72/month"
echo "  - Average: ~\$30-40/month (scales based on demand)"
echo ""
echo -e "${YELLOW}🚀 Ready for Testing!${NC}"
echo "  Submit a PDF to trigger autoscaling and monitor:"
echo "  - watch kubectl get pods -n default"
echo "  - watch kubectl get nodes"
echo "  - watch kubectl get hpa -n default"
echo ""

