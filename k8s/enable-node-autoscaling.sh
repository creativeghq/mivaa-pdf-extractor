#!/bin/bash

# Enable Node Autoscaling on existing DigitalOcean Kubernetes cluster
# This script updates the node pool to enable autoscaling (1-3 nodes)

set -e

echo "🔧 Enabling Node Autoscaling on DOKS Cluster"
echo "============================================="
echo ""

# Configuration
CLUSTER_ID="e56b1987-f9d0-4e4d-8e50-b27e12592f19"
MIN_NODES=1
MAX_NODES=4  # Increased to 4 to handle all PDF processing workloads

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if doctl is installed
if ! command -v doctl &> /dev/null; then
    echo -e "${RED}❌ doctl is not installed${NC}"
    echo "Install it from: https://docs.digitalocean.com/reference/doctl/how-to/install/"
    exit 1
fi

echo -e "${YELLOW}📋 Configuration:${NC}"
echo "  Cluster ID: $CLUSTER_ID"
echo "  Min Nodes: $MIN_NODES"
echo "  Max Nodes: $MAX_NODES"
echo ""

# Get cluster info
echo -e "${YELLOW}📊 Current cluster status:${NC}"
doctl kubernetes cluster get $CLUSTER_ID

echo ""
echo -e "${YELLOW}📊 Current node pools:${NC}"
doctl kubernetes cluster node-pool list $CLUSTER_ID

echo ""
read -p "Enable autoscaling on this cluster? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Get the node pool ID (assuming there's only one pool)
NODE_POOL_ID=$(doctl kubernetes cluster node-pool list $CLUSTER_ID --format ID --no-header | head -n 1)

if [ -z "$NODE_POOL_ID" ]; then
    echo -e "${RED}❌ No node pool found${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}🔧 Enabling autoscaling on node pool: $NODE_POOL_ID${NC}"

# Enable autoscaling
doctl kubernetes cluster node-pool update $CLUSTER_ID $NODE_POOL_ID \
    --auto-scale \
    --min-nodes $MIN_NODES \
    --max-nodes $MAX_NODES

echo -e "${GREEN}✅ Autoscaling enabled!${NC}"

echo ""
echo -e "${YELLOW}📊 Updated node pool configuration:${NC}"
doctl kubernetes cluster node-pool get $CLUSTER_ID $NODE_POOL_ID

echo ""
echo -e "${GREEN}🎉 Node autoscaling is now enabled!${NC}"
echo ""
echo -e "${YELLOW}📋 How it works:${NC}"
echo "  - Cluster Autoscaler (CA) monitors pod scheduling"
echo "  - When pods can't be scheduled (pending), CA adds nodes"
echo "  - When nodes are underutilized, CA removes nodes"
echo "  - Min nodes: $MIN_NODES (always running)"
echo "  - Max nodes: $MAX_NODES (scales up to this)"
echo ""
echo -e "${YELLOW}💰 Cost optimization:${NC}"
echo "  - Idle: $MIN_NODES node × \$24/month = \$24/month"
echo "  - Peak: $MAX_NODES nodes × \$24/month = \$72/month"
echo "  - Average: ~\$30-40/month (scales based on demand)"
echo ""
echo -e "${YELLOW}🔗 Works with:${NC}"
echo "  - KEDA ScaledObject (scales pods 0-5 based on job queue)"
echo "  - HPA (scales pods based on CPU/memory)"
echo "  - CA (scales nodes based on pod scheduling)"
echo ""
echo -e "${YELLOW}📝 Monitor autoscaling:${NC}"
echo "  kubectl get nodes -w"
echo "  kubectl get configmap cluster-autoscaler-status -n kube-system -oyaml"
echo "  kubectl describe nodes"

