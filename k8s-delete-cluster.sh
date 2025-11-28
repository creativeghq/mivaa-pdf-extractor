#!/bin/bash
# Kubernetes Cluster Deletion Script
# WARNING: This will delete the entire cluster and all resources!

set -e  # Exit on error

echo "🗑️  MIVAA Kubernetes Cluster Deletion"
echo "====================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

CLUSTER_NAME="mivaa-k8s-cluster"

echo -e "${RED}⚠️  WARNING: This will DELETE the entire Kubernetes cluster!${NC}"
echo -e "${RED}⚠️  All pods, services, and data will be lost!${NC}"
echo ""
echo "Cluster to delete: $CLUSTER_NAME"
echo ""

read -p "Are you ABSOLUTELY SURE you want to delete the cluster? (type 'DELETE' to confirm) " -r
echo
if [[ ! $REPLY == "DELETE" ]]; then
    echo "Deletion cancelled."
    exit 0
fi

# Delete cluster
echo -e "${YELLOW}🗑️  Deleting cluster...${NC}"
doctl kubernetes cluster delete $CLUSTER_NAME --force

echo ""
echo -e "${GREEN}✅ Cluster deleted${NC}"
echo ""
echo -e "${YELLOW}📝 Note:${NC}"
echo "The Container Registry was NOT deleted."
echo "To delete it manually, run:"
echo "  doctl registry delete mivaa-registry"

