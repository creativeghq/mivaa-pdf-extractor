#!/bin/bash

# Cleanup unnecessary Kubernetes infrastructure
# This script removes extra nodes and old LoadBalancers

set -e

echo "🧹 Cleaning up Kubernetes infrastructure..."
echo "============================================="
echo ""

# Configuration
CLUSTER_ID="e56b1987-f9d0-4e4d-8e50-b27e12592f19"

# Step 1: Check current cluster status
echo "📊 Current cluster status:"
echo "-------------------------"
doctl kubernetes cluster get $CLUSTER_ID --format ID,Name,Region,Status,NodePools
echo ""

# Step 2: List all nodes
echo "📋 Current nodes:"
echo "----------------"
kubectl get nodes -o wide
echo ""

# Step 3: List all services (to find LoadBalancers)
echo "🔍 Current services:"
echo "-------------------"
kubectl get svc -A
echo ""

# Step 4: Delete old LoadBalancer service if it exists
echo "🗑️  Checking for old LoadBalancer service..."
if kubectl get svc mivaa-pdf-extractor -n default -o jsonpath='{.spec.type}' 2>/dev/null | grep -q "LoadBalancer"; then
    echo "   Found old LoadBalancer service - deleting..."
    LB_IP=$(kubectl get svc mivaa-pdf-extractor -n default -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "N/A")
    echo "   LoadBalancer IP: $LB_IP"
    kubectl delete svc mivaa-pdf-extractor -n default
    echo "   ✅ Old LoadBalancer deleted"
else
    echo "   ✅ No old LoadBalancer found (already ClusterIP or doesn't exist)"
fi
echo ""

# Step 5: Check for duplicate LoadBalancers in DigitalOcean
echo "🔍 Checking DigitalOcean LoadBalancers..."
doctl compute load-balancer list --format ID,Name,IP,Status
echo ""

# Step 6: Scale down node pool to 1 node (if currently at 2+)
echo "📉 Scaling node pool to 1 node..."
NODE_POOL_ID=$(doctl kubernetes cluster node-pool list $CLUSTER_ID --format ID --no-header | head -n 1)
CURRENT_NODE_COUNT=$(doctl kubernetes cluster node-pool get $CLUSTER_ID $NODE_POOL_ID --format Count --no-header)

echo "   Current node count: $CURRENT_NODE_COUNT"
if [ "$CURRENT_NODE_COUNT" -gt 1 ]; then
    echo "   Scaling down to 1 node..."
    doctl kubernetes cluster node-pool update $CLUSTER_ID $NODE_POOL_ID --count 1
    echo "   ✅ Node pool scaled to 1 node"
    echo "   ⏳ Waiting for nodes to drain and terminate..."
    sleep 30
else
    echo "   ✅ Already at 1 node"
fi
echo ""

# Step 7: Verify final state
echo "✅ Cleanup complete! Final state:"
echo "================================="
echo ""
echo "Nodes:"
kubectl get nodes
echo ""
echo "Services:"
kubectl get svc -A
echo ""
echo "Pods:"
kubectl get pods -n default
echo ""
echo "DigitalOcean LoadBalancers:"
doctl compute load-balancer list --format ID,Name,IP,Status
echo ""
echo "✅ Infrastructure cleanup complete!"
echo "   - Node pool scaled to 1 node"
echo "   - Old LoadBalancer removed (if existed)"
echo "   - Only Ingress LoadBalancer should remain"

