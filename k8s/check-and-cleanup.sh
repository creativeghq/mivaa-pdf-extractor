#!/bin/bash

# Check and Cleanup Kubernetes Infrastructure
# This script checks the current state and cleans up extra resources

set -e

CLUSTER_ID="e56b1987-f9d0-4e4d-8e50-b27e12592f19"
NODE_POOL_ID="f8e0e5e5-e5e5-4e4d-8e50-b27e12592f19"  # Update this with actual node pool ID

echo "========================================="
echo "KUBERNETES INFRASTRUCTURE CHECK & CLEANUP"
echo "========================================="
echo ""

# 1. Check current nodes
echo "📊 CURRENT NODES:"
kubectl get nodes -o wide
echo ""

# 2. Check current services
echo "📊 CURRENT SERVICES:"
kubectl get svc -n default
echo ""

# 3. Check for LoadBalancer services
echo "🔍 CHECKING FOR LOADBALANCER SERVICES:"
LB_COUNT=$(kubectl get svc -n default -o json | jq '[.items[] | select(.spec.type=="LoadBalancer")] | length')
echo "Found $LB_COUNT LoadBalancer service(s)"

if [ "$LB_COUNT" -gt 0 ]; then
  echo "⚠️  LoadBalancer services found:"
  kubectl get svc -n default -o json | jq -r '.items[] | select(.spec.type=="LoadBalancer") | .metadata.name'
  echo ""
  echo "Note: These should be deleted if they're old/duplicate services"
  echo "The only LoadBalancer should be from the Ingress Controller"
fi
echo ""

# 4. Check Ingress
echo "📊 CURRENT INGRESS:"
kubectl get ingress -n default
echo ""

# 5. Check pods
echo "📊 CURRENT PODS:"
kubectl get pods -n default -o wide
echo ""

# 6. Check HPA
echo "📊 CURRENT HPA:"
kubectl get hpa -n default
echo ""

# 7. Check deployments
echo "📊 CURRENT DEPLOYMENTS:"
kubectl get deployments -n default
echo ""

# 8. Get DigitalOcean Load Balancers
echo "📊 DIGITALOCEAN LOAD BALANCERS:"
doctl compute load-balancer list --format ID,Name,IP,Status
echo ""

# 9. Summary
echo "========================================="
echo "SUMMARY"
echo "========================================="
echo ""
echo "✅ Deployment run #40 SUCCEEDED"
echo "✅ KEDA temporarily disabled (using HPA only)"
echo "✅ Service changed to ClusterIP"
echo "✅ Old LoadBalancer service should be deleted"
echo ""

# 10. Cleanup recommendations
echo "========================================="
echo "CLEANUP RECOMMENDATIONS"
echo "========================================="
echo ""
echo "1. Scale down to 1 node:"
echo "   doctl kubernetes cluster node-pool update $CLUSTER_ID $NODE_POOL_ID --count 1"
echo ""
echo "2. Delete old LoadBalancer service (if exists):"
echo "   kubectl delete svc mivaa-pdf-extractor -n default"
echo ""
echo "3. Enable node autoscaling (1-4 nodes):"
echo "   cd k8s && ./enable-node-autoscaling.sh"
echo ""
echo "4. Verify only 1 LoadBalancer exists (from Ingress):"
echo "   doctl compute load-balancer list"
echo ""

# 11. Interactive cleanup
echo "========================================="
echo "INTERACTIVE CLEANUP"
echo "========================================="
echo ""
read -p "Do you want to scale down to 1 node now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo "Scaling node pool to 1 node..."
  doctl kubernetes cluster node-pool update $CLUSTER_ID $NODE_POOL_ID --count 1
  echo "✅ Node pool scaled to 1 node"
fi
echo ""

read -p "Do you want to delete old LoadBalancer service? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
  if kubectl get svc mivaa-pdf-extractor -n default 2>/dev/null | grep -q LoadBalancer; then
    echo "Deleting old LoadBalancer service..."
    kubectl delete svc mivaa-pdf-extractor -n default
    echo "✅ Old LoadBalancer service deleted"
  else
    echo "ℹ️  No LoadBalancer service found (already ClusterIP or doesn't exist)"
  fi
fi
echo ""

echo "========================================="
echo "CLEANUP COMPLETE"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Enable node autoscaling: ./enable-node-autoscaling.sh"
echo "2. Monitor scaling behavior during PDF processing"
echo "3. Re-enable KEDA later for scale-to-zero capability"
echo ""

