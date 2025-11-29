#!/bin/bash

# Cleanup script to remove old LoadBalancer service
# This script removes the duplicate LoadBalancer that was created before switching to Ingress-only

set -e

echo "🧹 Cleaning up old LoadBalancer configuration..."

# Check if old LoadBalancer service exists
if kubectl get svc mivaa-pdf-extractor -n default -o jsonpath='{.spec.type}' 2>/dev/null | grep -q "LoadBalancer"; then
    echo "⚠️  Found old LoadBalancer service - deleting..."
    
    # Get the LoadBalancer IP before deletion (for reference)
    LB_IP=$(kubectl get svc mivaa-pdf-extractor -n default -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "N/A")
    echo "   Old LoadBalancer IP: $LB_IP"
    
    # Delete the service (this will delete the DigitalOcean LoadBalancer)
    kubectl delete svc mivaa-pdf-extractor -n default
    
    echo "✅ Old LoadBalancer service deleted"
    echo "   DigitalOcean LoadBalancer will be automatically removed in ~1 minute"
    
    # Wait a moment for the deletion to propagate
    sleep 5
    
    # Recreate the service as ClusterIP
    echo "📦 Recreating service as ClusterIP..."
    kubectl apply -f k8s/service.yaml
    
    echo "✅ Service recreated as ClusterIP"
else
    echo "✅ Service is already ClusterIP - no cleanup needed"
fi

# Verify the new configuration
echo ""
echo "📊 Current configuration:"
echo "   Service type: $(kubectl get svc mivaa-pdf-extractor -n default -o jsonpath='{.spec.type}')"
echo "   Cluster IP: $(kubectl get svc mivaa-pdf-extractor -n default -o jsonpath='{.spec.clusterIP}')"
echo ""

# Check Ingress status
echo "📊 Ingress status:"
kubectl get ingress mivaa-pdf-extractor-ingress -n default

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📋 Summary:"
echo "   - Old LoadBalancer deleted (if existed)"
echo "   - Service now uses ClusterIP (internal only)"
echo "   - External access via Ingress: https://v1api.materialshub.gr"
echo "   - Only 1 DigitalOcean LoadBalancer (from Ingress Controller)"
echo ""
echo "💰 Cost savings: ~$10/month (removed duplicate LoadBalancer)"

