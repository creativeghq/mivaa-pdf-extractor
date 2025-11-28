#!/bin/bash
# Automated DigitalOcean Droplet Resizing for PDF Processing
# 
# Usage:
#   ./droplet-resize.sh up    # Resize to 16GB before processing
#   ./droplet-resize.sh down  # Resize back to 4GB after processing
#
# Requirements:
#   - doctl CLI installed and authenticated
#   - DROPLET_ID environment variable set

set -e

DROPLET_ID="${DROPLET_ID:-}"
SMALL_SIZE="s-2vcpu-4gb"      # 4GB RAM - $24/month
LARGE_SIZE="s-4vcpu-16gb"     # 16GB RAM - $96/month

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if [ -z "$DROPLET_ID" ]; then
    echo -e "${RED}❌ Error: DROPLET_ID environment variable not set${NC}"
    echo "Find your droplet ID with: doctl compute droplet list"
    exit 1
fi

function get_current_size() {
    doctl compute droplet get "$DROPLET_ID" --format Size --no-header
}

function resize_droplet() {
    local target_size=$1
    local action=$2
    
    echo -e "${YELLOW}🔄 Resizing droplet to $target_size...${NC}"
    
    # Resize droplet (disk resize is permanent, so we only resize RAM/CPU)
    doctl compute droplet-action resize "$DROPLET_ID" \
        --size "$target_size" \
        --wait
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Droplet resized successfully to $target_size${NC}"
        
        # Wait for droplet to be fully ready
        echo -e "${YELLOW}⏳ Waiting for droplet to be ready...${NC}"
        sleep 30
        
        # Restart MIVAA service to ensure clean state
        echo -e "${YELLOW}🔄 Restarting MIVAA service...${NC}"
        ssh root@165.227.31.109 "systemctl restart mivaa-pdf-extractor"

        echo -e "${GREEN}✅ MIVAA service restarted${NC}"
    else
        echo -e "${RED}❌ Failed to resize droplet${NC}"
        exit 1
    fi
}

function check_memory() {
    echo -e "${YELLOW}📊 Current memory usage:${NC}"
    ssh root@165.227.31.109 "free -h"
}

case "$1" in
    up)
        current_size=$(get_current_size)
        echo -e "${YELLOW}Current size: $current_size${NC}"
        
        if [ "$current_size" = "$LARGE_SIZE" ]; then
            echo -e "${GREEN}✅ Already at large size ($LARGE_SIZE)${NC}"
            check_memory
            exit 0
        fi
        
        echo -e "${YELLOW}⬆️  Scaling UP for PDF processing${NC}"
        resize_droplet "$LARGE_SIZE" "up"
        check_memory
        
        echo -e "${GREEN}✅ Ready for PDF processing with CLIP models${NC}"
        ;;
        
    down)
        current_size=$(get_current_size)
        echo -e "${YELLOW}Current size: $current_size${NC}"
        
        if [ "$current_size" = "$SMALL_SIZE" ]; then
            echo -e "${GREEN}✅ Already at small size ($SMALL_SIZE)${NC}"
            check_memory
            exit 0
        fi
        
        echo -e "${YELLOW}⬇️  Scaling DOWN to save costs${NC}"
        
        # Check if any PDFs are currently processing
        echo -e "${YELLOW}🔍 Checking for active PDF processing jobs...${NC}"
        active_jobs=$(ssh root@104.248.68.3 "curl -s http://localhost:8000/api/jobs/active | jq -r '.count // 0'")
        
        if [ "$active_jobs" -gt 0 ]; then
            echo -e "${RED}⚠️  Warning: $active_jobs active jobs detected${NC}"
            read -p "Continue with resize? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}❌ Resize cancelled${NC}"
                exit 0
            fi
        fi
        
        resize_droplet "$SMALL_SIZE" "down"
        check_memory
        
        echo -e "${GREEN}✅ Scaled down - saving money! 💰${NC}"
        ;;
        
    status)
        current_size=$(get_current_size)
        echo -e "${GREEN}Current droplet size: $current_size${NC}"
        check_memory
        
        # Show cost estimate
        if [ "$current_size" = "$SMALL_SIZE" ]; then
            echo -e "${GREEN}💰 Current cost: ~$24/month${NC}"
        else
            echo -e "${YELLOW}💰 Current cost: ~$96/month${NC}"
        fi
        ;;
        
    *)
        echo "Usage: $0 {up|down|status}"
        echo ""
        echo "Commands:"
        echo "  up     - Resize to 16GB for PDF processing"
        echo "  down   - Resize to 4GB to save costs"
        echo "  status - Show current size and memory usage"
        exit 1
        ;;
esac

