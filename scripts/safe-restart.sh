#!/bin/bash

# Safe Restart Script for MIVAA PDF Extractor
# Blocks restarts when active jobs are running unless --force flag is used

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
API_URL="http://localhost:8000"
SERVICE_NAME="mivaa-pdf-extractor"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
FORCE=false
REASON=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        --reason)
            REASON="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--force] [--reason \"deployment reason\"]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🔒 MIVAA Safe Restart Protection${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if service is running
if ! systemctl is-active --quiet $SERVICE_NAME; then
    echo -e "${YELLOW}⚠️  Service is not running${NC}"
    echo -e "${GREEN}✅ Safe to start service${NC}"
    sudo systemctl start $SERVICE_NAME
    exit 0
fi

# Check for active jobs via API
echo -e "${BLUE}🔍 Checking for active jobs...${NC}"

ACTIVE_JOBS=$(curl -s "$API_URL/api/jobs/health" | jq -r '.active_jobs // 0' 2>/dev/null || echo "0")

if [ "$ACTIVE_JOBS" = "0" ] || [ -z "$ACTIVE_JOBS" ]; then
    echo -e "${GREEN}✅ No active jobs found${NC}"
    echo -e "${GREEN}✅ Safe to restart service${NC}"
    
    if [ -n "$REASON" ]; then
        echo -e "${BLUE}📝 Restart reason: $REASON${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}🔄 Restarting service...${NC}"
    sudo systemctl restart $SERVICE_NAME
    
    echo -e "${GREEN}✅ Service restarted successfully${NC}"
    sudo systemctl status $SERVICE_NAME --no-pager | head -15
    exit 0
fi

# Active jobs found
echo -e "${RED}❌ RESTART BLOCKED!${NC}"
echo -e "${YELLOW}⚠️  Found $ACTIVE_JOBS active job(s) currently processing${NC}"
echo ""

# Get job details
JOB_DETAILS=$(curl -s "$API_URL/api/jobs/health" | jq -r '.jobs[] | select(.status == "processing") | "  - Job \(.id): \(.filename) (\(.progress)%)"' 2>/dev/null || echo "  - Unable to fetch job details")

echo -e "${YELLOW}Active jobs:${NC}"
echo "$JOB_DETAILS"
echo ""

if [ "$FORCE" = true ]; then
    echo -e "${RED}⚠️  FORCE FLAG DETECTED - Proceeding with restart${NC}"
    echo -e "${RED}⚠️  This will interrupt all active jobs!${NC}"
    
    if [ -n "$REASON" ]; then
        echo -e "${BLUE}📝 Force restart reason: $REASON${NC}"
    fi
    
    echo ""
    read -p "Are you sure you want to force restart? (yes/no): " CONFIRM
    
    if [ "$CONFIRM" = "yes" ]; then
        echo -e "${BLUE}🔄 Force restarting service...${NC}"
        sudo systemctl restart $SERVICE_NAME
        echo -e "${YELLOW}⚠️  Service restarted - jobs were interrupted${NC}"
        sudo systemctl status $SERVICE_NAME --no-pager | head -15
        exit 0
    else
        echo -e "${GREEN}✅ Restart cancelled${NC}"
        exit 1
    fi
fi

echo -e "${BLUE}💡 Options:${NC}"
echo -e "  1. Wait for jobs to complete"
echo -e "  2. Use ${YELLOW}--force${NC} flag to restart anyway (will interrupt jobs)"
echo -e "  3. Cancel jobs via admin panel first"
echo ""
echo -e "${BLUE}Example:${NC}"
echo -e "  ${GREEN}$0 --force --reason \"Critical security patch\"${NC}"
echo ""

exit 1

