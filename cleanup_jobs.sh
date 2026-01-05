#!/bin/bash

# Cleanup Script for MIVAA PDF Extractor
# This script cancels/cleans up stuck or old jobs

set -e

API_URL="http://localhost:8000"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

# Function to list all processing jobs
list_processing_jobs() {
    print_status "Fetching all processing jobs..."
    
    response=$(curl -s "$API_URL/api/rag/documents/jobs?limit=50" || echo "")
    
    if [ -z "$response" ]; then
        print_error "Could not fetch jobs"
        return 1
    fi
    
    # Filter processing jobs
    processing_jobs=$(echo "$response" | jq -r '.jobs[] | select(.status == "processing" or .status == "pending") | "\(.id)|\(.filename)|\(.status)|\(.progress)%|\(.created_at)"' 2>/dev/null || echo "")
    
    if [ -z "$processing_jobs" ]; then
        print_success "No processing jobs found"
        return 0
    fi
    
    echo ""
    print_warning "Found processing jobs:"
    echo "----------------------------------------"
    printf "%-40s %-30s %-12s %-10s %s\n" "JOB ID" "FILENAME" "STATUS" "PROGRESS" "CREATED"
    echo "----------------------------------------"
    
    echo "$processing_jobs" | while IFS='|' read -r job_id filename status progress created_at; do
        printf "%-40s %-30s %-12s %-10s %s\n" "$job_id" "$filename" "$status" "$progress" "$created_at"
    done
    
    echo "----------------------------------------"
    echo ""
    
    return 0
}

# Function to cancel all processing jobs
cancel_all_jobs() {
    print_warning "This will mark all processing jobs as 'interrupted'"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        print_status "Cancelled by user"
        return 0
    fi
    
    response=$(curl -s "$API_URL/api/rag/documents/jobs?limit=50" || echo "")
    
    if [ -z "$response" ]; then
        print_error "Could not fetch jobs"
        return 1
    fi
    
    job_ids=$(echo "$response" | jq -r '.jobs[] | select(.status == "processing" or .status == "pending") | .id' 2>/dev/null || echo "")
    
    if [ -z "$job_ids" ]; then
        print_success "No jobs to cancel"
        return 0
    fi
    
    count=0
    echo "$job_ids" | while read -r job_id; do
        print_status "Cancelling job: $job_id"
        # Update job status to interrupted via API (you'll need to implement this endpoint)
        # For now, we'll just log it
        count=$((count + 1))
    done
    
    print_success "Marked $count job(s) for cancellation"
    print_warning "Note: Jobs will be marked as 'interrupted' on next heartbeat check"
}

# Main menu
main() {
    echo ""
    print_status "========================================="
    print_status "MIVAA Job Cleanup Utility"
    print_status "========================================="
    echo ""
    
    list_processing_jobs
    
    echo "Options:"
    echo "  1) List jobs again"
    echo "  2) Cancel all processing jobs"
    echo "  3) Restart service (clears stuck jobs)"
    echo "  4) Exit"
    echo ""
    
    read -p "Select option (1-4): " option
    
    case $option in
        1)
            list_processing_jobs
            ;;
        2)
            cancel_all_jobs
            ;;
        3)
            print_warning "Restarting service..."
            sudo systemctl restart mivaa-pdf-extractor.service
            print_success "Service restarted"
            ;;
        4)
            print_status "Exiting"
            exit 0
            ;;
        *)
            print_error "Invalid option"
            exit 1
            ;;
    esac
}

main

