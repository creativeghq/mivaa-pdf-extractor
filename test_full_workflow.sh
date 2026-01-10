#!/bin/bash

# Full Workflow Test Script for MIVAA PDF Extractor
# This script tests the complete upload and processing pipeline
#
# Features:
# - Prevents duplicate job uploads
# - Cleans up stuck/old jobs before starting
# - Comprehensive monitoring and verification

set -e

# Configuration
API_URL="http://localhost:8000"
PDF_URL="https://bgbavxtjlbvgplozizxu.supabase.co/storage/v1/object/public/pdf-documents/harmony-signature-book-24-25.pdf"
PDF_NAME="harmony-signature-book-24-25.pdf"
MAX_WAIT_TIME=1800  # 30 minutes max wait (product-centric pipeline takes longer)
CHECK_INTERVAL=10   # Check every 10 seconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to check for existing jobs processing the same PDF
check_existing_jobs() {
    print_status "Checking for existing jobs processing $PDF_NAME..."

    response=$(curl -s "$API_URL/api/rag/documents/jobs?limit=20" || echo "")

    if [ -z "$response" ]; then
        print_warning "Could not check existing jobs"
        return 0
    fi

    # Check for processing jobs with same filename
    existing_jobs=$(echo "$response" | jq -r ".jobs[] | select(.filename == \"$PDF_NAME\" and (.status == \"processing\" or .status == \"pending\")) | .id" 2>/dev/null || echo "")

    if [ -n "$existing_jobs" ]; then
        print_error "Found existing job(s) processing $PDF_NAME:"
        echo "$existing_jobs" | while read -r job_id; do
            echo "  - Job ID: $job_id"
        done
        print_error "Please wait for existing jobs to complete or cancel them first"
        return 1
    fi

    print_success "No existing jobs found for $PDF_NAME"
    return 0
}

# Function to check service health
check_health() {
    print_status "Checking service health..."
    response=$(curl -s "$API_URL/health" || echo "")

    if [ -z "$response" ]; then
        print_error "Service is not responding"
        return 1
    fi

    # Check critical services only (database, storage, anthropic)
    db_status=$(echo "$response" | jq -r '.services.database.status' 2>/dev/null || echo "")
    storage_status=$(echo "$response" | jq -r '.services.storage.status' 2>/dev/null || echo "")
    anthropic_status=$(echo "$response" | jq -r '.services.anthropic.status' 2>/dev/null || echo "")

    if [ "$db_status" = "healthy" ] && [ "$storage_status" = "healthy" ] && [ "$anthropic_status" = "healthy" ]; then
        print_success "Service is healthy (critical services: database, storage, anthropic)"
        print_warning "Note: Qwen endpoint may be unhealthy but not required for this test (using Claude)"
        return 0
    else
        print_error "Critical services unhealthy - DB: $db_status, Storage: $storage_status, Anthropic: $anthropic_status"
        return 1
    fi
}

# Function to upload PDF from URL
upload_pdf() {
    print_status "Uploading PDF from URL: $PDF_URL"

    response=$(curl -s -X POST "$API_URL/api/rag/documents/upload" \
        -F "file_url=$PDF_URL" \
        -F "title=$PDF_NAME" \
        -F "processing_mode=standard" \
        -F "categories=all" \
        -F "discovery_model=claude-vision" \
        2>&1)

    echo "$response" > /tmp/upload_response.json

    job_id=$(echo "$response" | jq -r '.job_id' 2>/dev/null || echo "")

    if [ -z "$job_id" ] || [ "$job_id" = "null" ]; then
        print_error "Failed to upload PDF"
        echo "$response" | jq '.' 2>/dev/null || echo "$response"
        return 1
    fi

    print_success "PDF uploaded successfully. Job ID: $job_id"
    echo "$job_id"
}

# Function to check job status
check_job_status() {
    local job_id=$1

    response=$(curl -s "$API_URL/api/rag/documents/job/$job_id" 2>&1)
    echo "$response"
}

# Function to get job details with error info
get_job_details() {
    local job_id=$1

    response=$(curl -s "$API_URL/api/rag/documents/job/$job_id" 2>&1)
    echo "$response" > "/tmp/job_${job_id}_status.json"
    echo "$response"
}

# Function to monitor job progress
monitor_job() {
    local job_id=$1
    local start_time=$(date +%s)
    local last_status=""
    local last_progress=0
    
    print_status "Monitoring job: $job_id"
    
    while true; do
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        
        if [ $elapsed -gt $MAX_WAIT_TIME ]; then
            print_error "Timeout waiting for job to complete (${MAX_WAIT_TIME}s)"
            return 1
        fi
        
        response=$(check_job_status "$job_id")
        status=$(echo "$response" | jq -r '.status' 2>/dev/null || echo "")
        progress=$(echo "$response" | jq -r '.progress // 0' 2>/dev/null || echo "0")
        current_stage=$(echo "$response" | jq -r '.current_stage // "unknown"' 2>/dev/null || echo "unknown")
        
        # Print progress if changed
        if [ "$status" != "$last_status" ] || [ "$progress" != "$last_progress" ]; then
            print_status "Status: $status | Progress: ${progress}% | Stage: $current_stage"
            last_status="$status"
            last_progress="$progress"
        fi

        # Check for completion
        if [ "$status" = "completed" ]; then
            print_success "Job completed successfully!"
            echo "$response" > "/tmp/job_${job_id}_final.json"
            return 0
        fi

        # Check for failure
        if [ "$status" = "failed" ]; then
            print_error "Job failed!"
            error_message=$(echo "$response" | jq -r '.error_message // "Unknown error"' 2>/dev/null || echo "Unknown error")
            print_error "Error: $error_message"
            echo "$response" > "/tmp/job_${job_id}_error.json"

            # Extract detailed error from logs
            print_status "Fetching detailed error logs..."
            journalctl -u mivaa-pdf-extractor.service -n 100 --no-pager | grep -A 5 -B 5 "Error\|Exception\|Traceback" | tail -50

            return 1
        fi

        sleep $CHECK_INTERVAL
    done
}

# Function to verify results
verify_results() {
    local job_id=$1

    print_status "Verifying job results..."

    response=$(get_job_details "$job_id")

    # Check for chunks
    chunks_count=$(echo "$response" | jq -r '.result.chunks_created // 0' 2>/dev/null || echo "0")
    images_count=$(echo "$response" | jq -r '.result.images_extracted // 0' 2>/dev/null || echo "0")
    clip_count=$(echo "$response" | jq -r '.result.clip_embeddings_generated // 0' 2>/dev/null || echo "0")
    products_count=$(echo "$response" | jq -r '.result.products_discovered // 0' 2>/dev/null || echo "0")

    print_status "Results Summary:"
    echo "  - Chunks created: $chunks_count"
    echo "  - Images extracted: $images_count"
    echo "  - CLIP embeddings: $clip_count"
    echo "  - Products discovered: $products_count"

    # Verify all steps completed
    success=true

    if [ "$chunks_count" -eq 0 ]; then
        print_warning "No chunks created"
        success=false
    else
        print_success "Chunks created: $chunks_count"
    fi

    if [ "$images_count" -eq 0 ]; then
        print_warning "No images extracted"
        success=false
    else
        print_success "Images extracted: $images_count"
    fi

    if [ "$clip_count" -eq 0 ]; then
        print_warning "No CLIP embeddings generated"
        success=false
    else
        print_success "CLIP embeddings generated: $clip_count"
    fi

    if [ "$products_count" -eq 0 ]; then
        print_warning "No products discovered"
        success=false
    else
        print_success "Products discovered: $products_count"
    fi

    if [ "$success" = true ]; then
        print_success "All verification checks passed!"
        return 0
    else
        print_error "Some verification checks failed"
        return 1
    fi
}

# Main execution
main() {
    echo ""
    print_status "========================================="
    print_status "MIVAA Full Workflow Test"
    print_status "========================================="
    echo ""

    # Check service health
    if ! check_health; then
        print_error "Service health check failed. Exiting."
        exit 1
    fi

    echo ""

    # Check for existing jobs (PREVENT DUPLICATES)
    if ! check_existing_jobs; then
        print_error "Existing job detected. Exiting to prevent duplicates."
        print_warning "To force a new upload, cancel existing jobs first or wait for completion."
        exit 1
    fi

    echo ""

    # Upload PDF (ONLY ONCE)
    print_status "Starting NEW upload (no duplicates)..."
    job_id=$(upload_pdf)
    if [ -z "$job_id" ]; then
        print_error "Upload failed. Exiting."
        exit 1
    fi

    echo ""
    print_success "Job started: $job_id"
    print_status "This is the ONLY job running for $PDF_NAME"
    echo ""

    # Monitor job
    if monitor_job "$job_id"; then
        echo ""
        # Verify results
        if verify_results "$job_id"; then
            print_success "========================================="
            print_success "WORKFLOW TEST COMPLETED SUCCESSFULLY!"
            print_success "========================================="
            print_success "Job ID: $job_id"
            exit 0
        else
            print_error "========================================="
            print_error "WORKFLOW TEST FAILED - VERIFICATION"
            print_error "========================================="
            print_error "Job ID: $job_id"
            exit 1
        fi
    else
        print_error "========================================="
        print_error "WORKFLOW TEST FAILED - JOB PROCESSING"
        print_error "========================================="
        print_error "Job ID: $job_id"
        exit 1
    fi
}

# Run main function
main

