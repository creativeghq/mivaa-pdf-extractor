#!/bin/bash

# Test Loop Script - Runs workflow test in a loop until success
# This helps identify and fix issues iteratively

set -e

# Configuration
MAX_ITERATIONS=10
WAIT_BETWEEN_TESTS=10  # seconds

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${CYAN}=========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}=========================================${NC}"
    echo ""
}

print_iteration() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  ITERATION $1 of $MAX_ITERATIONS${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to extract error from logs
extract_error() {
    echo ""
    echo -e "${YELLOW}Extracting error details from logs...${NC}"
    echo ""
    
    # Get recent errors from journalctl
    journalctl -u mivaa-pdf-extractor.service -n 200 --no-pager --since "2 minutes ago" | \
        grep -E "Error|Exception|Traceback|NameError|ModuleNotFoundError|ImportError|AttributeError" | \
        tail -30
    
    echo ""
}

# Function to check if service is running
check_service() {
    if systemctl is-active --quiet mivaa-pdf-extractor.service; then
        return 0
    else
        return 1
    fi
}

# Main loop
main() {
    print_header "MIVAA WORKFLOW TEST LOOP"
    
    echo "This script will:"
    echo "  1. Run the full workflow test"
    echo "  2. If it fails, extract error details"
    echo "  3. Wait for you to fix the issue"
    echo "  4. Repeat until success or max iterations"
    echo ""
    echo "Max iterations: $MAX_ITERATIONS"
    echo "Wait between tests: ${WAIT_BETWEEN_TESTS}s"
    echo ""
    
    for i in $(seq 1 $MAX_ITERATIONS); do
        print_iteration "$i"
        
        # Check if service is running
        if ! check_service; then
            print_error "Service is not running!"
            print_warning "Please start the service and press Enter to continue..."
            read -r
            continue
        fi
        
        # Run the workflow test
        if bash /var/www/mivaa-pdf-extractor/test_full_workflow.sh; then
            print_header "SUCCESS!"
            print_success "Workflow test completed successfully on iteration $i"
            print_success "All steps completed: Upload → Process → Extract → Embed"
            exit 0
        else
            print_error "Workflow test failed on iteration $i"
            
            # Extract error details
            extract_error
            
            # Check if we've reached max iterations
            if [ $i -eq $MAX_ITERATIONS ]; then
                print_header "MAX ITERATIONS REACHED"
                print_error "Failed after $MAX_ITERATIONS attempts"
                exit 1
            fi
            
            # Ask user what to do
            echo ""
            echo -e "${YELLOW}Options:${NC}"
            echo "  1. Fix the issue and press Enter to retry"
            echo "  2. Press Ctrl+C to exit"
            echo ""
            echo -n "Waiting for fix... "
            read -r
            
            # Wait a bit before next iteration
            echo "Waiting ${WAIT_BETWEEN_TESTS}s before next test..."
            sleep $WAIT_BETWEEN_TESTS
        fi
    done
    
    print_header "LOOP COMPLETED"
    print_error "All $MAX_ITERATIONS iterations failed"
    exit 1
}

# Run main
main

