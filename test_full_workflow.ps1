# PowerShell version of test_full_workflow.sh
# Usage: .\test_full_workflow.ps1 [-TestMode]

param(
    [switch]$TestMode,
    [switch]$Help
)

# Configuration
$API_URL = "http://localhost:8000"
$PDF_URL = "https://bgbavxtjlbvgplozizxu.supabase.co/storage/v1/object/public/pdf-documents/harmony-signature-book-24-25.pdf"
$PDF_NAME = "harmony-signature-book-24-25.pdf"
$MAX_WAIT_TIME = 1800  # 30 minutes
$CHECK_INTERVAL = 10   # 10 seconds

# Show help
if ($Help) {
    Write-Host ""
    Write-Host "MIVAA Full Workflow Test Script (PowerShell)"
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\test_full_workflow.ps1              # Normal mode (all products)"
    Write-Host "  .\test_full_workflow.ps1 -TestMode    # Test mode (first product only)"
    Write-Host ""
    Write-Host "Test Mode:"
    Write-Host "  - Discovers ALL products"
    Write-Host "  - Processes ONLY the first product"
    Write-Host "  - Faster completion (~2-5 minutes)"
    Write-Host ""
    exit 0
}

function Write-Status {
    param([string]$Message)
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ✅ $Message" -ForegroundColor Green
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ❌ $Message" -ForegroundColor Red
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ⚠️  $Message" -ForegroundColor Yellow
}

# Check service health
function Test-ServiceHealth {
    Write-Status "Checking service health..."
    
    try {
        $response = Invoke-RestMethod -Uri "$API_URL/health" -Method Get
        
        if ($response.services.database.status -eq "healthy" -and 
            $response.services.storage.status -eq "healthy" -and 
            $response.services.anthropic.status -eq "healthy") {
            Write-Success "Service is healthy"
            return $true
        } else {
            Write-Error-Custom "Critical services unhealthy"
            return $false
        }
    } catch {
        Write-Error-Custom "Service is not responding: $_"
        return $false
    }
}

# Upload PDF
function Start-PDFUpload {
    if ($TestMode) {
        Write-Warning-Custom "🧪 TEST MODE ENABLED - Will process ONLY the first product"
    }
    
    Write-Status "Uploading PDF from URL: $PDF_URL"
    
    $body = @{
        file_url = $PDF_URL
        title = $PDF_NAME
        processing_mode = "standard"
        categories = "all"
        discovery_model = "claude-vision"
        test_single_product = $TestMode.ToString().ToLower()
    }
    
    try {
        $response = Invoke-RestMethod -Uri "$API_URL/api/rag/documents/upload" -Method Post -Body $body
        
        if ($response.job_id) {
            if ($TestMode) {
                Write-Success "PDF uploaded in TEST MODE. Job ID: $($response.job_id)"
            } else {
                Write-Success "PDF uploaded successfully. Job ID: $($response.job_id)"
            }
            return $response.job_id
        } else {
            Write-Error-Custom "Failed to upload PDF"
            return $null
        }
    } catch {
        Write-Error-Custom "Upload failed: $_"
        return $null
    }
}

# Monitor job
function Watch-Job {
    param([string]$JobId)
    
    Write-Status "Monitoring job: $JobId"
    $startTime = Get-Date
    $lastStatus = ""
    $lastProgress = 0
    
    while ($true) {
        $elapsed = (Get-Date) - $startTime
        
        if ($elapsed.TotalSeconds -gt $MAX_WAIT_TIME) {
            Write-Error-Custom "Timeout waiting for job to complete"
            return $false
        }
        
        try {
            $response = Invoke-RestMethod -Uri "$API_URL/api/rag/documents/job/$JobId" -Method Get
            
            $status = $response.status
            $progress = if ($response.progress) { $response.progress } else { 0 }
            $stage = if ($response.current_stage) { $response.current_stage } else { "unknown" }
            
            if ($status -ne $lastStatus -or $progress -ne $lastProgress) {
                Write-Status "Status: $status | Progress: ${progress}% | Stage: $stage"
                $lastStatus = $status
                $lastProgress = $progress
            }
            
            if ($status -eq "completed") {
                Write-Success "Job completed successfully!"
                return $true
            }
            
            if ($status -eq "failed") {
                Write-Error-Custom "Job failed!"
                if ($response.error_message) {
                    Write-Error-Custom "Error: $($response.error_message)"
                }
                return $false
            }
        } catch {
            Write-Warning-Custom "Failed to get status: $_"
        }
        
        Start-Sleep -Seconds $CHECK_INTERVAL
    }
}

# Main execution
Write-Host ""
Write-Status "========================================="
if ($TestMode) {
    Write-Status "🧪 MIVAA TEST MODE - Single Product"
} else {
    Write-Status "MIVAA Full Workflow Test"
}
Write-Status "========================================="
Write-Host ""

if ($TestMode) {
    Write-Warning-Custom "TEST MODE ENABLED:"
    Write-Warning-Custom "  - Will discover ALL products"
    Write-Warning-Custom "  - Will process ONLY the first product"
    Write-Warning-Custom "  - Use this to validate fixes before full run"
    Write-Warning-Custom "  - Run without -TestMode flag for full processing"
    Write-Host ""
}

# Check service health
if (-not (Test-ServiceHealth)) {
    Write-Error-Custom "Service health check failed. Exiting."
    exit 1
}

Write-Host ""

# Upload PDF
$jobId = Start-PDFUpload
if (-not $jobId) {
    Write-Error-Custom "Upload failed. Exiting."
    exit 1
}

Write-Host ""
Write-Success "Job started: $jobId"
if ($TestMode) {
    Write-Status "🧪 TEST MODE: Processing first product only"
} else {
    Write-Status "Processing all products"
}
Write-Host ""

# Monitor job
if (Watch-Job -JobId $jobId) {
    Write-Host ""
    Write-Success "========================================="
    if ($TestMode) {
        Write-Success "🧪 TEST MODE COMPLETED SUCCESSFULLY!"
        Write-Success "========================================="
        Write-Success "Job ID: $jobId"
        Write-Host ""
        Write-Status "Next steps:"
        Write-Status "  1. ✅ Verify product was created correctly"
        Write-Status "  2. ✅ Check logs for any warnings"
        Write-Status "  3. ✅ If all looks good, run full test:"
        Write-Status "     .\test_full_workflow.ps1"
    } else {
        Write-Success "WORKFLOW TEST COMPLETED SUCCESSFULLY!"
        Write-Success "========================================="
        Write-Success "Job ID: $jobId"
    }
    exit 0
} else {
    Write-Host ""
    Write-Error-Custom "========================================="
    if ($TestMode) {
        Write-Error-Custom "🧪 TEST MODE FAILED"
    } else {
        Write-Error-Custom "WORKFLOW TEST FAILED"
    }
    Write-Error-Custom "========================================="
    Write-Error-Custom "Job ID: $jobId"
    exit 1
}

