# 📋 MIVAA Deployment Overview

## 🚀 Overview

The MIVAA PDF Extractor now includes comprehensive deployment overviews in all GitHub Actions workflows, providing complete visibility into the deployment process, system architecture, and operational status.

## 🎯 Deployment Strategy

### 🔄 Default Deployment (Automatic)
- **Workflow**: `deploy.yml` - "MIVAA Deployment (Default)"
- **Triggers**: Automatic on push to `main` or `production` branches
- **Use Case**: Standard deployments for regular code updates
- **Features**: Fast, reliable deployment with comprehensive overview
- **Manual Trigger**: Available via workflow_dispatch for manual deployments

### 🚀 Orchestrated Deployment (On-Demand)
- **Workflow**: `orchestrated-deployment.yml` - "Orchestrated MIVAA Deployment Pipeline (On-Demand)"
- **Triggers**: Manual only via workflow_dispatch
- **Use Case**: Advanced deployments requiring detailed analysis and validation
- **Features**: Multi-phase pipeline with intelligence, validation, and comprehensive reporting
- **Options**: Deployment mode, diagnostics control, target branch selection

## 🎯 Features

### 📊 Pre-Deployment Overview
- **Deployment Configuration**: Environment, target server, branch, commit details
- **Application Architecture**: Runtime, dependencies, process management
- **Key Components**: PDF processing, AI/ML models, database, authentication
- **Environment Variables**: Verification of all required secrets and configurations
- **Deployment Process**: Step-by-step breakdown of the deployment pipeline
- **Expected Outcomes**: Clear expectations for deployment results
- **📋 Summary Table**: All key information displayed on the main action page

### 📈 Post-Deployment Summary
- **Deployment Status**: Success/failure with timestamps and duration
- **Service Information**: Server details, service status, process management
- **🏥 Health Check Results**: Real-time endpoint testing with HTTP status codes
- **API Endpoints**: Complete list of available endpoints with health check URLs
- **Verification Checklist**: Confirmation of all deployment steps
- **🔧 Automatic Diagnostics**: Comprehensive system analysis when health checks fail
- **🔄 Auto-Recovery**: Automatic service restart and re-testing on failures
- **Troubleshooting Guide**: Quick access to logs, status commands, and common fixes
- **Next Steps**: Recommended actions after deployment
- **📋 Results Table**: Deployment results and service status on the main action page

### 🎯 GitHub Action Summary
**NEW FEATURES**: All deployment details are now displayed on the main GitHub Action page as organized tables, including:
- **Deployment Overview Table**: Configuration, branch, commit, trigger information
- **Application Architecture Table**: Service details, runtime, components
- **🏥 Health Check Results Table**: Real-time endpoint testing with HTTP status codes
- **Deployment Results Table**: Status, completion time, service health
- **Service Endpoints Table**: Direct links to all API endpoints with status indicators
- **🔧 Automatic Diagnostics**: Comprehensive system analysis when issues are detected
- **🔄 Auto-Recovery Status**: Automatic restart attempts and recovery verification
- **Quick Actions**: One-click access to health checks, documentation, and troubleshooting
- **Troubleshooting Commands**: Copy-paste ready commands for debugging

## 🔧 Implementation

## 🔧 How to Use Each Deployment Type

### 🔄 Using Default Deployment
**Automatic Triggers:**
- Push code to `main` or `production` branch
- Deployment starts automatically

**Manual Trigger:**
1. Go to GitHub Actions tab
2. Select "MIVAA Deployment (Default)"
3. Click "Run workflow"
4. Optionally provide deployment reason
5. Click "Run workflow" button

### 🚀 Using Orchestrated Deployment
**Manual Trigger Only:**
1. Go to GitHub Actions tab
2. Select "Orchestrated MIVAA Deployment Pipeline (On-Demand)"
3. Click "Run workflow"
4. Configure options:
   - **Deployment Mode**: fast-track, intelligent, comprehensive
   - **Skip Diagnostics**: Enable for faster deployment
   - **Target Branch**: main or production
   - **Deployment Reason**: Describe why you're using orchestrated deployment
5. Click "Run workflow" button

## 🔧 Implementation Details

### Standard Deployment Workflow (`deploy.yml`)

The standard deployment workflow includes:

1. **📋 Deployment Overview** - Comprehensive pre-deployment information
2. **🚀 Deploy to server** - Actual deployment execution
3. **📊 Deployment Summary & Health Check** - Post-deployment verification

### Orchestrated Deployment Workflow (`orchestrated-deployment.yml`)

The orchestrated workflow provides enhanced features:

1. **📋 Orchestrated Deployment Overview** - Phase-aware deployment information
2. **🚀 Execute deployment** - Multi-strategy deployment execution
3. **📊 Comprehensive Journey Summary** - Complete pipeline overview with metrics

## 📋 Information Provided

### 🏗️ Application Architecture
```
Service: MIVAA PDF Extractor (FastAPI)
Runtime: Python 3.9 with pyenv
Package Manager: uv (ultrafast Python package installer)
Process Manager: systemd (mivaa-pdf-extractor.service)
Deployment Path: /var/www/mivaa-pdf-extractor
```

### 🔧 Key Components
- **PDF Processing**: PyMuPDF + pymupdf4llm
- **AI/ML**: LlamaIndex, OpenAI, HuggingFace, Replicate
- **Database**: Supabase (PostgreSQL)
- **Authentication**: JWT with custom middleware
- **Monitoring**: Sentry error tracking
- **API Endpoints**: 37+ endpoints across 7 modules

### 🔐 Environment Variables Verification
All critical environment variables are verified as configured:
- SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY
- ANTHROPIC_API_KEY, OPENAI_API_KEY, HUGGINGFACE_API_KEY
- REPLICATE_API_TOKEN, JWT_SECRET_KEY, SENTRY_DSN
- MATERIAL_KAI_API_* (URL, KEY, CLIENT_ID, CLIENT_SECRET, etc.)

### 📦 Deployment Process Steps
1. 📥 Code checkout and SSH setup
2. 🔄 Git repository sync on server
3. 🐍 Python environment setup (pyenv + venv)
4. 📋 Dependency compilation (uv.lock generation)
5. ⚡ Ultra-fast dependency installation (uv)
6. ✅ Critical dependency verification
7. 🔄 Service restart (systemd)
8. 🏥 Health check and status verification

## 🌐 API Endpoints Information

### Available Endpoints
- **Health Check**: `https://v1api.materialshub.gr/health`
- **API Documentation**: `https://v1api.materialshub.gr/docs`
- **OpenAPI Schema**: `https://v1api.materialshub.gr/openapi.json`
- **PDF Processing**: `https://v1api.materialshub.gr/api/v1/pdf/*`
- **AI Analysis**: `https://v1api.materialshub.gr/api/v1/ai/*`
- **Vector Search**: `https://v1api.materialshub.gr/api/v1/search/*`

### Quick Health Check Commands
```bash
curl https://v1api.materialshub.gr/health
curl https://v1api.materialshub.gr/docs
```

## 🔧 Troubleshooting Information

### SSH Access
```bash
ssh root@104.248.68.3
```

### Service Management
```bash
# View logs
sudo journalctl -u mivaa-pdf-extractor -f

# Check status
sudo systemctl status mivaa-pdf-extractor

# Restart service
sudo systemctl restart mivaa-pdf-extractor

# Navigate to app directory
cd /var/www/mivaa-pdf-extractor
```

## 🏥 Health Check & Diagnostics Features

### 🔍 Real-Time Health Monitoring
- **HTTP Status Code Verification**: Tests actual endpoint responses (200, 502, 404, etc.)
- **Multiple Endpoint Testing**: Health, docs, redoc, and OpenAPI schema endpoints
- **Response Time Monitoring**: Tracks endpoint response times and availability
- **Retry Logic**: Multiple attempts with intelligent backoff

### 🔧 Automatic Diagnostics
When health checks fail, the system automatically:
- **System Resource Analysis**: CPU, memory, disk usage, and load averages
- **Service Status Investigation**: systemd service status and configuration
- **Network Diagnostics**: Port availability, process binding, and network interfaces
- **Application Log Analysis**: Recent logs, error logs, and service history
- **Environment Verification**: Python environment, dependencies, and application structure

### 🔄 Auto-Recovery Features
- **Automatic Service Restart**: Attempts to restart failed services
- **Post-Restart Verification**: Re-tests endpoints after recovery attempts
- **Recovery Status Reporting**: Clear indication of recovery success or failure
- **Escalation Path**: Provides manual intervention steps when auto-recovery fails

### 📊 Comprehensive Reporting
- **Real-Time Status Updates**: Live updates during health check execution
- **Detailed Error Information**: HTTP status codes, response content, and error messages
- **Diagnostic Summaries**: Organized presentation of system analysis results
- **Recovery Documentation**: Complete audit trail of recovery attempts

## 📈 Benefits

### 🎯 Enhanced Visibility
- Complete transparency into deployment process
- Real-time status updates and progress tracking
- **Live health monitoring** with actual HTTP status codes
- Clear expectations and outcomes

### 🚀 Improved Reliability
- Pre-deployment verification of all requirements
- **Real-time post-deployment health verification**
- **Automatic issue detection and recovery**
- Comprehensive troubleshooting information

### 🔧 Better Debugging
- Detailed environment and configuration information
- **Automatic diagnostic collection** when issues occur
- **Live system analysis** with resource monitoring
- Quick access to logs and status commands
- Clear next steps for issue resolution

### 📊 Professional Presentation
- Clean, organized deployment information
- **Real-time health status indicators**
- **Comprehensive diagnostic reports**
- Consistent formatting across all workflows
- Easy-to-read status indicators and summaries

## 🎉 Usage

The deployment overview is automatically included in all MIVAA deployment workflows. No additional configuration is required - simply trigger a deployment and view the comprehensive overview in the GitHub Actions logs.

### Viewing the Overview
1. Navigate to the GitHub repository
2. Go to the "Actions" tab
3. Select a deployment workflow run
4. View the deployment overview in the workflow logs
5. Check the workflow summary for additional details (orchestrated workflow only)

## 🔄 Future Enhancements

- Integration with monitoring dashboards
- Slack/Discord notifications with deployment summaries
- Automated performance benchmarking
- Integration with external monitoring tools
- Custom deployment metrics and analytics
