# 🚀 MIVAA Deployment Quick Reference

## 📋 Deployment Options Overview

### 🔄 Default Deployment (Recommended)
- **File**: `.github/workflows/deploy.yml`
- **Name**: "MIVAA Deployment (Default)"
- **Triggers**: 
  - ✅ **Automatic**: Push to `main` or `production`
  - ✅ **Manual**: workflow_dispatch
- **Best For**: Regular deployments, code updates, standard releases

### 🚀 Orchestrated Deployment (Advanced)
- **File**: `.github/workflows/orchestrated-deployment.yml`
- **Name**: "Orchestrated MIVAA Deployment Pipeline (On-Demand)"
- **Triggers**: 
  - ❌ **Automatic**: Disabled
  - ✅ **Manual**: workflow_dispatch only
- **Best For**: Complex deployments, validation required, detailed analysis

---

## 🎯 When to Use Each Option

### Use Default Deployment When:
- ✅ Regular code updates and bug fixes
- ✅ Feature releases that have been tested
- ✅ Quick deployments needed
- ✅ Standard deployment process is sufficient
- ✅ Automatic deployment on code push is desired

### Use Orchestrated Deployment When:
- 🔍 Major releases requiring validation
- 🔍 Deployments with potential risks
- 🔍 Need detailed analysis and reporting
- 🔍 Multiple deployment strategies to choose from
- 🔍 Comprehensive testing and validation required

---

## 🚀 How to Deploy

### 🔄 Default Deployment

**Automatic (Recommended):**
```bash
git push origin main
# Deployment starts automatically
```

**Manual:**
1. Go to GitHub → Actions
2. Select "MIVAA Deployment (Default)"
3. Click "Run workflow"
4. Optional: Add deployment reason
5. Click "Run workflow"

### 🚀 Orchestrated Deployment

**Manual Only:**
1. Go to GitHub → Actions
2. Select "Orchestrated MIVAA Deployment Pipeline (On-Demand)"
3. Click "Run workflow"
4. Configure options:
   - **Deployment Mode**: 
     - `fast-track` - Quick deployment (~15s)
     - `intelligent` - Smart deployment (~30s)
     - `comprehensive` - Full validation (~60s)
   - **Skip Diagnostics**: `true` for faster deployment
   - **Target Branch**: `main` or `production`
   - **Deployment Reason**: Describe why using orchestrated
5. Click "Run workflow"

---

## 📊 Feature Comparison

| Feature | Default Deployment | Orchestrated Deployment |
|---------|-------------------|-------------------------|
| **Automatic Trigger** | ✅ Yes | ❌ No (On-demand only) |
| **Manual Trigger** | ✅ Yes | ✅ Yes |
| **Deployment Speed** | 🟢 Fast (2-3 min) | 🟡 Variable (15s-60s) |
| **Validation** | 🟡 Basic | 🟢 Comprehensive |
| **Analysis** | 🟡 Standard | 🟢 Advanced |
| **Reporting** | 🟢 Good + Summary Tables | 🟢 Excellent + Summary Tables |
| **Summary Page** | ✅ Yes | ✅ Yes |
| **Configuration** | 🟡 Limited | 🟢 Extensive |
| **Risk Assessment** | ❌ No | ✅ Yes |
| **Multi-Phase** | ❌ No | ✅ Yes |
| **Strategy Selection** | ❌ No | ✅ Yes |

---

## 🔧 Configuration Options

### Default Deployment Options
- **deployment_reason**: Optional reason for manual deployment

### Orchestrated Deployment Options
- **deployment_mode**: 
  - `fast-track` - Minimal validation, fastest deployment
  - `intelligent` - Smart analysis and optimization
  - `comprehensive` - Full validation and testing
- **skip_diagnostics**: Skip validation phase for speed
- **target_branch**: Choose deployment branch
- **deployment_reason**: Detailed reason for orchestrated deployment

---

## 📈 Deployment Process

### Default Deployment Steps
1. 📋 **Deployment Overview** - System info and configuration
2. 🚀 **Deploy to Server** - Code deployment and service restart
3. 📊 **Deployment Summary** - Health check and verification

### Orchestrated Deployment Phases
1. 🧠 **Phase 1: Analysis** - Code analysis and strategy selection
2. 🔧 **Phase 2: Dependencies** - Dependency resolution and optimization
3. 🔬 **Phase 3: Validation** - System validation and diagnostics
4. 🚀 **Phase 4: Deployment** - Actual deployment execution
5. 🔍 **Phase 5: Verification** - Comprehensive health checks

---

## 🌐 Service Endpoints (Both Deployments)

After successful deployment, these endpoints are available:

- **Health Check**: http://104.248.68.3:8000/health
- **API Documentation**: http://104.248.68.3:8000/docs
- **ReDoc**: http://104.248.68.3:8000/redoc
- **OpenAPI Schema**: http://104.248.68.3:8000/openapi.json
- **PDF Processing**: http://104.248.68.3:8000/api/v1/pdf/*
- **AI Analysis**: http://104.248.68.3:8000/api/v1/ai/*
- **Vector Search**: http://104.248.68.3:8000/api/v1/search/*

---

## 🔧 Troubleshooting

### Quick Health Check
```bash
curl http://104.248.68.3:8000/health
```

### SSH Access
```bash
ssh root@104.248.68.3
```

### Service Management
```bash
sudo systemctl status mivaa-pdf-extractor
sudo systemctl restart mivaa-pdf-extractor
sudo journalctl -u mivaa-pdf-extractor -f
```

---

## 💡 Recommendations

### For Most Users:
- Use **Default Deployment** for 90% of deployments
- It's automatic, fast, and reliable
- Perfect for regular development workflow

### For Advanced Users:
- Use **Orchestrated Deployment** for:
  - Major releases
  - High-risk deployments
  - When detailed analysis is needed
  - Production deployments requiring validation

### Best Practices:
1. Let default deployment handle regular pushes automatically
2. Use orchestrated deployment for planned releases
3. **Check the Action Summary page** for organized deployment details
4. Always review deployment overview for system status
5. Monitor health endpoints after deployment
6. Keep deployment reasons descriptive for audit trail
7. **Use the summary tables** for quick status verification

---

🚀 **Quick Start**: Just push to main branch for automatic deployment!
🔧 **Advanced**: Use GitHub Actions → "Orchestrated MIVAA Deployment Pipeline (On-Demand)" for complex deployments.
📋 **NEW**: Check the Action Summary page for organized deployment details in easy-to-read tables!
