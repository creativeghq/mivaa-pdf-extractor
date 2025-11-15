# MIVAA PDF Extractor - Production Deployment Guide

## Current Production Architecture

- **Service**: `mivaa-api.service` (systemd)
- **Location**: `/var/www/mivaa-pdf-extractor`
- **Python**: Python 3.9 with pyenv
- **Virtual Environment**: `.venv` directory
- **Web Server**: NGINX reverse proxy to port 8000
- **Process Manager**: systemd (not Docker)
- **Domain**: https://v1api.materialshub.gr

## Deployment Process

### 1. Deploy Code Changes

```bash
# SSH to server
ssh user@v1api.materialshub.gr

# Navigate to application directory
cd /var/www/mivaa-pdf-extractor

# Pull latest code
git pull origin main

# Restart service
sudo systemctl restart mivaa-api

# Check status
sudo systemctl status mivaa-api

# View logs
sudo journalctl -u mivaa-api -f
```

### 2. Service Configuration

The systemd service file is located at: `/etc/systemd/system/mivaa-api.service`

**Environment Variables** are configured directly in the service file:
- SUPABASE_URL
- SUPABASE_ANON_KEY
- SUPABASE_SERVICE_ROLE_KEY
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- TOGETHER_API_KEY
- MATERIAL_KAI_API_KEY
- JWT_SECRET_KEY
- ENCRYPTION_KEY

### 3. Update Service Configuration

If you need to update environment variables:

```bash
# Edit service file
sudo nano /etc/systemd/system/mivaa-api.service

# Reload systemd
sudo systemctl daemon-reload

# Restart service
sudo systemctl restart mivaa-api
```

### 4. NGINX Configuration

NGINX configuration is at: `/etc/nginx/sites-available/default`

```bash
# Test NGINX configuration
sudo nginx -t

# Reload NGINX
sudo systemctl reload nginx

# Restart NGINX
sudo systemctl restart nginx
```

## Useful Commands

```bash
# Service management
sudo systemctl status mivaa-api      # Check service status
sudo systemctl restart mivaa-api     # Restart service
sudo systemctl stop mivaa-api        # Stop service
sudo systemctl start mivaa-api       # Start service

# Logs
sudo journalctl -u mivaa-api -f      # Follow service logs
sudo journalctl -u mivaa-api -n 100  # Last 100 lines
sudo tail -f /var/log/nginx/access.log  # NGINX access logs
sudo tail -f /var/log/nginx/error.log   # NGINX error logs

# Health check
curl http://localhost:8000/health    # Check API health
curl https://v1api.materialshub.gr/health  # Check via NGINX
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs for errors
sudo journalctl -u mivaa-api -n 50

# Common issues:
# 1. Missing API keys - check service file environment variables
# 2. Port 8000 already in use - check with: sudo lsof -i :8000
# 3. Python dependencies - check .venv is activated
```

### NGINX 502 Bad Gateway

```bash
# Check if service is running
sudo systemctl status mivaa-api

# Check if port 8000 is listening
sudo netstat -tlnp | grep 8000

# Check NGINX error logs
sudo tail -f /var/log/nginx/error.log
```

## Database Operations

Supabase RPC functions and database changes are deployed automatically when you:
1. Create/update functions via Supabase API
2. Changes are immediate (no deployment needed)

## Monitoring

- **Sentry**: Error tracking and monitoring
- **Logs**: systemd journal + NGINX logs
- **Health Endpoint**: `/health` endpoint for uptime monitoring

## Security

- **Firewall**: UFW configured (SSH, HTTP, HTTPS)
- **SSL**: Let's Encrypt certificates via Certbot
- **API Keys**: Stored in systemd service file (not in code)
- **CORS**: Configured in FastAPI application

## Support

For issues or questions, check:
1. Service logs: `sudo journalctl -u mivaa-api -f`
2. NGINX logs: `sudo tail -f /var/log/nginx/error.log`
3. Sentry dashboard for production errors

