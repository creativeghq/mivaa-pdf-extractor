# MIVAA PDF Extractor - Deployment Guide

This guide covers deploying the MIVAA PDF Extractor to Digital Ocean using Docker and GitHub Actions with **fully automated deployment**.

## Prerequisites

- Digital Ocean Droplet (Ubuntu 24.10 or later, minimum 2GB RAM recommended)
- Domain name (optional, for SSL)
- GitHub repository with Actions enabled

## Fully Automated Deployment

The deployment is now **completely automated** through GitHub Actions. No manual server setup is required!

### Step 1: Create Digital Ocean Droplet

1. Create a new Ubuntu 24.10 droplet on Digital Ocean
2. Add your SSH key during creation
3. Note the droplet's IP address

### Step 2: Configure GitHub Secrets

In your GitHub repository, go to Settings → Secrets and variables → Actions, and add these secrets:

#### Required Secrets
```
DEPLOY_HOST=your.server.ip.address
DEPLOY_USER=root
DEPLOY_SSH_KEY=your-private-ssh-key

# SSL Configuration (Optional)
DOMAIN_NAME=your-domain.com
ADMIN_EMAIL=your-email@example.com

# Application Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-supabase-anon-key
OPENAI_API_KEY=your-openai-api-key
SENTRY_DSN=your-sentry-dsn
MATERIAL_KAI_API_URL=https://your-material-kai-api.com
MATERIAL_KAI_API_KEY=your-material-kai-api-key
```

#### Container Registry
The deployment uses **GitHub Container Registry (GHCR)** instead of Docker Hub for better integration and security. No additional secrets are needed as it uses the built-in `GITHUB_TOKEN` automatically.

**Benefits of GHCR:**
- ✅ Better GitHub integration
- ✅ Improved security with GitHub's access controls
- ✅ No Docker Hub rate limits
- ✅ Automatic cleanup policies
- ✅ Free for public repositories

#### SSH Key Setup
Generate an SSH key pair for deployment:
```bash
ssh-keygen -t rsa -b 4096 -C "github-actions@yourdomain.com"
```

- Add the **public key** to your Digital Ocean droplet's `~/.ssh/authorized_keys`
- Add the **private key** as the `DEPLOY_SSH_KEY` secret in GitHub

### Step 3: Deploy

Simply push to the `main` branch or create a pull request. GitHub Actions will:

1. **Automatically detect** if this is the first deployment
2. **Setup the server** with all required dependencies (Docker, NGINX, firewall, etc.)
3. **Build and deploy** your application
4. **Configure NGINX** reverse proxy with security headers
5. **Verify deployment** with health checks

The entire process takes about 5-10 minutes for the first deployment, and 2-3 minutes for subsequent deployments.

## What Gets Automatically Configured

### Server Setup (First Deployment Only)
- ✅ System updates and security patches
- ✅ Docker and Docker Compose installation
- ✅ NGINX installation and configuration
- ✅ Firewall setup (UFW) with proper ports
- ✅ Application directory structure
- ✅ Repository cloning and permissions

### Every Deployment
- ✅ Latest code deployment
- ✅ Docker image building and deployment
- ✅ Environment variable configuration
- ✅ Service health checks
- ✅ Automatic rollback on failure
- ✅ Old image cleanup

## Monitoring Deployment

### GitHub Actions
Monitor your deployment in the GitHub Actions tab:
- Build logs show compilation progress
- Deploy logs show server setup and deployment status
- Health check results confirm successful deployment

### Server Access
SSH into your server to monitor:
```bash
ssh root@your.server.ip.address

# Check application status
cd /opt/mivaa-pdf-extractor
docker-compose ps

# View application logs
docker-compose logs -f

# Check NGINX status
sudo systemctl status nginx
```

## Health Checks

The application provides several health check endpoints:

- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed system status
- `GET /metrics` - Performance metrics
- `GET /performance/summary` - Performance summary

## SSL Configuration (Optional)

### Using Let's Encrypt with Certbot

After deployment, you can add SSL:

```bash
# SSH into your server
ssh root@your.server.ip.address

# Install Certbot
apt install certbot python3-certbot-nginx -y

# Obtain SSL certificate (replace with your domain)
certbot --nginx -d yourdomain.com

# Auto-renewal is set up automatically
```

## Log Management

Logs are stored in the following locations:

- Application logs: `/opt/mivaa-pdf-extractor/logs/`
- NGINX logs: `/var/log/nginx/`
- Docker logs: `docker-compose logs`
- Sentry: Real-time error tracking and monitoring

## Troubleshooting

### Common Issues

1. **Deployment fails**: Check GitHub Actions logs for detailed error messages
2. **Health check fails**: Verify all environment variables are set correctly
3. **Permission issues**: Ensure SSH key has proper access to the server
4. **Port conflicts**: Ensure ports 80 and 443 are available

### Debugging Commands

```bash
# Check application status
docker-compose ps
docker-compose logs mivaa-pdf-extractor

# Check NGINX status
systemctl status nginx
nginx -t

# Check disk space
df -h

# Check memory usage
free -h

# Check running processes
htop
```

### Re-running Deployment

If deployment fails, you can:

1. **Fix the issue** and push again (triggers automatic re-deployment)
2. **Manually trigger** the workflow from GitHub Actions tab
3. **SSH into server** and check logs for specific issues

## Security Features

The automated deployment includes:

1. **Firewall**: UFW configured to only allow SSH, HTTP, and HTTPS
2. **Rate limiting**: NGINX configured with rate limiting
3. **Security headers**: NGINX adds security headers
4. **Secrets management**: All sensitive data stored as GitHub secrets
5. **Container isolation**: Application runs in Docker container

## Performance Features

- **NGINX reverse proxy** with caching and compression
- **Health monitoring** with automatic restart on failure
- **Resource monitoring** via performance endpoints
- **Error tracking** with Sentry integration
- **Optimized Docker** multi-stage builds

## Architecture Overview

```
Internet → NGINX (Port 80/443) → Docker Container (Port 8000) → FastAPI Application
                                                               ↓
                                                           Supabase Database
                                                               ↓
                                                           External APIs
```

The deployment uses:
- **NGINX**: Reverse proxy, SSL termination, rate limiting
- **Docker**: Containerization and isolation
- **GitHub Actions**: CI/CD pipeline with automated server setup
- **Digital Ocean**: Cloud hosting
- **Supabase**: Database and authentication
- **Sentry**: Error tracking and monitoring

## Support

For issues and questions:

1. Check the GitHub Actions logs first
2. Review application logs on the server
3. Check this documentation
4. Create an issue in the GitHub repository

## Next Steps

After successful deployment:

1. **Configure SSL** if using a custom domain - see [SSL Deployment Guide](SSL_DEPLOYMENT_GUIDE.md)
2. **Set up monitoring** alerts in Sentry
3. **Configure backups** for uploaded files
4. **Test all endpoints** to ensure proper functionality
5. **Monitor performance** using the `/metrics` endpoint

## Additional Documentation

- **[SSL Deployment Guide](SSL_DEPLOYMENT_GUIDE.md)** - Comprehensive guide for automated SSL certificate management with Let's Encrypt
- **[GitHub Container Registry Setup](README.md#container-registry)** - Details on GHCR integration and benefits