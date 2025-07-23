# SSL Deployment Guide for MIVAA PDF Extractor

This guide provides comprehensive instructions for deploying the MIVAA PDF Extractor with automated SSL certificate management using Let's Encrypt.

## Overview

The deployment includes:
- Automated server setup with security hardening
- Let's Encrypt SSL certificate installation and renewal
- NGINX configuration with SSL termination
- Health monitoring and automatic restart capabilities
- Backup automation and log rotation

## Prerequisites

1. **Domain Name**: A registered domain pointing to your server's IP address
2. **Server Access**: SSH access to an Ubuntu/Debian server
3. **GitHub Secrets**: Required secrets configured in your repository

## Required GitHub Secrets

Configure the following secrets in your GitHub repository (`Settings > Secrets and variables > Actions`):

### Deployment Secrets
- `DEPLOY_HOST`: Your server's IP address or hostname
- `DEPLOY_USER`: SSH username for deployment
- `DEPLOY_SSH_KEY`: Private SSH key for server access

### SSL Configuration
- `DOMAIN_NAME`: Your domain name (e.g., `api.yourdomain.com`)
- `ADMIN_EMAIL`: Email address for Let's Encrypt notifications

### Application Secrets
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_ANON_KEY`: Your Supabase anonymous key
- `OPENAI_API_KEY`: Your OpenAI API key
- `SENTRY_DSN`: Your Sentry DSN for error tracking
- `MATERIAL_KAI_API_URL`: Material Kai Vision Platform API URL
- `MATERIAL_KAI_API_KEY`: Material Kai Vision Platform API key

## Deployment Process

### 1. Automatic Deployment

The deployment is fully automated through GitHub Actions. When you push to the `main` branch:

1. **Tests Run**: All tests are executed to ensure code quality
2. **Docker Image Built**: Application is containerized and pushed to GHCR
3. **Server Setup**: First-time server configuration (if needed)
4. **SSL Setup**: Automatic SSL certificate installation
5. **Application Deployment**: Latest version deployed with zero downtime

### 2. First-Time Setup

On the first deployment, the system automatically:

- Updates the server and installs required packages
- Installs Docker and Docker Compose
- Configures NGINX with security headers
- Sets up UFW firewall with proper rules
- Installs Fail2Ban for intrusion prevention
- Creates application directories with proper permissions
- Configures log rotation and backup automation

### 3. SSL Certificate Management

The SSL automation includes:

- **Automatic Installation**: Certificates are obtained from Let's Encrypt
- **Domain Validation**: Automatic verification of domain ownership
- **NGINX Configuration**: SSL-enabled virtual host setup
- **Automatic Renewal**: Certificates renew automatically before expiration
- **Health Monitoring**: SSL certificate expiration monitoring

## Manual SSL Setup (if needed)

If you need to manually configure SSL:

```bash
# SSH into your server
ssh your-user@your-server

# Run the SSL setup script
sudo /usr/local/bin/setup-ssl.sh your-domain.com your-email@example.com
```

## SSL Certificate Renewal

Certificates automatically renew through systemd timers:

- **Check Schedule**: Every 12 hours
- **Renewal Threshold**: 30 days before expiration
- **Automatic Restart**: NGINX reloads after renewal
- **Monitoring**: Health checks verify SSL status

### Manual Renewal Check

```bash
# Check renewal status
sudo /usr/local/bin/check-ssl-renewal.sh

# Force renewal (if needed)
sudo certbot renew --force-renewal
sudo systemctl reload nginx
```

## Monitoring and Health Checks

### Application Health

The deployment includes comprehensive health monitoring:

```bash
# Check application status
curl -f https://your-domain.com/health

# View application logs
docker-compose logs -f mivaa-pdf-extractor

# Check system status
systemctl status nginx
systemctl status docker
```

### SSL Health Monitoring

```bash
# Check SSL certificate status
openssl s_client -connect your-domain.com:443 -servername your-domain.com

# View SSL renewal logs
journalctl -u ssl-renewal-check.timer
journalctl -u ssl-renewal-check.service
```

### System Monitoring

```bash
# Check firewall status
sudo ufw status

# View security logs
sudo journalctl -u fail2ban

# Check disk usage
df -h
```

## Security Features

### Firewall Configuration
- SSH (port 22): Allowed
- HTTP (port 80): Allowed (redirects to HTTPS)
- HTTPS (port 443): Allowed
- All other ports: Denied by default

### Fail2Ban Protection
- SSH brute force protection
- NGINX rate limiting
- Automatic IP blocking for suspicious activity

### SSL Security
- TLS 1.2+ only
- Strong cipher suites
- HSTS headers
- Perfect Forward Secrecy

### NGINX Security Headers
```nginx
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header Referrer-Policy "strict-origin-when-cross-origin";
```

## Backup and Maintenance

### Automated Backups
- **Application Data**: Daily backups of uploads and logs
- **SSL Certificates**: Automatic backup before renewal
- **Configuration Files**: Version-controlled deployment configs

### Log Rotation
- **Application Logs**: Rotated daily, kept for 30 days
- **NGINX Logs**: Rotated weekly, compressed
- **System Logs**: Standard logrotate configuration

### Maintenance Tasks
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Clean Docker images
docker system prune -f

# Check log sizes
du -sh /var/log/*

# Restart services if needed
sudo systemctl restart nginx
docker-compose restart
```

## Troubleshooting

### Common Issues

#### SSL Certificate Issues
```bash
# Check certificate status
sudo certbot certificates

# Test renewal
sudo certbot renew --dry-run

# Check NGINX configuration
sudo nginx -t
```

#### Application Issues
```bash
# Check container status
docker-compose ps

# View detailed logs
docker-compose logs --tail=100 mivaa-pdf-extractor

# Restart application
docker-compose restart mivaa-pdf-extractor
```

#### Network Issues
```bash
# Check port availability
sudo netstat -tlnp | grep :443
sudo netstat -tlnp | grep :80

# Test domain resolution
nslookup your-domain.com
dig your-domain.com
```

### Emergency Procedures

#### SSL Certificate Emergency Renewal
```bash
# Stop NGINX temporarily
sudo systemctl stop nginx

# Force certificate renewal
sudo certbot renew --standalone --force-renewal

# Start NGINX
sudo systemctl start nginx
```

#### Application Recovery
```bash
# Pull latest image
docker pull ghcr.io/your-repo/mivaa-pdf-extractor:latest

# Restart with latest image
docker-compose down
docker-compose up -d
```

## Performance Optimization

### NGINX Optimization
- Gzip compression enabled
- Static file caching
- Connection keep-alive
- Worker process optimization

### Docker Optimization
- Multi-stage builds for smaller images
- Health checks for container monitoring
- Resource limits for stability

### SSL Optimization
- OCSP stapling enabled
- Session resumption
- Optimized cipher suites

## Support and Maintenance

### Regular Maintenance Schedule
- **Daily**: Automated backups and log rotation
- **Weekly**: Security updates check
- **Monthly**: SSL certificate status review
- **Quarterly**: Full system security audit

### Monitoring Alerts
Set up monitoring for:
- SSL certificate expiration (30 days warning)
- Application health check failures
- High resource usage
- Security incidents

### Contact Information
For deployment issues or questions:
- Check GitHub Actions logs for deployment status
- Review application logs for runtime issues
- Monitor system logs for security events

## Conclusion

This automated SSL deployment provides a production-ready, secure, and maintainable infrastructure for the MIVAA PDF Extractor. The system is designed to be self-healing and requires minimal manual intervention while providing comprehensive monitoring and security features.