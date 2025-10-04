#!/bin/bash

# MIVAA PDF Extractor - Automated Server Setup Script
# This script sets up a complete production environment with SSL automation

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}





info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Function to check nginx status
check_nginx_status() {
    log "🔍 Checking nginx status..."

    if systemctl is-active --quiet nginx; then
        log "✅ nginx is running"
        return 0
    else
        error "❌ nginx is not running"
        return 1
    fi
}

# Function to check nginx configuration
check_nginx_config() {
    log "🔧 Checking nginx configuration..."

    if nginx -t 2>/dev/null; then
        log "✅ nginx configuration is valid"
        return 0
    else
        error "❌ nginx configuration has errors:"
        nginx -t
        return 1
    fi
}

# Function to restart nginx safely
restart_nginx() {
    log "🔄 Restarting nginx..."

    # First check configuration
    if ! check_nginx_config; then
        error "Cannot restart nginx due to configuration errors"
        return 1
    fi

    # Restart nginx
    if systemctl restart nginx; then
        log "✅ nginx restarted successfully"

        # Wait a moment and verify it's running
        sleep 2
        if check_nginx_status; then
            log "✅ nginx is running after restart"
            return 0
        else
            error "❌ nginx failed to start after restart"
            return 1
        fi
    else
        error "❌ Failed to restart nginx"
        return 1
    fi
}

# Function to check nginx and report status for GitHub Actions
check_nginx_for_github() {
    log "🔍 Checking nginx status for GitHub Actions..."

    # Check if nginx is installed
    if ! command -v nginx &> /dev/null; then
        error "nginx is not installed"
        echo "::error::nginx is not installed on the server"
        return 1
    fi

    # Check nginx configuration
    if ! check_nginx_config; then
        error "nginx configuration is invalid"
        echo "::error::nginx configuration has errors"
        nginx -t 2>&1 | while read line; do
            echo "::error::$line"
        done
        return 1
    fi

    # Check nginx status
    if ! check_nginx_status; then
        error "nginx is not running"
        echo "::error::nginx service is not running"

        # Try to get more details
        local status_output=$(systemctl status nginx 2>&1 || echo "Failed to get status")
        echo "::error::nginx status: $status_output"

        return 1
    fi

    # Check if nginx is responding to HTTP requests
    if curl -f -s http://localhost > /dev/null 2>&1; then
        log "✅ nginx is responding to HTTP requests"
        echo "::notice::nginx is healthy and responding to requests"
        return 0
    else
        error "nginx is running but not responding to HTTP requests"
        echo "::warning::nginx is running but not responding to HTTP requests"
        return 1
    fi
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error "This script should not be run as root for security reasons"
   exit 1
fi

# Check if sudo is available
if ! command -v sudo &> /dev/null; then
    error "sudo is required but not installed"
    exit 1
fi

log "Starting MIVAA PDF Extractor server setup..."

# Update system packages
log "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
log "Installing essential packages..."
sudo apt install -y \
    curl \
    wget \
    git \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    ufw \
    fail2ban \
    htop \
    nano \
    vim

# Install Docker
log "Installing Docker..."
if ! command -v docker &> /dev/null; then
    # Add Docker's official GPG key
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    # Add Docker repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker Engine
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # Add current user to docker group
    sudo usermod -aG docker $USER
    
    # Enable and start Docker
    sudo systemctl enable docker
    sudo systemctl start docker
    
    log "Docker installed successfully"
else
    log "Docker is already installed"
fi

# Install Docker Compose (standalone)
log "Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_VERSION="v2.24.1"
    sudo curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    log "Docker Compose installed successfully"
else
    log "Docker Compose is already installed"
fi

# Install NGINX
log "Installing NGINX..."
if ! command -v nginx &> /dev/null; then
    sudo apt install -y nginx
    sudo systemctl enable nginx
    sudo systemctl start nginx
    log "NGINX installed successfully"
else
    log "NGINX is already installed"
fi

# Install Certbot for Let's Encrypt
log "Installing Certbot for Let's Encrypt SSL..."
if ! command -v certbot &> /dev/null; then
    sudo apt install -y certbot python3-certbot-nginx
    log "Certbot installed successfully"
else
    log "Certbot is already installed"
fi

# Configure UFW Firewall
log "Configuring UFW firewall..."
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (be careful with this)
sudo ufw allow ssh
sudo ufw allow 22/tcp

# Allow HTTP and HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Enable firewall
sudo ufw --force enable
log "Firewall configured successfully"

# Configure Fail2Ban
log "Configuring Fail2Ban..."
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Create fail2ban configuration for nginx
sudo tee /etc/fail2ban/jail.local > /dev/null <<EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true

[nginx-http-auth]
enabled = true

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
action = iptables-multiport[name=ReqLimit, port="http,https", protocol=tcp]
logpath = /var/log/nginx/*error.log
findtime = 600
bantime = 7200
maxretry = 10
EOF

sudo systemctl restart fail2ban
log "Fail2Ban configured successfully"

# Create application directory
APP_DIR="/opt/mivaa-pdf-extractor"
log "Creating application directory: $APP_DIR"
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR

# Create logs directory
sudo mkdir -p $APP_DIR/logs
sudo chown $USER:$USER $APP_DIR/logs

# Create NGINX configuration
log "Creating NGINX configuration..."
sudo tee /etc/nginx/sites-available/mivaa-pdf-extractor > /dev/null <<'EOF'
# MIVAA PDF Extractor NGINX Configuration
server {
    listen 80;
    server_name _;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=2r/s;
    
    # File upload size
    client_max_body_size 100M;
    
    # Proxy settings
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 300s;
    proxy_connect_timeout 75s;
    
    # Main application
    location / {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://127.0.0.1:8000;
    }
    
    # Upload endpoints with stricter rate limiting
    location ~ ^/(upload|extract|process) {
        limit_req zone=upload burst=5 nodelay;
        proxy_pass http://127.0.0.1:8000;
    }
    
    # Health check endpoint (no rate limiting)
    location /health {
        proxy_pass http://127.0.0.1:8000;
        access_log off;
    }
    
    # Static files (if any)
    location /static/ {
        alias /opt/mivaa-pdf-extractor/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Deny access to sensitive files
    location ~ /\. {
        deny all;
        access_log off;
        log_not_found off;
    }
    
    # Logging
    access_log /var/log/nginx/mivaa-pdf-extractor.access.log;
    error_log /var/log/nginx/mivaa-pdf-extractor.error.log;
}
EOF

# Enable the site
sudo ln -sf /etc/nginx/sites-available/mivaa-pdf-extractor /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test NGINX configuration
sudo nginx -t
sudo systemctl reload nginx

log "NGINX configuration created successfully"

# Create SSL setup script
log "Creating SSL automation script..."
sudo tee /usr/local/bin/setup-ssl.sh > /dev/null <<'EOF'
#!/bin/bash

# SSL Setup Script for MIVAA PDF Extractor
# This script sets up Let's Encrypt SSL certificates

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check if domain is provided
if [ -z "$1" ]; then
    error "Usage: $0 <domain> [email]"
    error "Example: $0 yourdomain.com admin@yourdomain.com"
    exit 1
fi

DOMAIN="$1"
EMAIL="${2:-admin@${DOMAIN}}"

log "Setting up SSL for domain: $DOMAIN"
log "Using email: $EMAIL"

# Validate domain format
if [[ ! "$DOMAIN" =~ ^[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}$ ]]; then
    error "Invalid domain format: $DOMAIN"
    exit 1
fi

# Check if domain resolves to this server
SERVER_IP=$(curl -s ifconfig.me || curl -s ipinfo.io/ip || echo "unknown")
DOMAIN_IP=$(dig +short "$DOMAIN" | tail -n1)

if [ "$SERVER_IP" != "$DOMAIN_IP" ]; then
    warn "Domain $DOMAIN does not resolve to this server IP ($SERVER_IP)"
    warn "Domain resolves to: $DOMAIN_IP"
    warn "SSL setup may fail. Please ensure DNS is properly configured."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update NGINX configuration with the domain
log "Updating NGINX configuration with domain..."
sudo sed -i "s/server_name _;/server_name $DOMAIN;/" /etc/nginx/sites-available/mivaa-pdf-extractor

# Test NGINX configuration
sudo nginx -t

# Reload NGINX
sudo systemctl reload nginx

# Obtain SSL certificate
log "Obtaining SSL certificate from Let's Encrypt..."
sudo certbot --nginx \
    --non-interactive \
    --agree-tos \
    --email "$EMAIL" \
    --domains "$DOMAIN" \
    --redirect

# Verify certificate
log "Verifying SSL certificate..."
if sudo certbot certificates | grep -q "$DOMAIN"; then
    log "SSL certificate successfully installed for $DOMAIN"
else
    error "SSL certificate installation failed"
    exit 1
fi

# Test automatic renewal
log "Testing automatic renewal..."
sudo certbot renew --dry-run

log "SSL setup completed successfully!"
log "Your site is now available at: https://$DOMAIN"

# Display certificate information
log "Certificate information:"
sudo certbot certificates | grep -A 10 "$DOMAIN"
EOF

sudo chmod +x /usr/local/bin/setup-ssl.sh

# Create SSL renewal check script
log "Creating SSL renewal monitoring script..."
sudo tee /usr/local/bin/check-ssl-renewal.sh > /dev/null <<'EOF'
#!/bin/bash

# SSL Renewal Check Script
# This script checks SSL certificate expiration and sends alerts

set -e

LOG_FILE="/var/log/ssl-renewal-check.log"
ALERT_DAYS=30  # Alert when certificate expires in 30 days or less

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check all certificates
for cert_path in /etc/letsencrypt/live/*/cert.pem; do
    if [ -f "$cert_path" ]; then
        domain=$(basename $(dirname "$cert_path"))
        
        # Get certificate expiration date
        exp_date=$(openssl x509 -in "$cert_path" -noout -enddate | cut -d= -f2)
        exp_epoch=$(date -d "$exp_date" +%s)
        current_epoch=$(date +%s)
        days_until_exp=$(( (exp_epoch - current_epoch) / 86400 ))
        
        log "Certificate for $domain expires in $days_until_exp days"
        
        if [ $days_until_exp -le $ALERT_DAYS ]; then
            log "WARNING: Certificate for $domain expires soon! ($days_until_exp days)"
            
            # Try to renew
            if sudo certbot renew --cert-name "$domain" --quiet; then
                log "Successfully renewed certificate for $domain"
                sudo systemctl reload nginx
            else
                log "ERROR: Failed to renew certificate for $domain"
            fi
        fi
    fi
done

# Clean up old log entries (keep last 1000 lines)
tail -n 1000 "$LOG_FILE" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE"
EOF

sudo chmod +x /usr/local/bin/check-ssl-renewal.sh

# Set up automatic SSL renewal
log "Setting up automatic SSL renewal..."

# Create systemd service for SSL renewal check
sudo tee /etc/systemd/system/ssl-renewal-check.service > /dev/null <<'EOF'
[Unit]
Description=SSL Certificate Renewal Check
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/check-ssl-renewal.sh
User=root
EOF

# Create systemd timer for SSL renewal check
sudo tee /etc/systemd/system/ssl-renewal-check.timer > /dev/null <<'EOF'
[Unit]
Description=Run SSL Certificate Renewal Check Daily
Requires=ssl-renewal-check.service

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Enable and start the timer
sudo systemctl daemon-reload
sudo systemctl enable ssl-renewal-check.timer
sudo systemctl start ssl-renewal-check.timer

# Create Docker health check script
log "Creating Docker health check script..."
tee $APP_DIR/health-check.sh > /dev/null <<'EOF'
#!/bin/bash

# Docker Health Check Script
# This script monitors the application and restarts if unhealthy

set -e

APP_DIR="/opt/mivaa-pdf-extractor"
LOG_FILE="$APP_DIR/logs/health-check.log"
MAX_FAILURES=3
FAILURE_COUNT_FILE="$APP_DIR/.health_failures"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Initialize failure count
if [ ! -f "$FAILURE_COUNT_FILE" ]; then
    echo "0" > "$FAILURE_COUNT_FILE"
fi

# Check application health
if curl -f -s http://localhost:8000/health > /dev/null; then
    log "Health check passed"
    echo "0" > "$FAILURE_COUNT_FILE"
    exit 0
else
    # Increment failure count
    current_failures=$(cat "$FAILURE_COUNT_FILE")
    new_failures=$((current_failures + 1))
    echo "$new_failures" > "$FAILURE_COUNT_FILE"
    
    log "Health check failed (attempt $new_failures/$MAX_FAILURES)"
    
    if [ $new_failures -ge $MAX_FAILURES ]; then
        log "Maximum failures reached. Restarting application..."
        
        cd "$APP_DIR"
        docker-compose restart
        
        # Reset failure count
        echo "0" > "$FAILURE_COUNT_FILE"
        
        # Wait a bit and check again
        sleep 30
        if curl -f -s http://localhost:8000/health > /dev/null; then
            log "Application restarted successfully"
        else
            log "ERROR: Application still unhealthy after restart"
        fi
    fi
    
    exit 1
fi
EOF

chmod +x $APP_DIR/health-check.sh

# Create systemd service for health monitoring
sudo tee /etc/systemd/system/mivaa-health-check.service > /dev/null <<EOF
[Unit]
Description=MIVAA PDF Extractor Health Check
After=network.target

[Service]
Type=oneshot
ExecStart=$APP_DIR/health-check.sh
User=$USER
WorkingDirectory=$APP_DIR
EOF

# Create systemd timer for health monitoring
sudo tee /etc/systemd/system/mivaa-health-check.timer > /dev/null <<'EOF'
[Unit]
Description=Run MIVAA Health Check Every 5 Minutes
Requires=mivaa-health-check.service

[Timer]
OnCalendar=*:0/5
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Enable health monitoring
sudo systemctl daemon-reload
sudo systemctl enable mivaa-health-check.timer
sudo systemctl start mivaa-health-check.timer

# Create backup script
log "Creating backup script..."
sudo tee /usr/local/bin/backup-mivaa.sh > /dev/null <<EOF
#!/bin/bash

# MIVAA PDF Extractor Backup Script

set -e

BACKUP_DIR="/opt/backups/mivaa"
APP_DIR="/opt/mivaa-pdf-extractor"
DATE=\$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="mivaa_backup_\$DATE.tar.gz"

# Create backup directory
sudo mkdir -p "\$BACKUP_DIR"

# Create backup
sudo tar -czf "\$BACKUP_DIR/\$BACKUP_FILE" \\
    --exclude='\$APP_DIR/logs/*' \\
    --exclude='\$APP_DIR/.env' \\
    "\$APP_DIR"

# Keep only last 7 backups
sudo find "\$BACKUP_DIR" -name "mivaa_backup_*.tar.gz" -mtime +7 -delete

echo "Backup created: \$BACKUP_DIR/\$BACKUP_FILE"
EOF

sudo chmod +x /usr/local/bin/backup-mivaa.sh

# Set up log rotation
log "Setting up log rotation..."
sudo tee /etc/logrotate.d/mivaa-pdf-extractor > /dev/null <<EOF
$APP_DIR/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        docker-compose -f $APP_DIR/docker-compose.yml restart > /dev/null 2>&1 || true
    endscript
}

/var/log/ssl-renewal-check.log {
    weekly
    missingok
    rotate 4
    compress
    delaycompress
    notifempty
    create 644 root root
}
EOF

# Create environment template
log "Creating environment template..."
tee $APP_DIR/.env.template > /dev/null <<'EOF'
# MIVAA PDF Extractor Environment Configuration
# Copy this file to .env and fill in your values

# Application Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database Configuration
SUPABASE_URL=your-supabase-url
SUPABASE_ANON_KEY=your-supabase-anon-key

# API Keys
OPENAI_API_KEY=your-openai-api-key
MATERIAL_KAI_API_URL=your-material-kai-api-url
MATERIAL_KAI_API_KEY=your-material-kai-api-key

# Monitoring
SENTRY_DSN=your-sentry-dsn

# Container Registry
GITHUB_REPOSITORY=owner/repo
IMAGE_TAG=latest
EOF

# Create useful aliases
log "Creating useful aliases..."
tee -a ~/.bashrc > /dev/null <<EOF

# MIVAA PDF Extractor aliases
alias mivaa-logs='docker-compose -f $APP_DIR/docker-compose.yml logs -f'
alias mivaa-status='docker-compose -f $APP_DIR/docker-compose.yml ps'
alias mivaa-restart='docker-compose -f $APP_DIR/docker-compose.yml restart'
alias mivaa-deploy='cd $APP_DIR && ./scripts/deploy.sh --deploy-app --force'
alias mivaa-deploy-no-nginx='cd $APP_DIR && ./scripts/deploy.sh --deploy-app --no-nginx --force'
alias mivaa-setup='cd $APP_DIR && ./scripts/deploy.sh --setup-server'
alias mivaa-full-deploy='cd $APP_DIR && ./scripts/deploy.sh --full --force'
alias mivaa-backup='sudo /usr/local/bin/backup-mivaa.sh'

# nginx management aliases
alias nginx-restart='cd $APP_DIR && ./scripts/deploy.sh --deploy-app --force-nginx --force'
alias nginx-check='cd $APP_DIR && ./scripts/deploy.sh --check-only'
alias nginx-status='check_nginx_status'

# SSL management aliases
alias ssl-setup='sudo /usr/local/bin/setup-ssl.sh'
alias ssl-check='sudo /usr/local/bin/check-ssl-renewal.sh'
EOF

# Display setup summary
log "Server setup completed successfully!"
echo
info "=== SETUP SUMMARY ==="
info "✅ System packages updated"
info "✅ Docker and Docker Compose installed"
info "✅ NGINX installed and configured"
info "✅ Certbot installed for SSL automation"
info "✅ UFW firewall configured (SSH, HTTP, HTTPS allowed)"
info "✅ Fail2Ban configured for security"
info "✅ Application directory created: $APP_DIR"
info "✅ SSL automation scripts created"
info "✅ Health monitoring configured"
info "✅ Log rotation configured"
info "✅ Backup script created"
echo
warn "=== NEXT STEPS ==="
warn "1. Copy your application code to: $APP_DIR"
warn "2. Create .env file from template: $APP_DIR/.env.template"
warn "3. Set up SSL certificate: sudo /usr/local/bin/setup-ssl.sh yourdomain.com"
warn "4. Deploy your application with: docker-compose up -d"
echo
info "=== USEFUL COMMANDS ==="
info "• View logs: mivaa-logs"
info "• Check status: mivaa-status"
info "• Restart containers: mivaa-restart (Docker restart only)"
info "• Deploy app: mivaa-deploy (comprehensive deployment with nginx)"
info "• Deploy app (no nginx): mivaa-deploy-no-nginx (deployment without nginx restart)"
info "• Setup server: mivaa-setup (initial server configuration)"
info "• Full deployment: mivaa-full-deploy (setup + deploy)"
info "• Restart nginx: nginx-restart (force nginx restart)"
info "• Check nginx: nginx-check (comprehensive nginx health check)"
info "• nginx status: nginx-status (quick nginx status)"
info "• Setup SSL: ssl-setup yourdomain.com"
info "• Check SSL: ssl-check"
info "• Create backup: mivaa-backup"
echo
log "Reboot recommended to ensure all changes take effect"
log "Run: sudo reboot"