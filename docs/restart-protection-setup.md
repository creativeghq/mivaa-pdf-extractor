# 🔒 Restart Protection System

## Overview

The Restart Protection System prevents accidental service restarts that would interrupt active PDF processing jobs. It provides:

1. **Safe Restart Script** - Checks for active jobs before restarting
2. **Admin API Endpoint** - Authorized restarts with job awareness
3. **Bash Alias Protection** - Blocks direct `systemctl restart` commands

---

## 🚀 Installation

### Step 1: Make Scripts Executable

```bash
cd /var/www/mivaa-pdf-extractor
chmod +x scripts/safe-restart.sh
chmod +x scripts/block-unsafe-restart.sh
```

### Step 2: Add Bash Alias (Block Direct Restarts)

Add to `/root/.bashrc`:

```bash
# Block unsafe restarts of MIVAA service
alias systemctl-restart-mivaa='bash /var/www/mivaa-pdf-extractor/scripts/block-unsafe-restart.sh'

# Override systemctl for mivaa-pdf-extractor
systemctl() {
    if [[ "$1" == "restart" && "$2" == "mivaa-pdf-extractor" ]]; then
        bash /var/www/mivaa-pdf-extractor/scripts/block-unsafe-restart.sh
    else
        command systemctl "$@"
    fi
}
```

Reload bashrc:
```bash
source /root/.bashrc
```

### Step 3: Set Admin Restart Token

Add to systemd service file (`/etc/systemd/system/mivaa-pdf-extractor.service`):

```ini
Environment=ADMIN_RESTART_TOKEN=your-secure-random-token-here
```

Generate secure token:
```bash
openssl rand -hex 32
```

Reload systemd:
```bash
sudo systemctl daemon-reload
sudo systemctl restart mivaa-pdf-extractor
```

---

## 📖 Usage

### Safe Restart (Recommended)

**Check and restart if safe**:
```bash
cd /var/www/mivaa-pdf-extractor
bash scripts/safe-restart.sh --reason "Deploy bug fix"
```

**Force restart (interrupts jobs)**:
```bash
bash scripts/safe-restart.sh --force --reason "Critical security patch"
```

### Admin API Endpoint

**Check for active jobs**:
```bash
curl http://localhost:8000/api/jobs/health
```

**Safe restart via API**:
```bash
curl -X POST http://localhost:8000/api/admin/restart-service \
  -H "Content-Type: application/json" \
  -d '{
    "force": false,
    "reason": "Deploy new features",
    "admin_token": "your-admin-token"
  }'
```

**Force restart via API**:
```bash
curl -X POST http://localhost:8000/api/admin/restart-service \
  -H "Content-Type: application/json" \
  -d '{
    "force": true,
    "reason": "Emergency deployment",
    "admin_token": "your-admin-token"
  }'
```

---

## 🔐 Security

1. **Admin Token**: Store in environment variable, never commit to git
2. **Bash Alias**: Prevents accidental direct restarts
3. **API Logging**: All restart attempts logged to Sentry
4. **Job Protection**: Active jobs marked as interrupted before restart

---

## 🎯 Behavior

| Scenario | Force=False | Force=True |
|----------|-------------|------------|
| No active jobs | ✅ Restart immediately | ✅ Restart immediately |
| Active jobs exist | ❌ Block restart | ⚠️ Interrupt jobs + restart |

---

## 📊 Monitoring

All restart attempts are logged:
- **Sentry**: Error tracking with restart reason
- **Systemd Journal**: `journalctl -u mivaa-pdf-extractor`
- **Database**: Interrupted jobs marked in `background_jobs` table

---

## 🚨 Emergency Override

If the protection system itself fails, use:

```bash
# Bypass all protection (USE WITH CAUTION!)
sudo /usr/bin/systemctl restart mivaa-pdf-extractor
```

This bypasses the bash alias by using the full path to systemctl.

---

## 🔄 GitHub Actions Integration

Update `.github/workflows/deploy.yml` to use safe restart:

```yaml
- name: 🔄 Safe Restart Service
  run: |
    ssh root@104.248.68.3 "
    cd /var/www/mivaa-pdf-extractor
    bash scripts/safe-restart.sh --force --reason 'GitHub Actions deployment'
    "
```

---

## ✅ Testing

Test the protection system:

```bash
# 1. Start a test job (will take ~1 hour)
cd /var/www/mivaa-pdf-extractor
python3 scripts/testing/comprehensive_nova_test.py

# 2. Try to restart (should be blocked)
systemctl restart mivaa-pdf-extractor
# Expected: ❌ RESTART BLOCKED!

# 3. Try safe restart (should be blocked)
bash scripts/safe-restart.sh
# Expected: ❌ RESTART BLOCKED! Found 1 active job(s)

# 4. Force restart (should work with confirmation)
bash scripts/safe-restart.sh --force --reason "Testing force restart"
# Expected: Prompts for confirmation, then restarts
```

