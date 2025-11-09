# VPS Deployment Guide - CryptoTradingBot

This guide will walk you through deploying the CryptoTradingBot on a VPS (Virtual Private Server) for testing Alpaca paper trading.

## Prerequisites

- VPS with Ubuntu 20.04 or later (2GB RAM minimum, 4GB recommended)
- Root or sudo access
- Domain name or static IP address
- SSH access to your VPS
- Alpaca paper trading account (https://alpaca.markets/)

---

## Step 1: VPS Setup

### 1.1 Connect to Your VPS

```bash
ssh root@your-vps-ip
# or
ssh user@your-vps-ip
```

### 1.2 Update System

```bash
sudo apt update
sudo apt upgrade -y
```

### 1.3 Install Required Software

```bash
# Install Python 3.11 and pip
sudo apt install -y python3.11 python3.11-pip python3.11-venv

# Install Git
sudo apt install -y git

# Install Nginx (for reverse proxy)
sudo apt install -y nginx

# Install PostgreSQL (optional, if you want to use it instead of SQLite)
# sudo apt install -y postgresql postgresql-contrib

# Install supervisor for process management
sudo apt install -y supervisor

# Install firewall
sudo apt install -y ufw
```

### 1.4 Configure Firewall

```bash
# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP
sudo ufw allow 80/tcp

# Allow HTTPS
sudo ufw allow 443/tcp

# Allow Flask app (if not using Nginx)
sudo ufw allow 5000/tcp

# Enable firewall
sudo ufw enable
sudo ufw status
```

---

## Step 2: Application Deployment

### 2.1 Clone Repository

```bash
# Navigate to web directory
cd /var/www

# Clone the repository
sudo git clone https://your-repo-url.git crypto-trading-bot

# Or upload files via SCP
# scp -r ./CryptoTradingBot user@your-vps:/var/www/crypto-trading-bot

# Set ownership
sudo chown -R $USER:$USER /var/www/crypto-trading-bot
cd crypto-trading-bot
```

### 2.2 Create Python Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2.3 Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install any additional dependencies
pip install gunicorn  # For production WSGI server
```

---

## Step 3: Configuration

### 3.1 Create .env File

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your settings
nano .env
```

### 3.2 Configure Environment Variables

Update the following in `.env`:

```bash
# Alpaca API Configuration
ALPACA_API_KEY=your_actual_api_key
ALPACA_SECRET_KEY=your_actual_secret_key
APCA_API_KEY_ID=${ALPACA_API_KEY}
APCA_API_SECRET_KEY=${ALPACA_SECRET_KEY}
ALPACA_PAPER_TRADING=true

# Flask Configuration
FLASK_ENV=production
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=false

# Generate a secure Flask secret key
FLASK_SECRET_KEY=your_secure_secret_key_here

# Database
DATABASE_PATH=data/db/trading_bot.db

# Server Configuration
SERVER_HOST=your-domain.com

# Frontend
REACT_APP_PRODUCTION_API_URL=http://your-domain.com:5000
```

### 3.3 Generate Secure Keys

```bash
# Generate Flask secret key
python -c "import secrets; print(secrets.token_hex(32))"

# Generate API key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate JWT secret key
python -c "import secrets; print(secrets.token_hex(32))"
```

### 3.4 Create Required Directories

```bash
# Create directories
mkdir -p data/db
mkdir -p logs
mkdir -p backups

# Set permissions
chmod 755 data/db
chmod 755 logs
chmod 755 backups
```

---

## Step 4: Initialize Database

```bash
# Activate virtual environment
source venv/bin/activate

# Initialize database
python backend/app.py --init-db

# Or use the main script
python main.py --init-db
```

---

## Step 5: Configure Supervisor

Supervisor will manage the Flask application process.

### 5.1 Create Supervisor Config

```bash
sudo nano /etc/supervisor/conf.d/crypto-trading-bot.conf
```

Add the following configuration:

```ini
[program:crypto-trading-bot]
command=/var/www/crypto-trading-bot/venv/bin/python /var/www/crypto-trading-bot/backend/app.py
directory=/var/www/crypto-trading-bot
user=your_user
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/supervisor/crypto-trading-bot.log
environment=FLASK_ENV="production"
```

### 5.2 Reload Supervisor

```bash
# Reload supervisor configuration
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start crypto-trading-bot

# Check status
sudo supervisorctl status crypto-trading-bot
```

---

## Step 6: Configure Nginx (Optional but Recommended)

Nginx will act as a reverse proxy for the Flask application.

### 6.1 Create Nginx Configuration

```bash
sudo nano /etc/nginx/sites-available/crypto-trading-bot
```

Add the following configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Static files
    location /static {
        alias /var/www/crypto-trading-bot/frontend/build/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
}
```

### 6.2 Enable Site

```bash
# Create symlink
sudo ln -s /etc/nginx/sites-available/crypto-trading-bot /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

---

## Step 7: Deploy Frontend (Optional)

If you want to serve the React frontend from the VPS:

### 7.1 Build Frontend

```bash
cd frontend
npm install
npm run build
```

### 7.2 Configure Nginx for Frontend

Update the Nginx configuration to serve the built frontend:

```nginx
location / {
    root /var/www/crypto-trading-bot/frontend/build;
    try_files $uri $uri/ /index.html;
}
```

---

## Step 8: SSL Certificate (Optional but Recommended)

### 8.1 Install Certbot

```bash
sudo apt install -y certbot python3-certbot-nginx
```

### 8.2 Obtain SSL Certificate

```bash
sudo certbot --nginx -d your-domain.com
```

Certbot will automatically configure Nginx to use HTTPS.

---

## Step 9: Test Deployment

### 9.1 Check Application Status

```bash
# Check supervisor status
sudo supervisorctl status

# Check logs
sudo supervisorctl tail -f crypto-trading-bot

# Or check application logs
tail -f /var/log/supervisor/crypto-trading-bot.log
```

### 9.2 Test API Endpoints

```bash
# Health check
curl http://localhost:5000/health

# Or from external machine
curl http://your-domain.com/health
```

### 9.3 Test Trading Bot

```bash
# SSH into VPS and run
cd /var/www/crypto-trading-bot
source venv/bin/activate

# Run in demo mode
python main.py --mode demo --symbol BTC/USD

# Run simulation
python main.py --mode simulate --symbol BTC/USD --cash 10000
```

---

## Step 10: Monitoring and Maintenance

### 10.1 View Logs

```bash
# Application logs
tail -f /var/log/supervisor/crypto-trading-bot.log

# Or custom log file
tail -f logs/trading_bot.log

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### 10.2 Restart Services

```bash
# Restart Flask application
sudo supervisorctl restart crypto-trading-bot

# Restart Nginx
sudo systemctl restart nginx

# Check status
sudo supervisorctl status
sudo systemctl status nginx
```

### 10.3 Update Application

```bash
cd /var/www/crypto-trading-bot

# Pull latest changes
git pull origin main

# Restart application
sudo supervisorctl restart crypto-trading-bot
```

---

## Troubleshooting

### Issue: Application not starting

```bash
# Check logs
sudo supervisorctl tail -f crypto-trading-bot

# Check if port 5000 is in use
sudo lsof -i :5000

# Test manually
cd /var/www/crypto-trading-bot
source venv/bin/activate
python backend/app.py
```

### Issue: Database errors

```bash
# Check database exists
ls -la data/db/

# Reinitialize database
python -m data.db.schema
```

### Issue: Nginx not working

```bash
# Test configuration
sudo nginx -t

# Check status
sudo systemctl status nginx

# View error logs
sudo tail -f /var/log/nginx/error.log
```

### Issue: Firewall blocking

```bash
# Check firewall status
sudo ufw status

# Allow specific port
sudo ufw allow 5000/tcp
```

---

## Security Checklist

- [ ] Firewall configured
- [ ] SSL certificate installed (HTTPS)
- [ ] Strong passwords for all accounts
- [ ] SSH key authentication enabled
- [ ] Environment variables secured (not in git)
- [ ] Database backups configured
- [ ] Log rotation configured
- [ ] Rate limiting enabled
- [ ] API keys stored securely
- [ ] Regular security updates

---

## Quick Reference

### Useful Commands

```bash
# Check application status
sudo supervisorctl status crypto-trading-bot

# View logs
sudo tail -f /var/log/supervisor/crypto-trading-bot.log

# Restart application
sudo supervisorctl restart crypto-trading-bot

# Check Nginx status
sudo systemctl status nginx

# Test Nginx configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx

# View firewall status
sudo ufw status

# Test API
curl http://localhost:5000/health
```

### Files and Directories

- **Application:** `/var/www/crypto-trading-bot`
- **Virtual Environment:** `/var/www/crypto-trading-bot/venv`
- **Configuration:** `/var/www/crypto-trading-bot/.env`
- **Database:** `/var/www/crypto-trading-bot/data/db/trading_bot.db`
- **Logs:** `/var/log/supervisor/crypto-trading-bot.log`
- **Nginx Config:** `/etc/nginx/sites-available/crypto-trading-bot`

---

## Support

For issues or questions:
1. Check logs in `/var/log/supervisor/`
2. Check application logs in `logs/`
3. Review this deployment guide
4. Check the main README.md

**Deployment Successful! 🎉**
