#!/bin/bash

# Crypto Trading Bot Deployment Script
# This script sets up and deploys the complete trading bot system

set -e

echo "🚀 Starting Crypto Trading Bot Deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating environment file..."
    cat > .env << EOF
# Alpaca API Configuration
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here

# Security Configuration
JWT_SECRET_KEY=$(openssl rand -base64 32)
API_KEY=$(openssl rand -base64 16)

# Database Configuration
DATABASE_URL=sqlite:///app/data/db/trading_bot.db

# Flask Configuration
FLASK_ENV=production
FLASK_SECRET_KEY=$(openssl rand -base64 32)

# Frontend Configuration
REACT_APP_API_URL=http://localhost:5000
EOF
    echo "✅ Environment file created. Please update with your actual API keys."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/logs data/db notebooks artifacts

# Set permissions
chmod 755 data/logs data/db notebooks artifacts

# Build and start services
echo "🔨 Building Docker images..."
docker-compose -f deployment/docker-compose.yml build

echo "🚀 Starting services..."
docker-compose -f deployment/docker-compose.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🏥 Checking service health..."

# Check backend health
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ Backend service is healthy"
else
    echo "❌ Backend service is not responding"
    docker-compose -f deployment/docker-compose.yml logs backend
    exit 1
fi

# Check frontend health
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend service is healthy"
else
    echo "❌ Frontend service is not responding"
    docker-compose -f deployment/docker-compose.yml logs frontend
    exit 1
fi

echo "🎉 Deployment completed successfully!"
echo ""
echo "📊 Services are running:"
echo "  - Frontend: http://localhost:3000"
echo "  - Backend API: http://localhost:5000"
echo "  - API Health: http://localhost:5000/health"
echo ""
echo "📝 Next steps:"
echo "  1. Update .env file with your Alpaca API keys"
echo "  2. Restart services: docker-compose -f deployment/docker-compose.yml restart"
echo "  3. Access the dashboard at http://localhost:3000"
echo ""
echo "🔧 Useful commands:"
echo "  - View logs: docker-compose -f deployment/docker-compose.yml logs"
echo "  - Stop services: docker-compose -f deployment/docker-compose.yml down"
echo "  - Restart services: docker-compose -f deployment/docker-compose.yml restart"
