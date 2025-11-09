# Crypto Trading Bot Simulator

A comprehensive cryptocurrency trading bot simulator that implements both descriptive (technical indicators) and prescriptive (machine learning) methods for automated trading decisions. Built for WGU Computer Science Capstone project.

## 🎯 Project Overview

This project demonstrates a complete end-to-end system for algorithmic trading simulation, featuring:

- **Real-time Data Processing**: Live cryptocurrency price feeds via Alpaca API
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages
- **Machine Learning**: Random Forest model for price prediction
- **Risk Management**: Stop-loss, position sizing, portfolio limits
- **Interactive Dashboard**: React-based web interface with real-time updates
- **Comprehensive Monitoring**: Health checks, logging, and performance metrics

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API    │    │   Data Layer    │
│   (React)       │◄──►│   (Flask)        │◄──►│   (SQLite)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Trading Engine  │
                       │ - Indicators    │
                       │ - Portfolio     │
                       │ - Risk Mgmt     │
                       │ - ML Models     │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ External APIs    │
                       │ - Alpaca Crypto  │
                       │ - CCXT Library   │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose (for full stack)
- Python 3.11+
- Git
- Alpaca API account (for live data) or CCXT-compatible exchange

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CryptoTradingBot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys** (optional, for Alpaca)
   - Edit `.env` file or set environment variables
   - Add your Alpaca API credentials

4. **Run modes**
   ```bash
   # Demo mode - test indicators (with Alpaca)
   python main.py --mode demo --symbol BTC/USD --provider alpaca
   
   # Demo mode - test indicators (with CCXT)
   python main.py --mode demo --symbol BTC/USDT --provider ccxt
   
   # Simulation mode - backtest
   python main.py --mode simulate --symbol BTC/USD --provider alpaca --cash 10000
   
   # Continuous trading (24/7) with Alpaca
   python main.py --mode run --symbol BTC/USD --provider alpaca --interval 60
   
   # Continuous trading with ML predictions
   python main.py --mode run --symbol BTC/USD --provider alpaca --enable-ml --model-path ml_models/trained_model.pkl
   
   # PPO Reinforcement Learning trading
   python scripts/train_ppo.py --symbol BTC/USD --timeframe 1h --days 365 --timesteps 100000
   python scripts/deploy_ppo.py --ppo-model models/ppo_trading_agent --ml-model ml_models/trained_model.pkl --symbol BTC/USD --paper-trading
   ```

### Complete Command-Line Options

The `main.py` script supports the following arguments:

#### Core Arguments
- `--mode`: Run mode - `demo`, `simulate`, or `run` (default: `demo`)
  - `demo`: Test indicators and data fetching
  - `simulate`: Backtest on historical data
  - `run`: Continuous 24/7 live trading

#### Trading Configuration
- `--symbol`: Trading symbol (default: `BTC/USD`)
  - Examples: `BTC/USD`, `ETH/USD`, `BTC/USDT`
- `--exchange`: Exchange name (default: `alpaca`)
- `--timeframe`: Data timeframe (default: `1m`)
  - Examples: `1m`, `5m`, `15m`, `1h`, `1d`
- `--cash`: Starting cash balance in USD (default: `10000.0`)

#### Data Provider
- `--provider`: Data provider - `ccxt` or `alpaca` (default: `alpaca`)
  - `alpaca`: Uses Alpaca Markets API (requires API keys)
  - `ccxt`: Uses CCXT library for exchange connectivity

#### Continuous Trading Options
- `--interval`: Update interval in seconds for continuous mode (default: `60`)

#### Machine Learning Options
- `--enable-ml`: Enable ML predictions (requires trained model)
- `--enable-auto-retraining`: Enable automatic model retraining (requires ML enabled)
- `--model-path`: Path to ML model file (default: `ml_models/trained_model.pkl`)

#### Testing Options
- `--test-order`: Place a small Alpaca paper trade and exit
- `--test-side`: Test order side - `buy` or `sell` (default: `buy`)
- `--test-notional`: USD notional for test order (default: `10.0`)

#### Example Usage

```bash
# Basic demo
python main.py --mode demo --symbol BTC/USD --provider alpaca

# Full simulation with custom parameters
python main.py --mode simulate --symbol ETH/USD --provider alpaca --cash 50000 --timeframe 5m

# Continuous trading with ML enabled
python main.py --mode run --symbol BTC/USD --provider alpaca --interval 60 --enable-ml

# Continuous trading with auto-retraining
python main.py --mode run --symbol BTC/USD --provider alpaca --enable-ml --enable-auto-retraining --interval 120

# Test Alpaca connection and place test order
python main.py --test-order --provider alpaca --symbol BTC/USD --test-side buy --test-notional 10.0
```

### Full Stack Deployment

1. **Run the deployment script**
   ```bash
   ./deployment/scripts/deploy.sh
   ```

2. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000
   - Health Check: http://localhost:5000/health

### 24/7 Continuous Trading

See [24/7 Deployment Guide](docs/deployment/DEPLOYMENT_24_7.md) for detailed instructions on:
- Setting up continuous trading service
- Model retraining and maintenance
- Production deployment with systemd/supervisor
- Monitoring and troubleshooting

### Prefect Workflow Orchestration

See [Prefect Integration Guide](docs/capstone/PREFECT_INTEGRATION.md) for:
- Workflow orchestration setup
- Model retraining workflows
- Monitoring and observability
- Production deployment with Prefect

### Complete Running Instructions

See [Running Instructions](docs/guides/RUNNING_INSTRUCTIONS.md) for detailed step-by-step guides:
- Local development setup
- VPS production deployment
- Docker deployment
- Service management
- Troubleshooting guide

### PPO Reinforcement Learning

See [PPO Quick Start Guide](docs/ml/PPO_QUICK_START.md) for:
- PPO agent training
- Hybrid trading system deployment
- Reinforcement learning configuration

## 📊 Features

### Core Requirements (WGU Capstone)

**Descriptive Method**: Technical indicators (RSI, MACD, Bollinger Bands, SMA/EMA)  
**Prescriptive Method**: Random Forest ML model for trading signals  
**Data Collection**: Live cryptocurrency data via Alpaca API  
**Decision Support**: Real-time trade recommendations with confidence scores  
**Data Wrangling**: Comprehensive cleaning, validation, and feature engineering  
**Data Exploration**: Jupyter notebooks with statistical analysis  
**Data Visualization**: Interactive charts and dashboards  
**Interactive Queries**: REST API with filtering and search capabilities  
**Machine Learning**: Random Forest with walk-forward validation  
**Accuracy Evaluation**: Precision, recall, F1-score, confusion matrix  
**Security Features**: Input validation, rate limiting, secure logging  
**Monitoring**: Health checks, structured logging, error tracking  
**Dashboard**: Three visualization types (candlestick, line, bar charts)  

### Technical Features

- **Real-time Data Processing**: Live price feeds with WebSocket support
- **24/7 Continuous Trading**: Automated trading service that runs continuously
- **Advanced Risk Management**: Position sizing, stop-loss, portfolio limits
- **Machine Learning Pipeline**: Feature engineering, model training, evaluation
- **Model Retraining Service**: Automatic model retraining with versioning
- **Prefect Workflow Orchestration**: Enterprise-grade workflow management for ML pipelines
- **Interactive Dashboard**: Real-time portfolio tracking and analytics
- **Comprehensive Logging**: Structured JSON logging with rotation
- **Docker Deployment**: Complete containerized deployment
- **Production Deployment**: systemd and supervisor configurations included
- **API Documentation**: RESTful API with comprehensive endpoints

## 🛠️ Technology Stack

### Backend
- **Python 3.11+**: Core application language
- **Flask**: REST API framework
- **SQLite**: Database for trades and portfolio data
- **CCXT**: Cryptocurrency exchange connectivity
- **Scikit-learn**: Machine learning models
- **Pandas/NumPy**: Data processing and analysis

### Frontend
- **React 18+**: User interface framework
- **Material-UI**: Component library
- **Recharts**: Data visualization
- **Axios**: HTTP client for API communication

### DevOps
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **Nginx**: Reverse proxy and load balancing

### Analysis
- **Jupyter Notebooks**: Data exploration and analysis
- **Matplotlib/Seaborn**: Statistical visualization
- **Pandas Profiling**: Automated data analysis

## 📁 Project Structure

```
CryptoTradingBot/
├── backend/                 # Flask API server
│   ├── app.py              # Main Flask application
│   └── security/           # Security modules
├── frontend/               # React dashboard
│   ├── src/
│   │   ├── components/     # Reusable components
│   │   ├── pages/          # Main pages
│   │   └── services/       # API services
│   └── public/             # Static assets
├── trading_engine/         # Core trading logic
│   ├── indicators.py       # Technical indicators
│   ├── portfolio.py        # Portfolio management
│   ├── risk_manager.py     # Risk management
│   └── decision_support.py # Trading decisions
├── ml_models/              # Machine learning
│   ├── features.py         # Feature engineering
│   ├── predictor.py        # ML models
│   └── evaluation.py       # Model evaluation
├── data/                   # Data management
│   ├── collector.py        # Data collection
│   ├── processor.py        # Data processing
│   └── db/                 # Database schema
├── deployment/             # Deployment configs
│   ├── docker/             # Docker files
│   ├── scripts/            # Deployment scripts
│   └── docker-compose.yml  # Container orchestration
├── tests/                  # Test suite
├── docs/                   # Documentation
│   ├── guides/             # Getting started guides
│   ├── architecture/       # System architecture docs
│   ├── ml/                 # Machine learning docs
│   ├── deployment/         # Deployment guides
│   └── capstone/           # Capstone project docs
└── config/                 # Configuration files
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Alpaca API Configuration
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here

# Security Configuration
JWT_SECRET_KEY=your_jwt_secret_key
API_KEY=your_api_key

# Database Configuration
DATABASE_URL=sqlite:///app/data/db/trading_bot.db

# Flask Configuration
FLASK_ENV=production
FLASK_SECRET_KEY=your_flask_secret_key

# Frontend Configuration
REACT_APP_API_URL=http://localhost:5000
```

### Trading Parameters

Configure trading parameters in the risk management system:

- **Max Position %**: Maximum percentage of portfolio per position (default: 20%)
- **Stop Loss %**: Default stop-loss percentage (default: 5%)
- **Max Leverage**: Maximum leverage allowed (default: 1.0)
- **Daily Loss Limit**: Maximum daily loss percentage (default: 2%)

## 📈 Usage

### Dashboard Overview

The main dashboard provides:
- **Portfolio Summary**: Total value, P&L, win rate, total trades
- **Real-time Charts**: Portfolio value over time, trade distribution
- **Current Prices**: Live cryptocurrency prices
- **Recent Trades**: Latest trading activity

### Portfolio Management

- **Position Tracking**: Real-time position monitoring
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate
- **Risk Assessment**: Portfolio VaR, concentration risk

### Trading Signals

- **Technical Analysis**: RSI, MACD, Bollinger Bands signals
- **ML Predictions**: Random Forest model predictions
- **Decision Support**: Combined signal analysis with confidence scores

### Analytics

- **Performance Analysis**: Historical returns, drawdown analysis
- **Risk Metrics**: VaR, volatility, correlation analysis
- **Model Evaluation**: ML model accuracy and performance

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/

# Run with coverage
python -m pytest --cov=. tests/
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **ML Tests**: Model validation and evaluation
- **Performance Tests**: Load and stress testing

## 📊 Performance Metrics

### Trading Performance
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

### ML Model Performance
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity to positive signals
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results

## 🔒 Security

### Implemented Security Features

- **Input Validation**: Pydantic models for all API inputs
- **Rate Limiting**: Request throttling to prevent abuse
- **Authentication**: JWT-based authentication system
- **API Key Management**: Secure API key storage and validation
- **SQL Injection Prevention**: Parameterized queries
- **CORS Protection**: Cross-origin request security
- **Secure Logging**: Audit trail without sensitive data exposure

### Security Best Practices

- Store API keys in environment variables
- Use HTTPS in production
- Implement proper error handling
- Regular security audits
- Monitor for suspicious activity

## 📚 Documentation

### Documentation Structure

All documentation is organized in the `docs/` directory:

- **Guides** (`docs/guides/`): Getting started, quick start, running instructions
- **Architecture** (`docs/architecture/`): System architecture, technical design
- **Machine Learning** (`docs/ml/`): ML setup, PPO guides, feature engineering
- **Deployment** (`docs/deployment/`): Docker, VPS, and 24/7 deployment guides
- **Capstone** (`docs/capstone/`): Project proposal, reflections, Prefect integration

### API Documentation

The REST API provides comprehensive endpoints:

- **Portfolio**: `/api/portfolio`, `/api/portfolio/positions`
- **Trades**: `/api/trades`, `/api/trades/{trade_id}`
- **Market Data**: `/api/market/prices`, `/api/market/history/{symbol}`
- **Technical Indicators**: `/api/indicators/{symbol}`
- **ML Predictions**: `/api/predictions/{symbol}`
- **Trading Signals**: `/api/signals/{symbol}`
- **Risk Management**: `/api/risk/assessment`
- **Performance**: `/api/metrics/performance`
- **System Health**: `/health`

### Jupyter Notebooks

- **Data Exploration**: Comprehensive data analysis and visualization
- **ML Evaluation**: Model training, validation, and performance analysis
- **Feature Engineering**: Technical indicator implementation and analysis

## 🚀 Deployment

### Docker Compose Deployment

```bash
# Navigate to deployment directory
cd deployment

# Configure environment (copy .env.example to .env and add API keys)
cp ../.env.example ../.env

# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Check service status
docker compose ps

# Stop services
docker compose down
```

**Services included:**
- Backend API (port 5000)
- Frontend Dashboard (port 3000)
- Continuous Trading Service (24/7)
- Nginx Reverse Proxy (optional)
- Redis Cache (optional)

**For detailed Docker guide:** See `docs/deployment/DOCKER_DEPLOYMENT.md`

### Production Deployment

1. **Set up production environment**
   - Configure production database
   - Set up SSL certificates
   - Configure reverse proxy

2. **Deploy with Docker**
   - Use production Docker images
   - Configure environment variables
   - Set up monitoring and logging

3. **Monitor and maintain**
   - Set up health checks
   - Configure log rotation
   - Monitor performance metrics

## 🤝 Contributing

### Development Setup

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   cd frontend && npm install
   ```
3. **Set up development environment**
4. **Run tests**
5. **Make changes and submit pull request**

### Code Standards

- Follow PEP 8 for Python code
- Use ESLint for JavaScript/React code
- Write comprehensive tests
- Document all functions and classes
- Follow semantic versioning

## 📄 License

This project is developed for educational purposes as part of the WGU Computer Science Capstone project.

## 🙏 Acknowledgments

- **Alpaca Markets**: For providing cryptocurrency trading API
- **CCXT Library**: For unified exchange connectivity
- **Scikit-learn**: For machine learning capabilities
- **React Community**: For excellent frontend framework
- **Material-UI**: For beautiful component library

## 📞 Support

For questions or issues:

1. Check the documentation
2. Review the Jupyter notebooks
3. Check the API health endpoint
4. Review system logs
5. Contact the development team

---

**Note**: This is a simulation system for educational purposes. It does not use real money and is not intended for actual trading without proper testing and validation.
