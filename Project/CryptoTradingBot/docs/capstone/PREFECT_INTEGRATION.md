# Prefect Integration Guide

This document explains how Prefect workflow orchestration is integrated into the CryptoTradingBot project.

## Overview

Prefect provides enterprise-grade workflow orchestration for ML model training and data pipelines. It offers:

- **Visual Workflow Monitoring**: See your workflows in real-time via Prefect UI
- **Automatic Retries**: Built-in retry logic with exponential backoff
- **Error Handling**: Better error tracking and logging
- **Dependency Management**: Clear task dependencies and execution order
- **Scheduling**: Built-in scheduling and triggers
- **Observability**: Detailed metrics and logs for each workflow run

## Installation

```bash
pip install prefect>=2.14.0
```

## Architecture

### Workflows

The project includes two main Prefect workflows:

1. **Model Retraining Pipeline** (`model_retraining_flow`)
   - Collects training data
   - Prepares features
   - Trains new model
   - Evaluates performance
   - Compares with current model
   - Activates if better

2. **Data Collection Pipeline** (`data_collection_flow`)
   - Collects market data from exchanges
   - Validates and processes data
   - Returns clean dataset

### Tasks

Each workflow consists of multiple tasks:

- `collect_training_data_task`: Fetches historical market data
- `prepare_features_task`: Feature engineering
- `train_model_task`: Model training
- `save_model_task`: Model persistence
- `evaluate_model_performance_task`: Model evaluation
- `compare_models_task`: Model comparison
- `activate_model_task`: Model activation

## Usage

### Running Workflows Locally

#### Option 1: Direct Python Execution

```python
from ml_models.prefect_flows import model_retraining_flow

result = model_retraining_flow(
    symbol="BTC/USDT",
    exchange="binance",
    provider="ccxt",
    days=30
)
```

#### Option 2: Command Line Script

```bash
# Using Prefect-based script
python scripts/retrain_model_prefect.py --symbol BTC/USDT --days 30
```

#### Option 3: Using Retraining Service with Prefect

```python
from ml_models.retraining_service import ModelRetrainingService

config = {
    'model_dir': 'ml_models',
    'symbol': 'BTC/USDT',
    'exchange': 'binance',
    'provider': 'ccxt',
    'use_prefect': True  # Enable Prefect
}

service = ModelRetrainingService(config)
result = service.retrain_if_needed()
```

### Running with Prefect Server

#### Start Prefect Server

```bash
# Terminal 1: Start Prefect server
prefect server start

# Server UI available at http://localhost:4200
```

#### Run Workflow with Prefect

```bash
# Run workflow and register with server
prefect deployment run ml-models/model-retraining-pipeline
```

### Deploying Workflows

#### Create Deployment

```bash
# Navigate to deployment directory
cd deployment/prefect

# Run deployment script
python deploy.py

# Or use Prefect CLI
prefect deploy ml_models/prefect_flows.py:model_retraining_flow --name model-retraining
```

#### Schedule Automated Runs

Edit `deployment/prefect/deploy.py` to customize schedules:

```python
model_retraining_flow.serve(
    name="model-retraining",
    cron="0 2 * * *",  # Daily at 2 AM
    # or
    interval=60 * 60 * 24 * 7,  # Weekly
)
```

## Monitoring

### Prefect UI

Access Prefect UI at `http://localhost:4200` to:

- View workflow runs
- Monitor task execution
- Check logs and errors
- View workflow graphs
- Track performance metrics

### Viewing Flow Runs

```bash
# List recent flow runs
prefect flow-run ls

# View specific run details
prefect flow-run inspect <run-id>

# View logs
prefect flow-run logs <run-id>
```

### Programmatic Monitoring

```python
from prefect import get_client

async with get_client() as client:
    # Get recent flow runs
    runs = await client.read_flow_runs()
    for run in runs:
        print(f"{run.name}: {run.state}")
```

## Configuration

### Environment Variables

```bash
# Optional: Prefect Cloud API key
export PREFECT_API_KEY=your_api_key_here

# Optional: Prefect Cloud workspace
export PREFECT_API_URL=https://api.prefect.cloud/api/accounts/[ACCOUNT_ID]/workspaces/[WORKSPACE_ID]
```

### Configuration Files

Workflow parameters can be configured in:

- `deployment/prefect/deploy.py`: Deployment settings
- Environment variables: API keys, paths
- Command-line arguments: Runtime parameters

## Benefits Over Standard Retraining

### Without Prefect
- Manual error handling
- No visual workflow monitoring
- Limited retry logic
- Hard to debug failures
- No execution history

### With Prefect
- Automatic retries with exponential backoff
- Visual workflow monitoring in UI
- Detailed execution logs
- Built-in error handling
- Execution history and metrics
- Easy scheduling and triggers
- Better observability

## Example Workflow Execution

```python
from ml_models.prefect_flows import model_retraining_flow

# Run workflow
result = model_retraining_flow(
    symbol="BTC/USDT",
    exchange="binance",
    provider="ccxt",
    days=30,
    model_dir="ml_models"
)

# Check results
if result['success']:
    print(f"Model {result['version']} trained successfully")
    print(f"Accuracy: {result['metrics']['validation_accuracy']}")
    if result['activated']:
        print("Model activated!")
else:
    print(f"Training failed: {result['error']}")
```

## Scheduling

### Cron Schedule

```python
from prefect import serve
from ml_models.prefect_flows import model_retraining_flow

# Schedule daily at 2 AM
model_retraining_flow.serve(
    name="daily-model-retraining",
    cron="0 2 * * *"
)
```

### Interval Schedule

```python
# Run every 7 days
model_retraining_flow.serve(
    name="weekly-model-retraining",
    interval=60 * 60 * 24 * 7
)
```

### Manual Trigger

```bash
# Trigger via CLI
prefect deployment run model-retraining/daily-model-retraining

# Or via API
curl -X POST http://localhost:4200/api/deployments/<deployment-id>/create_flow_run
```

## Error Handling

Prefect automatically handles:

- **Retries**: Failed tasks retry up to N times
- **Timeouts**: Tasks timeout after specified duration
- **Validation**: Input validation before task execution
- **Logging**: Comprehensive logging at each step

### Custom Error Handling

```python
@task(retries=3, retry_delay_seconds=30)
def my_task():
    # Prefect will automatically retry this up to 3 times
    # with 30-second delay between retries
    pass
```

## Integration with Existing System

The Prefect integration is **optional**. The system can work:

1. **Without Prefect**: Uses standard retraining service (cron-based)
2. **With Prefect**: Uses Prefect workflows for better orchestration

To enable Prefect:
- Set `use_prefect=True` in retraining service config
- Use `scripts/retrain_model_prefect.py` instead of `scripts/retrain_model.py`
- Deploy workflows to Prefect server

## Troubleshooting

### Prefect Server Not Starting

```bash
# Check if port 4200 is available
lsof -i :4200

# Start with different port
prefect server start --port 8080
```

### Workflow Failures

- Check Prefect UI for error details
- Review logs: `prefect flow-run logs <run-id>`
- Verify API keys and permissions
- Check data availability

### Import Errors

Ensure Prefect is installed:
```bash
pip install prefect>=2.14.0
```

## Best Practices

1. **Use Prefect for Production**: Enable Prefect for production deployments
2. **Monitor Workflows**: Regularly check Prefect UI for failures
3. **Set Appropriate Retries**: Configure retry logic for network-dependent tasks
4. **Logging**: Use Prefect's built-in logging for better observability
5. **Versioning**: Keep workflow versions in sync with model versions

## Next Steps

- Deploy workflows to Prefect Cloud for remote monitoring
- Set up automated scheduling for model retraining
- Integrate with monitoring/alerting systems
- Add more workflows (data validation, backtesting, etc.)

---

**Note**: Prefect integration is optional but recommended for production deployments and better observability.
