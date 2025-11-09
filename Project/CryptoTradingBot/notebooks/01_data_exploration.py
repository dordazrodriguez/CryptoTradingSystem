"""
Jupyter Notebook: Data Exploration for Cryptocurrency Trading Bot
Capstone Project - WGU Computer Science

This notebook demonstrates:
1. Data collection from cryptocurrency exchanges
2. Data wrangling and cleaning
3. Statistical analysis and exploration
4. Technical indicator calculations (descriptive method)
5. Data visualization (3 types: candlestick, line, bar charts)
"""

# Cell 1: Setup and Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.append('..')

# Import our custom modules
try:
    from data.data_feeder import DataFeed, FeedConfig, sma, rsi
    from data.processor import DataProcessor
    from trading_engine.indicators import TechnicalIndicators
    from data.db import get_db_manager
except ImportError as e:
    print(f"Import warning: {e}")
    print("Using fallback implementations")

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

print("=" * 60)
print("CryptoTradingBot - Data Exploration Notebook")
print("=" * 60)


# Cell 2: Data Collection
print("\n" + "=" * 60)
print("STEP 1: DATA COLLECTION")
print("=" * 60)

# Option 1: Use CCXT to fetch real data
try:
    from data.data_feeder import DataFeed, FeedConfig
    
    print("\nFetching real-time data from Binance...")
    feed_config = FeedConfig(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        limit=500
    )
    
    feed = DataFeed(feed_config)
    df = feed.fetch_ohlcv()
    
    print(f"✓ Collected {len(df)} candles")
    print(f"Date range: {df['ts'].min()} to {df['ts'].max()}")
    
except Exception as e:
    print(f"Could not fetch real data: {e}")
    print("\nGenerating synthetic data for demonstration...")
    
    # Generate synthetic data
    dates = pd.date_range(end=datetime.now(), periods=500, freq='1h')
    np.random.seed(42)
    
    # Create realistic price movement with trend
    trend = np.linspace(50000, 52000, 500)
    noise = np.random.randn(500) * 200
    prices = trend + noise
    
    df = pd.DataFrame({
        'ts': dates,
        'open': prices + np.random.randn(500) * 10,
        'high': prices + abs(np.random.randn(500) * 50),
        'low': prices - abs(np.random.randn(500) * 50),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 500)
    })
    
    # Ensure high > close > low
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    print(f"✓ Generated {len(df)} synthetic candles")

# Display data summary
print("\nDataset Summary:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn data types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())


# Cell 3: Data Cleaning and Wrangling
print("\n" + "=" * 60)
print("STEP 2: DATA CLEANING AND WRANGLING")
print("=" * 60)

# Check for missing values
missing_count = df.isnull().sum()
print(f"\nMissing values per column:")
print(missing_count[missing_count > 0])

# Handle missing values
if df.isnull().sum().sum() > 0:
    print("\nHandling missing values...")
    df = df.fillna(method='bfill').fillna(method='ffill')
    print("✓ Missing values filled")

# Check for duplicates
duplicate_count = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicate_count}")
if duplicate_count > 0:
    df = df.drop_duplicates()
    print("✓ Duplicates removed")

# Validate price data
print("\nValidating price data...")
print(f"Rows where high < low: {(df['high'] < df['low']).sum()}")
print(f"Rows with negative values: {(df[['open', 'high', 'low', 'close', 'volume']] < 0).any().sum()}")

# Remove invalid rows
initial_rows = len(df)
df = df[(df['high'] >= df['low']) & (df[['open', 'high', 'low', 'close', 'volume']] > 0).all(axis=1)]
removed_rows = initial_rows - len(df)
print(f"✓ Removed {removed_rows} invalid rows")

# Calculate basic statistics
print("\nDescriptive Statistics:")
print(df[['open', 'high', 'low', 'close', 'volume']].describe())


# Cell 4: Technical Indicators (Descriptive Method)
print("\n" + "=" * 60)
print("STEP 3: TECHNICAL INDICATORS (DESCRIPTIVE METHOD)")
print("=" * 60)

# Add simple moving averages
df['sma_5'] = df['close'].rolling(window=5).mean()
df['sma_10'] = df['close'].rolling(window=10).mean()
df['sma_20'] = df['close'].rolling(window=20).mean()
df['sma_50'] = df['close'].rolling(window=50).mean()

# Add exponential moving averages
df['ema_12'] = df['close'].ewm(span=12).mean()
df['ema_26'] = df['close'].ewm(span=26).mean()

# Add RSI (Relative Strength Index)
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi'] = calculate_rsi(df['close'])

# Add MACD (Moving Average Convergence Divergence)
df['macd'] = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9).mean()
df['macd_histogram'] = df['macd'] - df['macd_signal']

# Add Bollinger Bands
window = 20
df['bb_middle'] = df['close'].rolling(window).mean()
bb_std = df['close'].rolling(window).std()
df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
df['bb_width'] = df['bb_upper'] - df['bb_lower']

print(f"\n✓ Calculated technical indicators:")
print(f"  - Simple Moving Averages: SMA(5, 10, 20, 50)")
print(f"  - Exponential Moving Averages: EMA(12, 26)")
print(f"  - RSI (14-period)")
print(f"  - MACD (12, 26, 9)")
print(f"  - Bollinger Bands (20-period, 2 std)")

# Display indicator summary
print("\nIndicator Summary (Latest):")
latest = df.iloc[-1]
print(f"  Close Price: ${latest['close']:.2f}")
print(f"  RSI: {latest['rsi']:.2f}")
print(f"  MACD: {latest['macd']:.4f}")
print(f"  Bollinger Band Position: {((latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']) * 100):.1f}%")


# Cell 5: Statistical Analysis
print("\n" + "=" * 60)
print("STEP 4: STATISTICAL ANALYSIS")
print("=" * 60)

# Calculate returns
df['returns'] = df['close'].pct_change()
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

# Remove NaN values for analysis
df_clean = df.dropna()

print("\nReturn Statistics:")
print(df_clean['returns'].describe())

print("\nLog Return Statistics:")
print(df_clean['log_returns'].describe())

# Correlation analysis
print("\nCorrelation Matrix:")
correlation_cols = ['open', 'high', 'low', 'close', 'volume', 'returns']
corr_matrix = df_clean[correlation_cols].corr()
print(corr_matrix)

# Volatility analysis
df_clean['volatility'] = df_clean['returns'].rolling(window=20).std()
print(f"\nAverage Volatility (20-period): {df_clean['volatility'].mean():.4f}")

# Distribution analysis
print("\nPrice Distribution:")
print(f"  Mean: ${df_clean['close'].mean():.2f}")
print(f"  Median: ${df_clean['close'].median():.2f}")
print(f"  Std Dev: ${df_clean['close'].std():.2f}")
print(f"  Skewness: {df_clean['close'].skew():.2f}")
print(f"  Kurtosis: {df_clean['close'].kurtosis():.2f}")


# Cell 6: Data Visualization - Type 1: Candlestick Chart
print("\n" + "=" * 60)
print("STEP 5: DATA VISUALIZATION")
print("=" * 60)

# Visualization 1: Candlestick Chart with Indicators
fig = go.Figure()

# Candlestick
fig.add_trace(go.Candlestick(
    x=df['ts'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    name='BTC/USDT'
))

# Add moving averages
fig.add_trace(go.Scatter(
    x=df['ts'],
    y=df['sma_20'],
    name='SMA(20)',
    line=dict(color='blue', width=1)
))

fig.add_trace(go.Scatter(
    x=df['ts'],
    y=df['sma_50'],
    name='SMA(50)',
    line=dict(color='orange', width=1)
))

# Add Bollinger Bands
fig.add_trace(go.Scatter(
    x=df['ts'],
    y=df['bb_upper'],
    name='BB Upper',
    line=dict(color='red', width=1, dash='dash'),
    showlegend=True
))

fig.add_trace(go.Scatter(
    x=df['ts'],
    y=df['bb_lower'],
    name='BB Lower',
    line=dict(color='red', width=1, dash='dash'),
    fill='tonexty',
    fillcolor='rgba(255,0,0,0.1)',
    showlegend=True
))

fig.update_layout(
    title='Bitcoin (BTC/USDT) - Candlestick Chart with Technical Indicators',
    xaxis_title='Time',
    yaxis_title='Price (USDT)',
    height=600,
    template='plotly_dark'
)

fig.show()
print("✓ Visualization 1: Candlestick chart created")


# Visualization 2: Line Chart - Portfolio Value Over Time
print("\n\nCreating Line Chart: RSI and MACD Analysis")

fig2 = go.Figure()

# RSI subplot
fig2.add_trace(go.Scatter(
    x=df['ts'],
    y=df['rsi'],
    name='RSI',
    line=dict(color='purple', width=2),
    mode='lines'
))

# Add overbought/oversold lines
fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")

# MACD subplot
fig2.add_trace(go.Scatter(
    x=df['ts'],
    y=df['macd'],
    name='MACD',
    line=dict(color='orange', width=2),
    yaxis='y2'
))

fig2.add_trace(go.Scatter(
    x=df['ts'],
    y=df['macd_signal'],
    name='MACD Signal',
    line=dict(color='blue', width=2, dash='dash'),
    yaxis='y2'
))

fig2.update_layout(
    title='Technical Indicators: RSI and MACD',
    xaxis_title='Time',
    yaxis=dict(title='RSI', side='left'),
    yaxis2=dict(title='MACD', side='right', overlaying='y'),
    height=500,
    template='plotly_dark'
)

fig2.show()
print("✓ Visualization 2: RSI and MACD line chart created")


# Visualization 3: Bar Chart - Trading Signals and Market Conditions
print("\n\nCreating Bar Chart: Trading Signals Distribution")

# Calculate trading signals
df['buy_signal'] = ((df['sma_5'] > df['sma_20']) & (df['rsi'] < 70) & (df['macd'] > df['macd_signal'])).astype(int)
df['sell_signal'] = ((df['sma_5'] < df['sma_20']) & (df['rsi'] > 30) & (df['macd'] < df['macd_signal'])).astype(int)

# Count signals by condition
signal_counts = pd.DataFrame({
    'Condition': ['Buy Signals', 'Sell Signals', 'Neutral'],
    'Count': [
        df['buy_signal'].sum(),
        df['sell_signal'].sum(),
        len(df) - df['buy_signal'].sum() - df['sell_signal'].sum()
    ]
})

fig3 = go.Figure(data=[
    go.Bar(
        x=signal_counts['Condition'],
        y=signal_counts['Count'],
        marker_color=['green', 'red', 'gray'],
        text=signal_counts['Count'],
        textposition='auto'
    )
])

fig3.update_layout(
    title='Trading Signal Distribution',
    xaxis_title='Signal Type',
    yaxis_title='Count',
    height=400,
    template='plotly_dark'
)

fig3.show()
print("✓ Visualization 3: Trading signals bar chart created")


# Cell 7: Summary and Conclusions
print("\n" + "=" * 60)
print("SUMMARY AND CONCLUSIONS")
print("=" * 60)

print("\nData Exploration Complete!")
print(f"  ✓ Collected and cleaned {len(df)} data points")
print(f"  ✓ Calculated {len([col for col in df.columns if col not in ['ts', 'open', 'high', 'low', 'close', 'volume']])} technical indicators")
print(f"  ✓ Performed statistical analysis")
print(f"  ✓ Created 3 types of visualizations:")
print(f"    1. Candlestick chart with overlay indicators")
print(f"    2. Line chart with RSI and MACD")
print(f"    3. Bar chart showing trading signal distribution")

print("\nKey Findings:")
print(f"  • Average close price: ${df['close'].mean():.2f}")
print(f"  • Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
print(f"  • Average volatility: {df_clean['returns'].std():.4f}")
print(f"  • Current RSI: {df['rsi'].iloc[-1]:.2f}")
print(f"  • Trading signals generated: {df['buy_signal'].sum() + df['sell_signal'].sum()}")

print("\n✓ Data exploration notebook completed successfully!")

# Save processed data
df.to_csv('../data/processed_data.csv', index=False)
print("\n✓ Processed data saved to data/processed_data.csv")
