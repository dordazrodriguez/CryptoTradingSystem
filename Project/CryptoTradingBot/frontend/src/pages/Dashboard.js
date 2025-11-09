import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Paper,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { apiService } from '../services/api';

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dashboardData, setDashboardData] = useState({
    portfolio: null,
    prices: null,
    trades: null,
    orders: null,
    performance: null,
    portfolioHistory: null,
  });

  useEffect(() => {
    // Initial load
    fetchDashboardData(false);
    // Keep polling for non-price data every 10s (background, no global spinner)
    const interval = setInterval(() => fetchDashboardData(true), 10000);

    // Live prices via SSE
    const base = process.env.REACT_APP_API_URL || window.location.origin;
    let es;
    try {
      es = new EventSource(`${base}/stream/prices`);
      es.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          // Only update if prices actually changed to avoid unnecessary re-renders
          setDashboardData((prev) => {
            const prevPrices = prev?.prices?.prices || {};
            const nextPrices = data?.prices || {};
            const sameKeys = Object.keys(prevPrices).length === Object.keys(nextPrices).length &&
              Object.keys(nextPrices).every((k) => prevPrices[k] === nextPrices[k]);
            if (sameKeys) return prev;
            return { ...prev, prices: data };
          });
        } catch (e) {
          // ignore malformed event
        }
      };
    } catch (_) {}

    return () => {
      clearInterval(interval);
      if (es) es.close();
    };
  }, []);

  const fetchDashboardData = async (background = false) => {
    try {
      if (!background) setLoading(true);
      setError(null);

      // Prices are pushed via SSE; avoid fetching here to reduce re-renders
      const [portfolioRes, tradesRes, ordersRes, performanceRes, historyRes] = await Promise.allSettled([
        apiService.getPortfolio(),
        apiService.getTrades({ limit: 50 }),
        apiService.getOrders({ limit: 100, status: 'all' }), // Get recent orders from Alpaca
        apiService.getPerformanceMetrics(),
        apiService.getPortfolioHistory({ hours: 24 }), // Get last 24 hours of history
      ]);

      setDashboardData((prev) => ({
        ...prev,
        portfolio: portfolioRes.status === 'fulfilled' ? portfolioRes.value.data : prev.portfolio,
        trades: tradesRes.status === 'fulfilled' ? tradesRes.value.data : prev.trades,
        orders: ordersRes.status === 'fulfilled' ? ordersRes.value.data : prev.orders,
        performance: performanceRes.status === 'fulfilled' ? performanceRes.value.data : prev.performance,
        portfolioHistory: historyRes.status === 'fulfilled' ? historyRes.value.data : prev.portfolioHistory,
      }));
    } catch (err) {
      setError('Failed to fetch dashboard data');
      console.error('Dashboard data fetch error:', err);
    } finally {
      if (!background) setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  // Prepare data for visualizations (fallback to performance metrics if portfolio is missing)
  const totalValue =
    dashboardData.portfolio?.portfolio?.total_value ??
    dashboardData.performance?.portfolio_metrics?.total_value ?? null;

  const totalPnl =
    dashboardData.portfolio?.portfolio?.total_pnl ??
    dashboardData.performance?.portfolio_metrics?.total_pnl ?? null;

  const winRate = dashboardData.performance?.portfolio_metrics?.win_rate ?? null;
  const totalTrades = dashboardData.performance?.portfolio_metrics?.total_trades ?? 0;

  // Use real portfolio history data if available, otherwise fallback to current value
  const portfolioValueData = dashboardData.portfolioHistory?.history && dashboardData.portfolioHistory.history.length > 0
    ? dashboardData.portfolioHistory.history.map((item) => ({
        time: item.time,
        value: item.value,
        timestamp: item.timestamp,
      }))
    : totalValue
    ? [
        // Fallback: Show current value if no history available
        { time: 'Now', value: totalValue },
      ]
    : [];

  const tradeDistributionData = dashboardData.trades?.trades
    ? dashboardData.trades.trades.reduce((acc, trade) => {
        const side = trade.side;
        acc[side] = (acc[side] || 0) + 1;
        return acc;
      }, {})
    : { buy: 0, sell: 0 };

  const pieChartData = Object.entries(tradeDistributionData).map(([side, count]) => ({
    name: side.charAt(0).toUpperCase() + side.slice(1),
    value: count,
  }));

  const COLORS = ['#00e676', '#ff4081'];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Trading Dashboard
      </Typography>

      <Grid container spacing={3}>
        {/* Portfolio Overview Cards */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Value
              </Typography>
              <Typography variant="h5" component="div">
                ${totalValue != null ? totalValue.toLocaleString() : 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total P&L
              </Typography>
              <Typography
                variant="h5"
                component="div"
                color={totalPnl != null && totalPnl >= 0 ? 'success.main' : 'error.main'}
              >
                ${totalPnl != null ? totalPnl.toLocaleString() : 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Win Rate
              </Typography>
              <Typography variant="h5" component="div">
                {winRate != null ? `${Number(winRate).toFixed(1)}%` : 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Trades
              </Typography>
              <Typography variant="h5" component="div">
                {totalTrades}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Visualization 1: Portfolio Value Line Chart */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Portfolio Value Over Time
            </Typography>
            {portfolioValueData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={portfolioValueData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip 
                    formatter={(value) => [`$${value.toLocaleString()}`, 'Value']}
                    labelFormatter={(label) => `Time: ${label}`}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#00e676"
                    strokeWidth={2}
                    dot={{ fill: '#00e676', r: 4 }}
                    name="Portfolio Value"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Typography color="textSecondary">
                  No historical data available yet. Portfolio snapshots will appear as trades are executed.
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Visualization 2: Trade Distribution Bar Chart */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Trade Distribution
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={pieChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Visualization 3: Trade Distribution Pie Chart */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Buy vs Sell Distribution
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieChartData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {pieChartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Current Prices */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Current Market Prices
            </Typography>
            <Grid container spacing={2}>
              {dashboardData.prices?.prices &&
                Object.entries(dashboardData.prices.prices).map(([symbol, price]) => (
                  <Grid item xs={6} sm={4} md={3} key={symbol}>
                    <Card variant="outlined">
                      <CardContent sx={{ p: 2 }}>
                        <Typography variant="subtitle2" color="textSecondary">
                          {symbol}
                        </Typography>
                        <Typography variant="h6">
                          ${price.toLocaleString()}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
            </Grid>
          </Paper>
        </Grid>

        {/* Recent Orders from Alpaca */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Recent Orders (Alpaca)
            </Typography>
            <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
              {dashboardData.orders?.orders?.length > 0 ? (
                dashboardData.orders.orders.slice(0, 20).map((order) => (
                  <Box
                    key={order.order_id || order.client_order_id}
                    sx={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      p: 1.5,
                      borderBottom: '1px solid #333',
                      backgroundColor: order.status === 'filled' ? 'rgba(0, 230, 118, 0.05)' : 
                                      order.status === 'partially_filled' ? 'rgba(255, 193, 7, 0.05)' :
                                      order.status === 'cancelled' ? 'rgba(255, 64, 129, 0.05)' : 'transparent',
                    }}
                  >
                    <Box sx={{ flex: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                        <Typography variant="body2" fontWeight="bold">
                          {order.symbol || 'N/A'} - {order.side ? order.side.toUpperCase() : 'N/A'}
                        </Typography>
                        <Typography 
                          variant="caption" 
                          sx={{
                            px: 1,
                            py: 0.25,
                            borderRadius: 1,
                            backgroundColor: 
                              order.status === 'filled' ? 'success.main' :
                              order.status === 'partially_filled' ? 'warning.main' :
                              order.status === 'cancelled' ? 'error.main' :
                              order.status === 'pending_new' || order.status === 'new' ? 'info.main' :
                              'text.secondary',
                            color: 'white',
                            fontWeight: 'bold',
                          }}
                        >
                          {order.status ? order.status.replace('_', ' ').toUpperCase() : 'N/A'}
                        </Typography>
                        <Typography 
                          variant="caption" 
                          color="textSecondary"
                          sx={{ ml: 1 }}
                        >
                          {order.type ? order.type.toUpperCase() : 'MARKET'}
                        </Typography>
                      </Box>
                      <Typography variant="caption" color="textSecondary">
                        {order.timestamp ? new Date(order.timestamp).toLocaleString() : 'N/A'}
                      </Typography>
                    </Box>
                    <Box textAlign="right" sx={{ ml: 2 }}>
                      <Typography variant="body2">
                        {order.filled_quantity > 0 
                          ? `${order.filled_quantity.toFixed(6)} / ${order.quantity.toFixed(6)}`
                          : `${order.quantity.toFixed(6)}`
                        }
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        {order.filled_avg_price 
                          ? `@ $${order.filled_avg_price.toFixed(2)}`
                          : order.limit_price 
                          ? `Limit: $${order.limit_price.toFixed(2)}`
                          : order.stop_price
                          ? `Stop: $${order.stop_price.toFixed(2)}`
                          : 'Market'
                        }
                      </Typography>
                      {order.filled_avg_price && order.filled_quantity > 0 && (
                        <Typography
                          variant="caption"
                          color={order.side === 'buy' ? 'success.main' : 'error.main'}
                          sx={{ display: 'block', mt: 0.5 }}
                        >
                          {order.side === 'buy' ? '-' : '+'}$
                          {(order.filled_avg_price * order.filled_quantity).toFixed(2)}
                        </Typography>
                      )}
                    </Box>
                  </Box>
                ))
              ) : dashboardData.orders?.error ? (
                <Typography color="warning.main">
                  Unable to fetch orders: {dashboardData.orders.error}
                </Typography>
              ) : (
                <Typography color="textSecondary">
                  No recent orders. Orders will appear here once placed on Alpaca.
                </Typography>
              )}
            </Box>
          </Paper>
        </Grid>

        {/* Recent Trades (from database) */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Recent Trades (Database)
            </Typography>
            <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
              {dashboardData.trades?.trades?.length > 0 ? (
                dashboardData.trades.trades.slice(0, 10).map((trade) => (
                  <Box
                    key={trade.trade_id}
                    sx={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      p: 1,
                      borderBottom: '1px solid #333',
                    }}
                  >
                    <Box>
                      <Typography variant="body2">
                        {trade.symbol} - {trade.side.toUpperCase()}
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        {trade.timestamp ? new Date(trade.timestamp).toLocaleString() : 'N/A'}
                      </Typography>
                    </Box>
                    <Box textAlign="right">
                      <Typography variant="body2">
                        {trade.quantity ? trade.quantity.toFixed(6) : 'N/A'} @ ${trade.price ? trade.price.toFixed(2) : 'N/A'}
                      </Typography>
                      <Typography
                        variant="caption"
                        color={trade.side === 'buy' ? 'success.main' : 'error.main'}
                      >
                        {trade.net_amount !== undefined ? `${trade.side === 'buy' ? '-' : '+'}$${Math.abs(trade.net_amount).toFixed(2)}` : 'N/A'}
                      </Typography>
                    </Box>
                  </Box>
                ))
              ) : (
                <Typography color="textSecondary">
                  No recent trades. Trades will appear here once executed by the trading bot.
                </Typography>
              )}
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
