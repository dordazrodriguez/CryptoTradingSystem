import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  CircularProgress,
  Alert,
  Paper,
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
} from 'recharts';
import { apiService } from '../services/api';

const Analytics = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [analyticsData, setAnalyticsData] = useState({
    performance: null,
    risk: null,
    history: null,
  });

  useEffect(() => {
    fetchAnalyticsData();
    const interval = setInterval(fetchAnalyticsData, 60000); // Update every minute
    return () => clearInterval(interval);
  }, []);

  const fetchAnalyticsData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [performanceRes, riskRes, historyRes] = await Promise.allSettled([
        apiService.getPerformanceMetrics(),
        apiService.getRiskAssessment(),
        apiService.getPortfolioHistory({ hours: 24 * 7, limit: 500 }),
      ]);

      setAnalyticsData({
        performance: performanceRes.status === 'fulfilled' ? performanceRes.value.data : null,
        risk: riskRes.status === 'fulfilled' ? riskRes.value.data : null,
        history: historyRes.status === 'fulfilled' ? historyRes.value.data : null,
      });
    } catch (err) {
      setError('Failed to fetch analytics data');
      console.error('Analytics fetch error:', err);
    } finally {
      setLoading(false);
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

  const performance = analyticsData.performance?.portfolio_metrics;
  const risk = analyticsData.risk?.risk_assessment;

  // Build performance history from backend portfolio-history
  const rawHistory = analyticsData.history?.history || [];
  // Compute normalized cumulative returns based on value changes
  const performanceHistory = (() => {
    if (rawHistory.length < 2) return [];
    const start = rawHistory[0].value || 0;
    if (!start) return [];
    return rawHistory.map((pt) => ({
      date: pt.time,
      // return as fraction (e.g., 0.0123)
      return: (pt.value - start) / start,
    }));
  })();

  // Aggregate monthly returns from history
  const monthlyReturns = (() => {
    if (rawHistory.length < 2) return [];
    const byMonth = {};
    const firstValByMonth = {};
    const lastValByMonth = {};
    rawHistory.forEach((pt) => {
      const monthKey = (pt.timestamp || '').slice(0, 7); // YYYY-MM
      if (!firstValByMonth[monthKey]) firstValByMonth[monthKey] = pt.value;
      lastValByMonth[monthKey] = pt.value;
    });
    Object.keys(firstValByMonth).forEach((m) => {
      const startV = firstValByMonth[m];
      const endV = lastValByMonth[m];
      if (startV) {
        byMonth[m] = (endV - startV) / startV; // fraction
      }
    });
    // Convert to chart-friendly format and keep chronological order
    return Object.keys(byMonth)
      .sort()
      .map((m) => ({ month: m.split('-')[1], return: byMonth[m] }));
  })();

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Performance Analytics
      </Typography>

      <Grid container spacing={3}>
        {/* Key Metrics Cards */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Return
              </Typography>
              <Typography
                variant="h5"
                component="div"
                color={performance?.total_return >= 0 ? 'success.main' : 'error.main'}
              >
                {performance?.total_return !== undefined && performance?.total_return !== null
                  ? `${Number(performance.total_return).toFixed(2)}%`
                  : 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Sharpe Ratio
              </Typography>
              <Typography variant="h5" component="div">
                {performance?.sharpe_ratio?.toFixed(2) || 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Max Drawdown
              </Typography>
              <Typography variant="h5" component="div" color="error.main">
                {performance?.max_drawdown
                  ? `${(performance.max_drawdown * 100).toFixed(2)}%`
                  : 'N/A'}
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
                {performance?.win_rate !== undefined && performance?.win_rate !== null
                  ? `${Number(performance.win_rate).toFixed(1)}%`
                  : 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Chart */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Cumulative Returns
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={performanceHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip formatter={(value) => [`${(Number(value) * 100).toFixed(2)}%`, 'Return']} />
                <Line
                  type="monotone"
                  dataKey="return"
                  stroke="#00e676"
                  strokeWidth={2}
                  dot={{ fill: '#00e676' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Monthly Returns */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Monthly Returns
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={monthlyReturns}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip formatter={(value) => [`${(Number(value) * 100).toFixed(2)}%`, 'Return']} />
                <Bar dataKey="return" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Risk Assessment */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Risk Assessment
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography color="textSecondary" variant="body2">
                    Portfolio VaR
                  </Typography>
                  <Typography variant="h6">
                    ${risk?.portfolio_var?.toFixed(2) || 'N/A'}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography color="textSecondary" variant="body2">
                    Max Position %
                  </Typography>
                  <Typography variant="h6">
                    {risk?.max_position_percent
                      ? `${(risk.max_position_percent * 100).toFixed(1)}%`
                      : 'N/A'}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography color="textSecondary" variant="body2">
                    Risk Level
                  </Typography>
                  <Typography
                    variant="h6"
                    color={
                      risk?.overall_risk_status?.risk_within_limits
                        ? 'success.main'
                        : 'error.main'
                    }
                  >
                    {risk?.overall_risk_status?.risk_within_limits ? 'Low' : 'High'}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography color="textSecondary" variant="body2">
                    Can Trade
                  </Typography>
                  <Typography
                    variant="h6"
                    color={
                      risk?.overall_risk_status?.can_trade
                        ? 'success.main'
                        : 'error.main'
                    }
                  >
                    {risk?.overall_risk_status?.can_trade ? 'Yes' : 'No'}
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Trade Statistics */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Trade Statistics
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography color="textSecondary" variant="body2">
                    Total Trades
                  </Typography>
                  <Typography variant="h6">
                    {performance?.total_trades || 0}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography color="textSecondary" variant="body2">
                    Winning Trades
                  </Typography>
                  <Typography variant="h6" color="success.main">
                    {performance?.winning_trades || 0}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography color="textSecondary" variant="body2">
                    Losing Trades
                  </Typography>
                  <Typography variant="h6" color="error.main">
                    {performance?.losing_trades || 0}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography color="textSecondary" variant="body2">
                    Profit Factor
                  </Typography>
                  <Typography variant="h6">
                    {performance?.profit_factor?.toFixed(2) || 'N/A'}
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Analytics;
