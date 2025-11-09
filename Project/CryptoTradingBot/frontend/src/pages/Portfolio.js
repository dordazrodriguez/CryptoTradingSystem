import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  CircularProgress,
  Alert,
} from '@mui/material';
import { apiService } from '../services/api';

const Portfolio = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [portfolioData, setPortfolioData] = useState(null);

  useEffect(() => {
    fetchPortfolioData();
    const interval = setInterval(fetchPortfolioData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchPortfolioData = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.getPortfolio();
      setPortfolioData(response.data);
    } catch (err) {
      setError('Failed to fetch portfolio data');
      console.error('Portfolio fetch error:', err);
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

  const portfolio = portfolioData?.portfolio;
  const performance = portfolioData?.performance;

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Portfolio Overview
      </Typography>

      <Grid container spacing={3}>
        {/* Portfolio Summary Cards */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Value
              </Typography>
              <Typography variant="h5" component="div">
                ${portfolio?.total_value?.toLocaleString() || 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Cash Balance
              </Typography>
              <Typography variant="h5" component="div">
                ${portfolio?.cash_balance?.toLocaleString() || 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Invested Value
              </Typography>
              <Typography variant="h5" component="div">
                ${portfolio?.invested_value?.toLocaleString() || 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Return
              </Typography>
              <Typography
                variant="h5"
                component="div"
                color={portfolio?.total_return >= 0 ? 'success.main' : 'error.main'}
              >
                {portfolio?.total_return
                  ? `${(portfolio.total_return * 100).toFixed(2)}%`
                  : 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Positions Table */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Positions
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Symbol</TableCell>
                      <TableCell align="right">Quantity</TableCell>
                      <TableCell align="right">Avg Cost</TableCell>
                      <TableCell align="right">Current Price</TableCell>
                      <TableCell align="right">Market Value</TableCell>
                      <TableCell align="right">Unrealized P&L</TableCell>
                      <TableCell align="right">P&L %</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {portfolio?.positions && Object.keys(portfolio.positions).length > 0 ? (
                      Object.entries(portfolio.positions).map(([symbol, position]) => (
                        <TableRow key={symbol}>
                          <TableCell>
                            <Chip label={symbol} size="small" />
                          </TableCell>
                          <TableCell align="right">{position.quantity.toFixed(6)}</TableCell>
                          <TableCell align="right">${position.avg_cost.toFixed(2)}</TableCell>
                          <TableCell align="right">${position.current_price?.toFixed(2) || 'N/A'}</TableCell>
                          <TableCell align="right">
                            ${position.market_value?.toFixed(2) || 'N/A'}
                          </TableCell>
                          <TableCell
                            align="right"
                            sx={{
                              color: position.unrealized_pnl >= 0 ? 'success.main' : 'error.main',
                            }}
                          >
                            ${position.unrealized_pnl?.toFixed(2) || 'N/A'}
                          </TableCell>
                          <TableCell
                            align="right"
                            sx={{
                              color: position.unrealized_pnl >= 0 ? 'success.main' : 'error.main',
                            }}
                          >
                            {position.unrealized_pnl
                              ? `${((position.unrealized_pnl / (position.quantity * position.avg_cost)) * 100).toFixed(2)}%`
                              : 'N/A'}
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={7} align="center">
                          <Typography color="textSecondary">
                            No current positions
                          </Typography>
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Metrics */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Metrics
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6} sm={3}>
                  <Box textAlign="center">
                    <Typography color="textSecondary" variant="body2">
                      Win Rate
                    </Typography>
                    <Typography variant="h6">
                      {performance?.win_rate
                        ? `${(performance.win_rate * 100).toFixed(1)}%`
                        : 'N/A'}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Box textAlign="center">
                    <Typography color="textSecondary" variant="body2">
                      Total Trades
                    </Typography>
                    <Typography variant="h6">
                      {performance?.total_trades || 0}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Box textAlign="center">
                    <Typography color="textSecondary" variant="body2">
                      Sharpe Ratio
                    </Typography>
                    <Typography variant="h6">
                      {performance?.sharpe_ratio?.toFixed(2) || 'N/A'}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Box textAlign="center">
                    <Typography color="textSecondary" variant="body2">
                      Max Drawdown
                    </Typography>
                    <Typography variant="h6" color="error.main">
                      {performance?.max_drawdown
                        ? `${(performance.max_drawdown * 100).toFixed(2)}%`
                        : 'N/A'}
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Portfolio;
