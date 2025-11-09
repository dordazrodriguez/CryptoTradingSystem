import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
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
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Tabs,
  Tab,
} from '@mui/material';
import { apiService } from '../services/api';

const Trades = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [trades, setTrades] = useState([]);
  const [orders, setOrders] = useState([]);
  const [activeTab, setActiveTab] = useState(0); // 0 = Database Trades, 1 = Alpaca Orders
  const [filters, setFilters] = useState({
    symbol: '',
    limit: 100,
    start_date: '',
    end_date: '',
  });
  const [orderFilters, setOrderFilters] = useState({
    limit: 100,
    status: 'all',
    symbol: '',
    side: '',
  });

  useEffect(() => {
    fetchTrades();
    fetchOrders();
  }, [filters, orderFilters]);

  const fetchTrades = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const params = Object.fromEntries(
        Object.entries(filters).filter(([_, value]) => value !== '')
      );
      
      const response = await apiService.getTrades(params);
      setTrades(response.data.trades || []);
    } catch (err) {
      setError('Failed to fetch trades');
      console.error('Trades fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchOrders = async () => {
    try {
      const params = Object.fromEntries(
        Object.entries(orderFilters).filter(([_, value]) => value !== '' && value !== 'all')
      );
      
      const response = await apiService.getOrders(params);
      setOrders(response.data.orders || []);
    } catch (err) {
      console.error('Orders fetch error:', err);
      // Don't set error state for orders - it's optional
    }
  };

  const handleFilterChange = (field, value) => {
    setFilters(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleOrderFilterChange = (field, value) => {
    setOrderFilters(prev => ({
      ...prev,
      [field]: value,
    }));
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

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Trade History
      </Typography>

      {/* Tabs for Database Trades vs Alpaca Orders */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)}>
          <Tab label={`Database Trades (${trades.length})`} />
          <Tab label={`Alpaca Orders (${orders.length})`} />
        </Tabs>
      </Paper>

      {/* Database Trades Section */}
      {activeTab === 0 && (
        <>
          {/* Filters for Database Trades */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Filters
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <TextField
                    fullWidth
                    label="Symbol"
                    value={filters.symbol}
                    onChange={(e) => handleFilterChange('symbol', e.target.value)}
                    placeholder="e.g., BTC/USD"
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <FormControl fullWidth>
                    <InputLabel>Limit</InputLabel>
                    <Select
                      value={filters.limit}
                      onChange={(e) => handleFilterChange('limit', e.target.value)}
                    >
                      <MenuItem value={50}>50</MenuItem>
                      <MenuItem value={100}>100</MenuItem>
                      <MenuItem value={200}>200</MenuItem>
                      <MenuItem value={500}>500</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <TextField
                    fullWidth
                    label="Start Date"
                    type="date"
                    value={filters.start_date}
                    onChange={(e) => handleFilterChange('start_date', e.target.value)}
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <TextField
                    fullWidth
                    label="End Date"
                    type="date"
                    value={filters.end_date}
                    onChange={(e) => handleFilterChange('end_date', e.target.value)}
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Database Trades Table */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Database Trades ({trades.length})
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Trade ID</TableCell>
                      <TableCell>Symbol</TableCell>
                      <TableCell>Side</TableCell>
                      <TableCell align="right">Quantity</TableCell>
                      <TableCell align="right">Price</TableCell>
                      <TableCell align="right">Commission</TableCell>
                      <TableCell align="right">Net Amount</TableCell>
                      <TableCell>Timestamp</TableCell>
                      <TableCell>Strategy</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {trades.length > 0 ? (
                      trades.map((trade) => (
                        <TableRow key={trade.trade_id}>
                          <TableCell>
                            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                              {trade.trade_id?.slice(-8) || 'N/A'}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Chip label={trade.symbol} size="small" />
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={trade.side?.toUpperCase() || 'N/A'}
                              size="small"
                              color={trade.side === 'buy' ? 'success' : 'error'}
                            />
                          </TableCell>
                          <TableCell align="right">
                            {trade.quantity?.toFixed(6) || 'N/A'}
                          </TableCell>
                          <TableCell align="right">
                            ${trade.price?.toFixed(2) || 'N/A'}
                          </TableCell>
                          <TableCell align="right">
                            ${trade.commission?.toFixed(2) || '0.00'}
                          </TableCell>
                          <TableCell align="right">
                            <Typography
                              variant="body2"
                              color={trade.side === 'buy' ? 'error.main' : 'success.main'}
                            >
                              {trade.side === 'buy' ? '-' : '+'}${Math.abs(trade.net_amount || 0).toFixed(2)}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2">
                              {trade.timestamp ? new Date(trade.timestamp).toLocaleString() : 'N/A'}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={trade.strategy || 'Manual'}
                              size="small"
                              variant="outlined"
                            />
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={9} align="center">
                          <Typography color="textSecondary">
                            No trades found
                          </Typography>
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </>
      )}

      {/* Alpaca Orders Section */}
      {activeTab === 1 && (
        <>
          {/* Filters for Alpaca Orders */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Filters
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <TextField
                    fullWidth
                    label="Symbol"
                    value={orderFilters.symbol}
                    onChange={(e) => handleOrderFilterChange('symbol', e.target.value)}
                    placeholder="e.g., BTCUSD"
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <FormControl fullWidth>
                    <InputLabel>Status</InputLabel>
                    <Select
                      value={orderFilters.status}
                      onChange={(e) => handleOrderFilterChange('status', e.target.value)}
                    >
                      <MenuItem value="all">All</MenuItem>
                      <MenuItem value="filled">Filled</MenuItem>
                      <MenuItem value="partially_filled">Partially Filled</MenuItem>
                      <MenuItem value="pending_new">Pending</MenuItem>
                      <MenuItem value="cancelled">Cancelled</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <FormControl fullWidth>
                    <InputLabel>Side</InputLabel>
                    <Select
                      value={orderFilters.side}
                      onChange={(e) => handleOrderFilterChange('side', e.target.value)}
                    >
                      <MenuItem value="">All</MenuItem>
                      <MenuItem value="buy">Buy</MenuItem>
                      <MenuItem value="sell">Sell</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <FormControl fullWidth>
                    <InputLabel>Limit</InputLabel>
                    <Select
                      value={orderFilters.limit}
                      onChange={(e) => handleOrderFilterChange('limit', e.target.value)}
                    >
                      <MenuItem value={50}>50</MenuItem>
                      <MenuItem value={100}>100</MenuItem>
                      <MenuItem value={200}>200</MenuItem>
                      <MenuItem value={500}>500</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Alpaca Orders Table */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Alpaca Orders ({orders.length})
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Order ID</TableCell>
                      <TableCell>Symbol</TableCell>
                      <TableCell>Side</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell align="right">Quantity</TableCell>
                      <TableCell align="right">Filled Qty</TableCell>
                      <TableCell align="right">Avg Fill Price</TableCell>
                      <TableCell align="right">Limit Price</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Submitted At</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {orders.length > 0 ? (
                      orders.map((order) => (
                        <TableRow key={order.order_id || order.client_order_id}>
                          <TableCell>
                            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                              {order.order_id ? order.order_id.slice(-12) : order.client_order_id?.slice(-12) || 'N/A'}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Chip label={order.symbol || 'N/A'} size="small" />
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={order.side?.toUpperCase() || 'N/A'}
                              size="small"
                              color={order.side === 'buy' ? 'success' : 'error'}
                            />
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={order.type?.toUpperCase() || 'MARKET'}
                              size="small"
                              variant="outlined"
                            />
                          </TableCell>
                          <TableCell align="right">
                            {order.quantity?.toFixed(6) || 'N/A'}
                          </TableCell>
                          <TableCell align="right">
                            {order.filled_quantity > 0 ? order.filled_quantity.toFixed(6) : '0.000000'}
                          </TableCell>
                          <TableCell align="right">
                            {order.filled_avg_price ? `$${order.filled_avg_price.toFixed(2)}` : '-'}
                          </TableCell>
                          <TableCell align="right">
                            {order.limit_price ? `$${order.limit_price.toFixed(2)}` : '-'}
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={order.status?.replace('_', ' ').toUpperCase() || 'N/A'}
                              size="small"
                              color={
                                order.status === 'filled' ? 'success' :
                                order.status === 'partially_filled' ? 'warning' :
                                order.status === 'cancelled' ? 'error' :
                                'default'
                              }
                            />
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2">
                              {order.timestamp ? new Date(order.timestamp).toLocaleString() : 
                               order.created_at ? new Date(order.created_at).toLocaleString() : 'N/A'}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={10} align="center">
                          <Typography color="textSecondary">
                            No orders found from Alpaca
                          </Typography>
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </>
      )}
    </Box>
  );
};

export default Trades;
