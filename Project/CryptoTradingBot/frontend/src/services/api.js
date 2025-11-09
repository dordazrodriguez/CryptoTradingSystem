import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:5000',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth headers
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    const apiKey = localStorage.getItem('apiKey');
    if (apiKey) {
      config.headers['X-API-Key'] = apiKey;
    }
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling errors
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API service methods
export const apiService = {
  // Health check
  healthCheck: () => api.get('/health'),

  // Portfolio endpoints
  getPortfolio: () => api.get('/api/portfolio'),
  getPositions: () => api.get('/api/portfolio/positions'),

  // Trade endpoints
  getTrades: (params = {}) => api.get('/api/trades', { params }),
  getTrade: (tradeId) => api.get(`/api/trades/${tradeId}`),
  
  // Order endpoints (from Alpaca)
  getOrders: (params = {}) => api.get('/api/orders', { params }),

  // Market data endpoints
  getCurrentPrices: () => api.get('/api/market/prices'),
  getPriceHistory: (symbol, params = {}) => api.get(`/api/market/history/${symbol}`, { params }),

  // Technical indicators
  getIndicators: (symbol, params = {}) => api.get(`/api/indicators/${symbol}`, { params }),

  // ML predictions
  getPredictions: (symbol, params = {}) => api.get(`/api/predictions/${symbol}`, { params }),

  // Trading signals
  getTradingSignals: (symbol) => api.get(`/api/signals/${symbol}`),

  // Risk management
  getRiskAssessment: () => api.get('/api/risk/assessment'),

  // Performance metrics
  getPerformanceMetrics: () => api.get('/api/metrics/performance'),
  
  // Portfolio history for charts
  getPortfolioHistory: (params = {}) => api.get('/api/metrics/portfolio-history', { params }),

  // System logs
  getSystemLogs: (params = {}) => api.get('/api/logs', { params }),
};

export default api;
