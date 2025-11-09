import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Divider,
  Alert,
  CircularProgress,
} from '@mui/material';
import { apiService } from '../services/api';

const Settings = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [settings, setSettings] = useState({
    // Trading settings
    maxPositionPercent: 0.2,
    stopLossPercent: 0.05,
    maxLeverage: 1.0,
    dailyLossLimit: 0.02,
    maxDrawdownLimit: 0.15,
    
    // ML settings
    mlModelEnabled: true,
    mlConfidenceThreshold: 0.6,
    
    // Risk settings
    riskManagementEnabled: true,
    autoStopLoss: true,
    
    // API settings
    apiKey: process.env.REACT_APP_ALPACA_API_KEY || '',
    secretKey: process.env.REACT_APP_ALPACA_SECRET_KEY || '',
    paperTrading: String(process.env.REACT_APP_PAPER_TRADING || 'true') === 'true',
  });

  const handleSettingChange = (field, value) => {
    setSettings(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleSaveSettings = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);

      // In a real implementation, you would send these settings to the backend
      // For now, we'll just simulate a save operation
      await new Promise(resolve => setTimeout(resolve, 1000));

      setSuccess('Settings saved successfully!');
    } catch (err) {
      setError('Failed to save settings');
      console.error('Settings save error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleTestConnection = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await apiService.healthCheck();
      
      if (response.data.status === 'healthy') {
        setSuccess('Connection test successful!');
      } else {
        setError('Connection test failed');
      }
    } catch (err) {
      setError('Connection test failed');
      console.error('Connection test error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2 }}>
          {success}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Trading Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Trading Settings
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Max Position %"
                    type="number"
                    value={settings.maxPositionPercent}
                    onChange={(e) => handleSettingChange('maxPositionPercent', parseFloat(e.target.value))}
                    inputProps={{ min: 0, max: 1, step: 0.01 }}
                    helperText="Maximum percentage of portfolio per position"
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Stop Loss %"
                    type="number"
                    value={settings.stopLossPercent}
                    onChange={(e) => handleSettingChange('stopLossPercent', parseFloat(e.target.value))}
                    inputProps={{ min: 0, max: 1, step: 0.01 }}
                    helperText="Default stop loss percentage"
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Max Leverage"
                    type="number"
                    value={settings.maxLeverage}
                    onChange={(e) => handleSettingChange('maxLeverage', parseFloat(e.target.value))}
                    inputProps={{ min: 1, max: 10, step: 0.1 }}
                    helperText="Maximum leverage allowed"
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Daily Loss Limit %"
                    type="number"
                    value={settings.dailyLossLimit}
                    onChange={(e) => handleSettingChange('dailyLossLimit', parseFloat(e.target.value))}
                    inputProps={{ min: 0, max: 1, step: 0.01 }}
                    helperText="Maximum daily loss percentage"
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Max Drawdown Limit %"
                    type="number"
                    value={settings.maxDrawdownLimit}
                    onChange={(e) => handleSettingChange('maxDrawdownLimit', parseFloat(e.target.value))}
                    inputProps={{ min: 0, max: 1, step: 0.01 }}
                    helperText="Maximum drawdown percentage"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* ML Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Machine Learning Settings
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.mlModelEnabled}
                        onChange={(e) => handleSettingChange('mlModelEnabled', e.target.checked)}
                      />
                    }
                    label="Enable ML Model"
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="ML Confidence Threshold"
                    type="number"
                    value={settings.mlConfidenceThreshold}
                    onChange={(e) => handleSettingChange('mlConfidenceThreshold', parseFloat(e.target.value))}
                    inputProps={{ min: 0, max: 1, step: 0.1 }}
                    helperText="Minimum confidence for ML signals"
                    disabled={!settings.mlModelEnabled}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Risk Management */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Risk Management
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.riskManagementEnabled}
                        onChange={(e) => handleSettingChange('riskManagementEnabled', e.target.checked)}
                      />
                    }
                    label="Enable Risk Management"
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.autoStopLoss}
                        onChange={(e) => handleSettingChange('autoStopLoss', e.target.checked)}
                      />
                    }
                    label="Auto Stop Loss"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* API Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                API Settings
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="API Key"
                    type="password"
                    value={settings.apiKey}
                    onChange={(e) => handleSettingChange('apiKey', e.target.value)}
                    helperText="Alpaca API Key"
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Secret Key"
                    type="password"
                    value={settings.secretKey}
                    onChange={(e) => handleSettingChange('secretKey', e.target.value)}
                    helperText="Alpaca Secret Key"
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.paperTrading}
                        onChange={(e) => handleSettingChange('paperTrading', e.target.checked)}
                      />
                    }
                    label="Paper Trading"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Action Buttons */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box display="flex" gap={2}>
                <Button
                  variant="contained"
                  onClick={handleSaveSettings}
                  disabled={loading}
                  startIcon={loading ? <CircularProgress size={20} /> : null}
                >
                  Save Settings
                </Button>
                
                <Button
                  variant="outlined"
                  onClick={handleTestConnection}
                  disabled={loading}
                  startIcon={loading ? <CircularProgress size={20} /> : null}
                >
                  Test Connection
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Settings;
