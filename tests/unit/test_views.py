import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

def test_home_page_get(client):
    """Test that the home page loads successfully with the form."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Stock Price Prediction' in response.data
    assert b'Ticker Symbol' in response.data

def test_ticker_post_invalid_redirects_home(client):
    """Submitting empty or invalid form should redirect to home."""
    response = client.post('/ticker', data={}, follow_redirects=False)
    assert response.status_code == 302
    assert response.headers['Location'] in ['/', 'http://localhost/']

@patch('app.views.StockPredictor')
def test_ticker_post_valid_renders_results(mock_predictor_cls, client):
    """Submitting a valid ticker invokes StockPredictor and displays results."""
    mock_instance = MagicMock()
    
    forecast_df = pd.DataFrame({
        'ds': [pd.Timestamp('2026-01-01')],
        'yhat': [150.0],
        'yhat_lower': [140.0],
        'yhat_upper': [160.0],
        'delta': [0.01]
    })
    
    performance_df = pd.DataFrame({
        'horizon': [pd.Timedelta(days=30)],
        'mape': [0.05]
    })
    
    mock_instance.result = {
        'stock_info': {
            'info': {
                'symbol': 'AAPL',
                'longName': 'Apple Inc.',
                'longBusinessSummary': 'Technology company.',
                'currentPrice': 150.0,
                'exchange': 'NASDAQ',
                'currency': 'USD',
                'legalType': 'Equity',
                'dayLow': 148.0,
                'dayHigh': 152.0,
                'volume': 1000000
            },
            'now': pd.Timestamp('2026-09-06'),
            'dividends': pd.Series(dtype=float)
        },
        'params_info': {
            'periods': 365,
            'historical_periods': 1000,
            'weekday_periods': 260,
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0
        },
        'forecast': forecast_df,
        'performance': performance_df,
        'returns': {
            'requested_period': 0.15,
            'annualised': 0.15
        },
        'fig_paths': {
            'plotly_price': '{}',
            'plotly_components': '{}',
            'plotly_mape': '{}'
        }
    }
    mock_predictor_cls.return_value = mock_instance

    response = client.post('/ticker', data={'ticker': 'AAPL', 'days': '365'})
    assert response.status_code == 200
    assert b'Apple Inc.' in response.data
    assert b'AAPL' in response.data

@patch('app.views.StockPredictor')
def test_preload_endpoint(mock_predictor_cls, client):
    """Test that the /preload endpoint calls StockPredictor.preload()."""
    mock_instance = MagicMock()
    mock_predictor_cls.return_value = mock_instance

    response = client.get('/preload')
    assert response.status_code == 200
    assert b'Working ... Check the container logs' in response.data
    mock_instance.preload.assert_called_once()
