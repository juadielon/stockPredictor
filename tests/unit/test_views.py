import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

def test_home_page_get(client):
    """Test that the home page loads successfully with the form."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Stock Price Prediction' in response.data
    assert b'Ticker Symbol' in response.data

def test_ticker_post_invalid_shows_errors(client):
    response = client.post('/ticker', data={}, follow_redirects=False)
    assert response.status_code == 400
    assert b'This field is required' in response.data

@patch('app.views.StockPredictor')
def test_invalid_horizon_does_not_forecast(mock_predictor_cls, client):
    response = client.post('/ticker', data={'ticker': 'NDQ.AX', 'days': '-1'})
    assert response.status_code == 400
    mock_predictor_cls.assert_not_called()

@patch('app.views.StockPredictor', side_effect=ValueError('No trading dates'))
def test_forecast_failure_shows_controlled_error(mock_predictor_cls, client):
    response = client.post('/ticker', data={'ticker': 'NDQ.AX', 'days': '1'})
    assert response.status_code == 422
    assert b'Unable to forecast this ticker and period' in response.data
    mock_predictor_cls.assert_called_once_with('ndq.ax', 1)

@pytest.mark.parametrize('stale', [False, True])
@patch('app.views.StockPredictor')
def test_ticker_post_valid_renders_results(mock_predictor_cls, client, stale):
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
            'origin': pd.Timestamp('2025-01-01'),
            'forecast_endpoint': pd.Timestamp('2026-01-01'),
            'elapsed_days': 365,
            'historical_periods': 1000,
            'weekday_periods': 260,
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0
        },
        'forecast': forecast_df,
        'cache_info': {
            'generated_at_label': '12/09/2026 00:00 UTC',
            'data_cutoff': '2026-09-11',
            'parameters': {'source': 'legacy'},
            'stale': stale,
            'warning': 'Refresh unavailable. Showing the last saved forecast.',
        },
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
    assert b'Last observation: 01/01/2025' in response.data
    assert b'Forecast endpoint: 01/01/2026' in response.data
    assert b'Adjusted-price return estimate, not total investment return' in response.data
    assert b'Forecast saved: 12/09/2026 00:00 UTC' in response.data
    assert b'Model settings: legacy' in response.data
    assert (b'Refresh unavailable' in response.data) == stale

@patch('app.views.StockPredictor')
def test_preload_endpoint(mock_predictor_cls, client):
    response = client.get('/preload')
    assert response.status_code == 410
    assert b'command line' in response.data
    mock_predictor_cls.assert_not_called()


@patch('app.views.StockPredictor', side_effect=RuntimeError('provider unavailable'))
def test_provider_error_is_controlled(mock_predictor_cls, client):
    response = client.post('/ticker', data={'ticker': 'ndq.ax', 'days': '90'})
    assert response.status_code == 503
    assert b'Forecast temporarily unavailable' in response.data


@patch('app.preload.StockPredictor')
def test_preload_command_summary_and_options(mock_predictor_cls):
    from app import app
    predictor = mock_predictor_cls.return_value
    predictor.preload.return_value = {'succeeded': ['ndq.ax'], 'failed': []}
    result = app.test_cli_runner().invoke(args=['preload', '--days', '90', '--ticker', 'ndq.ax', '--retune'])
    assert result.exit_code == 0
    assert '1 refreshed, 0 failed' in result.output
    predictor.preload.assert_called_once_with(90, retune=True, tickers=('ndq.ax',))
    predictor.cache.close.assert_called_once()


@patch('app.preload.StockPredictor')
def test_preload_command_partial_failure_exit(mock_predictor_cls):
    from app import app
    mock_predictor_cls.return_value.preload.return_value = {'succeeded': ['ndq.ax'], 'failed': ['bad']}
    result = app.test_cli_runner().invoke(args=['preload'])
    assert result.exit_code == 1
    assert '1 refreshed, 1 failed' in result.output
    assert 'Failed tickers: bad' in result.output


@patch('app.preload.StockPredictor')
def test_preload_invalid_horizon_does_not_open_cache(mock_predictor_cls):
    from app import app
    result = app.test_cli_runner().invoke(args=['preload', '--days', '0'])
    assert result.exit_code != 0
    mock_predictor_cls.assert_not_called()
