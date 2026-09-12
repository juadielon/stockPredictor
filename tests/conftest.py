import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from app import app as flask_app

flask_app.config['TESTING'] = True
flask_app.config['WTF_CSRF_ENABLED'] = False

@pytest.fixture
def client():
    """Flask test client with testing configuration."""
    with flask_app.test_client() as test_client:
        yield test_client

@pytest.fixture
def sample_historical_data():
    """Generates synthetic historical stock data for testing."""
    dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
    generator = np.random.default_rng(42)
    prices = 100.0 + np.cumsum(generator.normal(0.1, 1.0, size=len(dates)))
    df = pd.DataFrame({
        'Open': prices,
        'High': prices + 2.0,
        'Low': prices - 2.0,
        'Close': prices,
        'Volume': 1000000
    }, index=dates)
    return df

@pytest.fixture
def sample_stock_info(sample_historical_data):
    """Provides a realistic sample stock_info dictionary."""
    return {
        'info': {
            'symbol': 'AAPL',
            'longName': 'Apple Inc.',
            'longBusinessSummary': 'Tech company',
            'currentPrice': 150.0,
            'exchange': 'NMS',
            'currency': 'USD',
            'legalType': 'Equity',
            'dayLow': 148.0,
            'dayHigh': 152.0,
            'volume': 50000000
        },
        'dividends': pd.Series([0.22, 0.23], index=[pd.Timestamp('2023-02-10'), pd.Timestamp('2023-05-12')]),
        'historical_data': sample_historical_data,
        'now': datetime(2026, 9, 12, 12, 0)
    }
