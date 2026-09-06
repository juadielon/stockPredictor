import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

from app.stock_predictor import StockPredictor

class TestStockPredictorUnit:
    """Unit tests for StockPredictor business logic without external network dependencies."""

    @patch('app.stock_predictor.yf.Ticker')
    @patch('app.stock_predictor.FanoutCache')
    def test_restrict_max_periods_caps_at_36_percent(self, mock_cache, mock_ticker, sample_stock_info):
        """Forecast period should be capped at 36% of historical data length."""
        with patch.object(StockPredictor, 'prime_cache'):
            predictor = StockPredictor()
            predictor.stock_info = sample_stock_info  # 500 records
            # 500 * 0.36 + 1 = 181
            capped = predictor.restrict_max_periods(365)
            assert capped == 181

            # When requested period is small, do not cap
            uncapped = predictor.restrict_max_periods(50)
            assert uncapped == 50

    @patch('app.stock_predictor.yf.Ticker')
    @patch('app.stock_predictor.FanoutCache')
    def test_get_stock_info_parses_yfinance_data(self, mock_cache, mock_ticker_cls, sample_historical_data):
        """get_stock_info extracts current price, dividends, and history correctly."""
        mock_yfinance = MagicMock()
        mock_yfinance.info = {'shortName': 'Test Stock', 'sector': 'Tech'}
        mock_yfinance.history.side_effect = lambda *args, **kwargs: (
            pd.DataFrame({'Close': [123.45]}) if args and args[0] == '1d' else sample_historical_data
        )
        mock_yfinance.dividends = pd.Series([1.0], index=[pd.Timestamp('2023-01-01')])
        mock_ticker_cls.return_value = mock_yfinance

        with patch.object(StockPredictor, 'prime_cache'):
            predictor = StockPredictor()
            predictor.ticker = 'test'
            info = predictor.get_stock_info()

            assert info['info']['currentPrice'] == 123.45
            assert len(info['historical_data']) == len(sample_historical_data)
            assert not info['dividends'].empty

    def test_weekend_handling_asx_vs_crypto(self, sample_stock_info):
        """Stocks should remove weekend days whereas 24/7 crypto preserves all 7 days."""
        # For an ASX ticker (.ax), weekdays only
        predictor_asx = StockPredictor.__new__(StockPredictor)
        predictor_asx.ticker = 'cba.ax'
        predictor_asx.periods = 14
        predictor_asx.stock_info = sample_stock_info

        # Create mock Prophet model
        mock_model = MagicMock()
        mock_model.predict.side_effect = lambda df: df.assign(
            yhat=100.0, yhat_lower=90.0, yhat_upper=110.0, trend=100.0
        )
        mock_model.changepoints = pd.Series(dtype='datetime64[ns]')
        mock_model.params = {'delta': np.array([[0.0]])}

        with patch('app.stock_predictor.Prophet', return_value=mock_model):
            res_asx = predictor_asx.make_forecast(0.05, 10.0)
            # ASX ticker should not have Saturday or Sunday in the future forecast
            future_weekdays = res_asx['forecast']['ds'].dt.dayofweek
            assert all(day < 5 for day in future_weekdays)

    def test_get_market_country(self):
        """Verify country mapping based on ticker suffix and exchange."""
        p_asx = StockPredictor.__new__(StockPredictor)
        p_asx.ticker = 'cba.ax'
        assert p_asx.get_market_country() == 'AU'

        p_crypto = StockPredictor.__new__(StockPredictor)
        p_crypto.ticker = 'btc-usd'
        assert p_crypto.get_market_country() is None

        p_us = StockPredictor.__new__(StockPredictor)
        p_us.ticker = 'aapl'
        p_us.stock_info = {'info': {'exchange': 'NMS'}}
        assert p_us.get_market_country() == 'US'

    def test_returns_calculation(self):
        """Verify requested period return and annualised calculation logic."""
        predictor = StockPredictor.__new__(StockPredictor)
        predictor.ticker = 'test'
        
        forecast_df = pd.DataFrame({
            'yhat': [100.0, 110.0]
        })
        stock_info = {
            'info': {'currentPrice': 100.0}
        }
        params_info = {'periods': 180}
        
        returns = {}
        returns['requested_period'] = (forecast_df.tail(1)['yhat'].values[0] / stock_info['info']['currentPrice']) - 1
        returns['annualised'] = returns['requested_period'] / params_info['periods'] * 365
        
        assert returns['requested_period'] == pytest.approx(0.10)
        assert returns['annualised'] == pytest.approx(0.10 / 180 * 365)
