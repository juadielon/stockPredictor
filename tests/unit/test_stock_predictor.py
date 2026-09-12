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
            mock_yfinance.history.assert_any_call('1d', auto_adjust=False)

    def test_empty_quote_rejected(self):
        predictor = StockPredictor.__new__(StockPredictor)
        predictor.ticker = 'missing'
        with patch('app.stock_predictor.yf.Ticker') as ticker:
            ticker.return_value.history.return_value = pd.DataFrame()
            with pytest.raises(ValueError, match='No current closing price'):
                predictor.get_stock_info()

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

    @pytest.mark.parametrize('ticker', ['cba.ax', 'btc-usd'])
    def test_forecast_dates_follow_local_origin(self, ticker):
        predictor = StockPredictor.__new__(StockPredictor)
        predictor.ticker = ticker
        predictor.periods = 7
        predictor.stock_info = {
            'historical_data': pd.DataFrame(
                {'Close': [99.0, 100.0]},
                index=pd.date_range('2026-09-03', periods=2, tz='Australia/Sydney')
            )
        }
        model = MagicMock()
        model.predict.side_effect = lambda frame: frame.assign(yhat=100.0)
        with patch('app.stock_predictor.Prophet', return_value=model):
            result = predictor.make_forecast()

        dates = result['forecast']['ds']
        origin = pd.Timestamp('2026-09-04')
        assert result['historical_data']['ds'].iloc[-1] == origin
        assert (dates > origin).all()
        assert dates.iloc[-1] == pd.Timestamp('2026-09-11')
        assert len(dates) == (5 if ticker.endswith('.ax') else 7)

    @pytest.mark.parametrize('periods, expected', [
        (7, ['2026-12-29', '2026-12-30', '2026-12-31']),
        (10, ['2026-12-29', '2026-12-30', '2026-12-31']),
        (4, []),
    ])
    def test_asx_holiday_endpoints(self, periods, expected):
        predictor = StockPredictor.__new__(StockPredictor)
        predictor.ticker = 'CBA.AX'
        predictor.periods = periods
        predictor.stock_info = {'historical_data': pd.DataFrame(
            {'Close': [99.0, 100.0]},
            index=pd.date_range('2026-12-23', periods=2)
        )}
        model = MagicMock()
        model.predict.side_effect = lambda frame: frame.assign(yhat=100.0)
        with patch('app.stock_predictor.Prophet', return_value=model):
            if not expected:
                with pytest.raises(ValueError, match='No trading dates'):
                    predictor.make_forecast()
                model.predict.assert_not_called()
                return
            result = predictor.make_forecast()
        assert result['forecast']['ds'].tolist() == list(pd.to_datetime(expected))
        assert result['params_info']['elapsed_days'] == 7
        assert result['params_info']['weekday_periods'] == 3
        assert result['params_info']['requested_endpoint'] == pd.Timestamp('2026-12-24') + pd.Timedelta(days=periods)

    @pytest.mark.parametrize('ticker', ['cba.ax', 'btc-usd'])
    def test_real_prophet_forecast_smoke(self, ticker):
        predictor = StockPredictor.__new__(StockPredictor)
        predictor.ticker = ticker
        predictor.periods = 7
        dates = pd.date_range('2026-02-01', periods=80, freq='B' if ticker.endswith('.ax') else 'D')
        generator = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(generator.normal(0.1, 0.2, len(dates)))
        predictor.stock_info = {'historical_data': pd.DataFrame({'Close': prices}, index=dates)}
        result = predictor.make_forecast()
        forecast = result['forecast']
        assert (forecast['ds'] > dates[-1]).all()
        assert (forecast['ds'] <= dates[-1] + pd.Timedelta(days=7)).all()
        assert np.isfinite(forecast[['yhat', 'yhat_lower', 'yhat_upper']]).all().all()
        assert (forecast['yhat_lower'] <= forecast['yhat_upper']).all()
        assert result['params_info']['origin_price'] == prices[-1]

    def test_duplicate_session_dates_rejected(self):
        predictor = StockPredictor.__new__(StockPredictor)
        predictor.ticker = 'btc-usd'
        predictor.periods = 7
        predictor.stock_info = {'historical_data': pd.DataFrame(
            {'Close': [100.0, 101.0]}, index=pd.to_datetime(['2026-09-01', '2026-09-01'])
        )}
        with patch('app.stock_predictor.Prophet') as model:
            with pytest.raises(ValueError, match='Historical data'):
                predictor.make_forecast()
            model.assert_not_called()

    @pytest.mark.parametrize('periods', ['abc', 0, -1, 1.5, 731, True])
    def test_invalid_horizon_rejected_before_io(self, periods):
        with patch('app.stock_predictor.FanoutCache') as cache:
            with patch('app.stock_predictor.yf.Ticker') as ticker:
                with pytest.raises(ValueError, match='integer between 1 and 730'):
                    StockPredictor('cba.ax', periods)
                cache.assert_not_called()
                ticker.assert_not_called()

    @pytest.mark.parametrize('prices', [[], [100.0], [100.0, float('nan')], [100.0, 0.0]])
    def test_invalid_history_rejected_before_fit(self, prices):
        predictor = StockPredictor.__new__(StockPredictor)
        predictor.ticker = 'btc-usd'
        predictor.periods = 7
        predictor.stock_info = {'historical_data': pd.DataFrame(
            {'Close': prices}, index=pd.date_range('2026-09-01', periods=len(prices))
        )}
        with patch('app.stock_predictor.Prophet') as model:
            with pytest.raises(ValueError, match='Historical data'):
                predictor.make_forecast()
            model.assert_not_called()

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
        predictor = StockPredictor.__new__(StockPredictor)
        predictor.stock_info = {'info': {'currentPrice': 200.0}}
        returns = predictor.calculate_returns(
            pd.DataFrame({'yhat': [110.0]}),
            {'origin_price': 100.0, 'elapsed_days': 180, 'periods': 182}
        )
        assert returns['requested_period'] == pytest.approx(0.10)
        assert returns['annualised'] == pytest.approx(1.1 ** (365 / 180) - 1)

    @pytest.mark.parametrize('origin_price, endpoint_price, elapsed_days', [
        (0.0, 100.0, 30), (100.0, -1.0, 30),
        (100.0, float('nan'), 30), (100.0, float('inf'), 30),
        (100.0, 110.0, 0), (1.0, 1e100, 1),
    ])
    def test_invalid_returns_rejected(self, origin_price, endpoint_price, elapsed_days):
        predictor = StockPredictor.__new__(StockPredictor)
        with pytest.raises(ValueError):
            predictor.calculate_returns(
                pd.DataFrame({'yhat': [endpoint_price]}),
                {'origin_price': origin_price, 'elapsed_days': elapsed_days}
            )
