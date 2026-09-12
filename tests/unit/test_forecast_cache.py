from unittest.mock import patch

import pandas as pd
import pytest
from filelock import Timeout

from app.forecast_cache import ForecastCache
from app.stock_predictor import StockPredictor


def test_result_reuse_expiry_and_version(tmp_path):
    cache = ForecastCache(tmp_path)
    with patch('app.forecast_cache.time.time', return_value=1000):
        cache.save_result(' NDQ.AX ', 90, {'cache_info': {}, 'forecast': 'saved'})
    assert cache.result('ndq.ax', 180) is None
    with patch('app.forecast_cache.time.time', return_value=1001):
        assert cache.result('ndq.ax', 90)['forecast'] == 'saved'
    with patch('app.forecast_cache.time.time', return_value=1000 + cache.forecast_lifetime):
        assert cache.result('ndq.ax', 90) is None
        assert cache.result('ndq.ax', 90, allow_stale=True)['cache_info']['stale']
    cache.model_version = 'future-model'
    assert cache.result('ndq.ax', 90, allow_stale=True) is None
    cache.close()


def test_legacy_import_does_not_renew_expiry(tmp_path):
    entries = [{'ticker': 'NDQ.AX', 'changepoint_prior_scale': 0.05}]
    cache = ForecastCache(tmp_path)
    with patch('app.forecast_cache.time.time', return_value=1000):
        cache.import_legacy(entries)
        saved = cache.parameters('ndq.ax', None)
        assert saved['tuned_at'] is None
        assert saved['source'] == 'legacy'
    cache.close()
    reopened = ForecastCache(tmp_path)
    with patch('app.forecast_cache.time.time', return_value=1000 + cache.parameter_lifetime):
        reopened.import_legacy(entries)
        assert reopened.parameters('ndq.ax', None) is None
    reopened.close()


def test_lock_rejects_duplicate_work_and_releases(tmp_path):
    cache = ForecastCache(tmp_path)
    with cache.lock('NDQ.AX', 90):
        with pytest.raises(Timeout):
            with cache.lock('ndq.ax', 90):
                pass
        with cache.lock('btc-usd', 90):
            pass
    with cache.lock('ndq.ax', 90):
        pass
    cache.close()


def test_remembered_tickers_preserve_json(tmp_path):
    cache = ForecastCache(tmp_path / 'cache')
    path = tmp_path / 'tickers.json'
    original = '[{"ticker": "NDQ.AX"}]'
    path.write_text(original)
    cache.remember_ticker(' BTC-USD ')
    assert cache.tickers(path) == ['btc-usd', 'ndq.ax']
    assert path.read_text() == original
    cache.close()


def test_predictor_reuses_result_without_network_or_model(tmp_path):
    cache = ForecastCache(tmp_path / 'cache')
    cache.save_result('ndq.ax', 90, {'cache_info': {}, 'forecast': 'saved'})
    cache.close()
    with patch.object(StockPredictor, 'get_stock_info') as fetch:
        with patch.object(StockPredictor, 'forecaster') as forecast:
            predictor = StockPredictor(' NDQ.AX ', 90, cache_directory=tmp_path / 'cache',
                                       ticker_file=str(tmp_path / 'missing.json'))
    assert predictor.result['forecast'] == 'saved'
    fetch.assert_not_called()
    forecast.assert_not_called()


def test_failed_batch_continues_and_preserves_previous_result(tmp_path):
    predictor = StockPredictor(cache_directory=tmp_path / 'cache',
                               ticker_file=str(tmp_path / 'missing.json'))
    predictor.cache.save_result('bad', 90, {'cache_info': {}, 'forecast': 'previous'})
    with patch.object(predictor, 'get_stock_info', side_effect=ValueError('provider unavailable')):
        summary = predictor.preload(90, tickers=['bad', 'good'])
    assert summary == {'succeeded': [], 'failed': ['bad', 'good']}
    assert predictor.cache.result('bad', 90)['forecast'] == 'previous'
    predictor.cache.close()


def test_failed_request_returns_labelled_previous_result(tmp_path):
    cache = ForecastCache(tmp_path / 'cache')
    cache.save_result('ndq.ax', 90, {'cache_info': {}, 'forecast': 'previous'})
    cache.close()
    with patch.object(StockPredictor, 'get_stock_info', side_effect=ValueError('provider')):
        predictor = StockPredictor('ndq.ax', 90, refresh=True,
                                   cache_directory=tmp_path / 'cache',
                                   ticker_file=str(tmp_path / 'missing.json'))
    assert predictor.result['cache_info']['stale']
    assert 'Refresh unavailable' in predictor.result['cache_info']['warning']


def test_result_does_not_outlive_parameters(tmp_path):
    cache = ForecastCache(tmp_path)
    with patch('app.forecast_cache.time.time', return_value=1000):
        cache.save_result('ndq.ax', 90, {'cache_info': {'parameters': {'expires_at': 1100}}})
    with patch('app.forecast_cache.time.time', return_value=1100):
        assert cache.result('ndq.ax', 90) is None
    cache.close()


def test_refresh_reuses_parameters_and_persists_complete_result(tmp_path, sample_stock_info):
    directory = tmp_path / 'cache'
    cache = ForecastCache(directory)
    cache.save_parameters('ndq.ax', 90, 0.02, 1.0)
    cache.close()
    diagnostics = {'df_performance': pd.DataFrame({'mape': [0.1]}), 'df_cross_validation': pd.DataFrame()}
    with patch.object(StockPredictor, 'get_stock_info', return_value=sample_stock_info) as fetch:
        with patch.object(StockPredictor, 'diagnose_model', return_value=diagnostics) as diagnose:
            with patch.object(StockPredictor, 'make_graphs', return_value={'plotly_price': '{}'}):
                with patch.object(StockPredictor, 'make_forecast_finding_best_params') as tune:
                    first = StockPredictor('ndq.ax', 90, cache_directory=directory,
                                           ticker_file=str(tmp_path / 'missing.json'))
                    second = StockPredictor('ndq.ax', 90, cache_directory=directory,
                                            ticker_file=str(tmp_path / 'missing.json'))
    fetch.assert_called_once()
    diagnose.assert_called_once()
    tune.assert_not_called()
    pd.testing.assert_frame_equal(first.result['forecast'], second.result['forecast'])
    assert first.result['returns'] == second.result['returns']
    assert second.result['fig_paths'] == {'plotly_price': '{}'}
    assert second.result['params_info']['changepoint_prior_scale'] == 0.02
    assert second.result['cache_info']['parameters']['source'] == 'tuned'
    assert len(second.result['cache_info']['data_fingerprint']) == 64


def test_retune_is_explicit_and_commits_only_success(tmp_path, sample_stock_info):
    predictor = StockPredictor(cache_directory=tmp_path / 'cache',
                               ticker_file=str(tmp_path / 'missing.json'))
    predictor.ticker = 'ndq.ax'
    diagnostics = {'df_performance': pd.DataFrame({'mape': [0.1]}), 'df_cross_validation': pd.DataFrame()}

    def tune():
        return {'forecast_info': predictor.make_forecast(0.1, 1.0),
                'diagnostics': diagnostics, 'changepoint_prior_scale': 0.1,
                'seasonality_prior_scale': 1.0}

    with patch.object(predictor, 'get_stock_info', return_value=sample_stock_info):
        with patch.object(predictor, 'diagnose_model', return_value=diagnostics):
            with patch.object(predictor, 'make_graphs', return_value={}):
                with patch.object(predictor, 'make_forecast_finding_best_params', side_effect=tune) as search:
                    predictor.load_forecast(90, refresh=True)
                    search.assert_not_called()
                    assert predictor.result['cache_info']['parameters']['source'] == 'default'
                    predictor.load_forecast(90, retune=True)
                    search.assert_called_once()
    saved = predictor.cache.parameters('ndq.ax', 90)
    assert saved['cps'] == 0.1
    assert saved['tuned_at'] is not None
    assert predictor.cache.parameters('ndq.ax', 180) is None
    with patch.object(predictor, 'get_stock_info', side_effect=ValueError('offline')):
        with pytest.raises(ValueError, match='offline'):
            predictor.load_forecast(90, retune=True)
    assert predictor.cache.parameters('ndq.ax', 90) == saved
    assert predictor.cache.result('ndq.ax', 90)['cache_info']['parameters'] == saved
    predictor.cache.close()


def test_batch_success_failure_and_duplicate_tickers(tmp_path):
    predictor = StockPredictor(cache_directory=tmp_path / 'cache',
                               ticker_file=str(tmp_path / 'missing.json'))
    with patch.object(predictor, 'load_forecast', side_effect=[ValueError('bad'), None]) as refresh:
        summary = predictor.preload(90, tickers=['bad', ' NDQ.AX ', 'ndq.ax'])
    assert summary == {'succeeded': ['ndq.ax'], 'failed': ['bad']}
    assert refresh.call_count == 2
    refresh.assert_called_with(90, refresh=True, retune=False, allow_stale=False)
    with predictor.cache.lock('preload-batch', None):
        with pytest.raises(Timeout):
            predictor.preload(90, tickers=['ndq.ax'])
    predictor.cache.close()


def test_expired_parameters_use_defaults_without_retuning(tmp_path, sample_stock_info):
    predictor = StockPredictor(cache_directory=tmp_path / 'cache',
                               ticker_file=str(tmp_path / 'missing.json'))
    predictor.ticker = 'ndq.ax'
    with patch('app.forecast_cache.time.time', return_value=0):
        predictor.cache.save_parameters('ndq.ax', 90, 0.1, 1.0)
    diagnostics = {'df_performance': pd.DataFrame({'mape': [0.1]}), 'df_cross_validation': pd.DataFrame()}
    with patch.object(predictor, 'get_stock_info', return_value=sample_stock_info):
        with patch.object(predictor, 'diagnose_model', return_value=diagnostics):
            with patch.object(predictor, 'make_graphs', return_value={}):
                with patch.object(predictor, 'make_forecast_finding_best_params') as tune:
                    predictor.load_forecast(90)
                    assert predictor.result['cache_info']['parameters']['source'] == 'default'
                    tune.assert_not_called()
    predictor.cache.close()