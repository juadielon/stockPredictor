import yfinance as yf
import pandas as pd
import pandas_market_calendars as market_calendars
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from datetime import datetime
import numpy as np
from app.forecast_cache import ForecastCache
from filelock import FileLock, Timeout
import hashlib
import logging

import os.path
import json
import tempfile

import time
import math
import plotly.graph_objs as go

class StockPredictor:
    def __init__(self, ticker='', periods=365, *, refresh=False, retune=False,
                 cache_directory='./tmp/forecasts-v2',
                 ticker_file='./tmp/tickers_change_point_prior_scale.json'):
        self.ticker = ticker.strip().lower()
        periods = self.validate_periods(periods)
        self.cache = ForecastCache(cache_directory)
        self.cache_obj_file_path = ticker_file
        self.cache_obj = []
        try:
            self.prime_cache()
            if self.ticker:
                self.load_forecast(periods, refresh=refresh, retune=retune)
        except Exception:
            self.cache.close()
            raise
        if self.ticker:
            self.cache.close()

    def load_forecast(self, periods, *, refresh=False, retune=False, allow_stale=True):
        periods = self.validate_periods(periods)
        self.requested_periods = periods
        cached = self.cache.result(self.ticker, periods)
        if cached is not None and not refresh and not retune:
            self.result = cached
            return
        try:
            with self.cache.lock(self.ticker, periods):
                self.stock_info = self.get_stock_info()
                self.stock_info['now'] = datetime.now()
                self.periods = self.restrict_max_periods(periods)
                self.forecaster(retune=retune)
        except Exception as error:
            previous = self.cache.result(self.ticker, periods, allow_stale=True)
            if allow_stale and previous is not None and not retune:
                previous['cache_info']['stale'] = True
                previous['cache_info']['warning'] = 'Refresh unavailable. Showing the last saved forecast.'
                self.result = previous
                logging.getLogger(__name__).warning('Refresh failed for %s', self.ticker, exc_info=True)
                return
            if isinstance(error, Timeout):
                raise ValueError('A forecast for this ticker and period is already running.') from error
            raise

    def preload(self, periods=365, *, retune=False, tickers=None):
        periods = self.validate_periods(periods)
        tickers = tickers if tickers is not None else self.cache.tickers(self.cache_obj_file_path)
        summary = {'succeeded': [], 'failed': []}
        with self.cache.lock('preload-batch', None):
            for ticker in sorted({ticker.strip().lower() for ticker in tickers}):
                self.ticker = ticker
                try:
                    self.load_forecast(periods, refresh=True, retune=retune, allow_stale=False)
                    summary['succeeded'].append(ticker)
                    print(f'{ticker}: refreshed', flush=True)
                except Exception:
                    summary['failed'].append(ticker)
                    logging.getLogger(__name__).exception('Refresh failed for %s', ticker)
                    print(f'{ticker}: failed; previous result retained', flush=True)
        return summary

    def prime_cache(self):
        if not os.path.isfile(self.cache_obj_file_path):
            print('File ' + self.cache_obj_file_path + ' was not found')
            return

        with open(self.cache_obj_file_path, 'r', encoding='utf-8') as json_file:
            self.cache_obj = json.load(json_file)

        self.cache.import_legacy(self.cache_obj)

    def write_cache(self, changepoint_prior_scale, seasonality_prior_scale):
        return self.cache.save_parameters(
            self.ticker, self.requested_periods, changepoint_prior_scale,
            seasonality_prior_scale, str(self.stock_info['historical_data'].index.max())
        )

    def restrict_max_periods(self, periods):
        # Make sure the new number of periods to use is not bigger than 36% of the historical periods
        periods = self.validate_periods(periods)
        historical_periods_count = len(self.stock_info['historical_data'])
        # Estimate the number of maximum periods allowed. This was derived by trial and error
        max_periods = int(historical_periods_count * 0.36) + 1

        return max_periods if periods > max_periods else periods

    @staticmethod
    def validate_periods(periods):
        if isinstance(periods, bool) or not str(periods).strip().isdigit():
            raise ValueError('Forecast days must be an integer between 1 and 730.')
        periods = int(periods)
        if not 1 <= periods <= 730:
            raise ValueError('Forecast days must be an integer between 1 and 730.')
        return periods

    def forecaster(self, retune=False):
        """
        Forecast the given ticker/quote a number of days into the future from today

        Inputs:
        ticker - is the ticker/quote of the stock as defined by Yahoo Finance
        periods - is the number of days into the future to forecast
        """

        params = (self.cache.parameters(self.ticker, self.requested_periods)
              or self.cache.parameters(self.ticker, None)
              or {'cps': 0.05, 'sps': 10.0, 'source': 'default', 'tuned_at': None})
        if retune:
            print('Retuning model parameters for', self.ticker)
            optimal_forecast = self.make_forecast_finding_best_params()
            print('Results were for ticker', self.ticker)

            # Calculate deltas
            delta = optimal_forecast['forecast_info']['forecast']['yhat'].pct_change()
            optimal_forecast['forecast_info']['forecast'] = optimal_forecast['forecast_info']['forecast'].assign(delta = delta.values)

            # Ensure df_performance is in forecast_info for make_graphs
            optimal_forecast['forecast_info']['df_performance'] = optimal_forecast['diagnostics']['df_performance']
            optimal_forecast['forecast_info']['df_cross_validation'] = optimal_forecast['diagnostics']['df_cross_validation']

            fig_paths = self.make_graphs(optimal_forecast['forecast_info'])
            result = {
                'params_info': optimal_forecast['forecast_info']['params_info'],
                'forecast': optimal_forecast['forecast_info']['forecast'],
                'performance': optimal_forecast['diagnostics']['df_performance']
            }
        else:
            print('Using model parameters from', params['source'])
            changepoint_prior_scale = params['cps']
            seasonality_prior_scale = params['sps']

            # Test the model using up to 25% of historical data as the horizon, max 365 days
            horizon_days = min(365, int(len(self.stock_info['historical_data']) * 0.25))

            forecast_info = self.make_forecast(changepoint_prior_scale, seasonality_prior_scale)
            forecast_info['change_point_prior_scale'] = changepoint_prior_scale
            forecast_info['params_info']['horizon_days'] = horizon_days

            diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
            forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']

            # Calculate deltas
            delta = forecast_info['forecast']['yhat'].pct_change()
            forecast_info['forecast'] = forecast_info['forecast'].assign(delta = delta.values)

            # Ensure df_performance is in forecast_info for make_graphs
            forecast_info['df_performance'] = diagnostics['df_performance']

            fig_paths = self.make_graphs(forecast_info)
            result = {
                'params_info': forecast_info['params_info'],
                'forecast': forecast_info['forecast'],
                'performance': diagnostics['df_performance']
            }
        result['stock_info'] = self.stock_info
        result['fig_paths'] = fig_paths
        result['returns'] = self.calculate_returns(result['forecast'], result['params_info'])
        history_hash = pd.util.hash_pandas_object(self.stock_info['historical_data'], index=True)
        result['cache_info'] = {
            'data_cutoff': str(self.stock_info['historical_data'].index.max()),
            'data_fingerprint': hashlib.sha256(history_hash.values.tobytes()).hexdigest(),
        }
        with self.cache.store.transact():
            if retune:
                params = self.write_cache(
                    optimal_forecast['changepoint_prior_scale'],
                    optimal_forecast['seasonality_prior_scale']
                )
            result['cache_info']['parameters'] = params
            self.cache.save_result(self.ticker, self.requested_periods, result)
            self.cache.remember_ticker(self.ticker)
        self.result = result

    def calculate_returns(self, forecast, params_info):
        origin_price = float(params_info['origin_price'])
        endpoint_price = float(forecast['yhat'].iloc[-1])
        elapsed_days = params_info['elapsed_days']
        if (elapsed_days <= 0 or not math.isfinite(origin_price)
                or not math.isfinite(endpoint_price) or origin_price <= 0 or endpoint_price <= 0):
            raise ValueError('Returns require positive finite prices and a future forecast date.')
        price_ratio = endpoint_price / origin_price
        try:
            annualised = math.expm1(math.log(price_ratio) * 365 / elapsed_days)
        except (OverflowError, ValueError) as error:
            raise ValueError('The forecast cannot be annualised reliably.') from error
        if not math.isfinite(annualised):
            raise ValueError('The forecast cannot be annualised reliably.')
        return {'requested_period': price_ratio - 1, 'annualised': annualised}

    def get_market_country(self):
        """
        Determine market country for holiday calculations based on ticker and exchange info.
        Returns a country code (e.g., 'AU', 'US') or None for 24/7 markets like crypto.
        """
        ticker_lower = self.ticker.lower()
        
        # Crypto or forex tickers typically trade 24/7 without market holidays
        if any(c in ticker_lower for c in ['-usd', '-aud', '-eur', '-gbp', '=x']):
            return None
            
        # ASX Australian Securities Exchange
        if ticker_lower.endswith('.ax'):
            return 'AU'
            
        # Check exchange info from Yahoo Finance if available
        info = getattr(self, 'stock_info', {}).get('info', {})
        exchange = info.get('exchange', '').upper()
        
        # Australian exchanges
        if exchange in ['ASX', 'ASX - ALL MARKETS']:
            return 'AU'
            
        # UK
        if ticker_lower.endswith('.l') or exchange in ['LSE']:
            return 'GB'
            
        # Canada
        if ticker_lower.endswith('.to') or ticker_lower.endswith('.v') or exchange in ['TSX', 'TORONTO']:
            return 'CA'
            
        # Default for US equities (NYSE, NASDAQ, AMEX, etc.) or unknown
        return 'US'

    #@cache.memoize(typed=True, expire=43200)  # cache for 12 hours
    def get_stock_info(self):
        """
        Retrieves stock's information from Yahoo Finance

        Inputs:
        ticker - is the ticker/quote of the stock as defined by Yahoo Finance
        """

        print('Retrieving data from Yahoo Finance for ticker ', self.ticker)

        # Get historical data from Yahoo Finance
        stock_data = yf.Ticker(self.ticker)

        info = stock_data.info
        latest_data = stock_data.history('1d', auto_adjust=False)
        if latest_data.empty or 'Close' not in latest_data:
            raise ValueError('No current closing price is available for this ticker.')
        info['currentPrice'] = float(latest_data['Close'].iloc[-1])
        if not math.isfinite(info['currentPrice']) or info['currentPrice'] <= 0:
            raise ValueError('The current closing price must be positive and finite.')
        # info['longBusinessSummary'] = info['longBusinessSummary'].value.decode('utf-8','ignore').encode("utf-8")
        
        dividends = stock_data.dividends

        # For Polkadot, request newer data as old data has some weird prices.
        # This is happening after Yahoo Finance changed the ticker from dot1-aud to dot-aud
        if self.ticker == 'dot-aud':
            historical_data = stock_data.history(start='2020-08-20', auto_adjust=True)
        else:
            # Yahoo Finance allows to retrieve historical data for:
            # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            historical_data = stock_data.history('max', auto_adjust=True)

        # Remove outliers using IQR method
        # Q1 = historical_data['Close'].quantile(0.25)
        # Q3 = historical_data['Close'].quantile(0.75)
        # IQR = Q3 - Q1
        # lower_bound = Q1 - 2.5 * IQR
        # upper_bound = Q3 + 2.5 * IQR
        # print(f"DEBUG: Outlier bounds: lower={lower_bound}, upper={upper_bound}")
        # print(f"DEBUG: Rows before outlier removal: {len(historical_data)}")
        # outliers = historical_data[(historical_data['Close'] < lower_bound) | (historical_data['Close'] > upper_bound)]['Close']
        # print('Close price values removed as outliers from historical data:', outliers)
        # historical_data = historical_data[(historical_data['Close'] >= lower_bound) & (historical_data['Close'] <= upper_bound)]
        # print(f"DEBUG: Rows after outlier removal: {len(historical_data)}")
        # print(f"DEBUG: Last historical ds: {historical_data.index.max()}")

        return {
            'info': info,
            'dividends': dividends,
            'historical_data': historical_data
        }

    def make_forecast_finding_best_changepoint_prior_scale1(self):
        """
        Find the best changepoint prior scale to use, returning the forecast.
        According to the fphropet manual, the changepoint prior scale is probably the most
        impactful parameter: "It determines the flexibility of the trend, and in particular
        how much the trend changes at the trend changepoints. If it is too small, the trend
        will be underfit and variance that should have been modeled with trend changes will
        instead end up being handled with the noise term. If it is too large, the trend will
        overfit and in the most extreme case you can end up with the trend capturing yearly
        seasonality. The default of 0.05 works for many time series, but this could be tuned;
        a range of [0.001, 0.5] would likely be about right. Parameters like this
        (regularization penalties; this is effectively a lasso penalty) are often tuned on a
        log scale."

        This method starts with a change point prior scale of 0.01, evaluating the Mean
        Absolute Percent Error (MAPE) and continuing with the next change point until the MAPE
        starts to increase. This method choose the first minimum value and not necesarily the
        absolute minimum value.

        Inputs:
        historical_data - is the historical stock data
        periods - is the number of days to forecast
        """

        min_mape = 100
        changepoint_prior_scale = 0
        continue_loop = True
        # Test the model using 25% of historical data as the horizon
        horizon_days = int(len(self.stock_info['historical_data']) * 0.25)

        while continue_loop:
            changepoint_prior_scale += 0.01
            forecast_info = self.make_forecast(changepoint_prior_scale)
            forecast_info['params_info']['horizon_days'] = horizon_days

            diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
            forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
            mape = diagnostics['df_performance'].tail(1).mape.values

            print('mape=', mape)
            print('temp changepoint_prior_scale=', changepoint_prior_scale)
            if mape < min_mape:
                min_mape = mape
                result = {
                    'forecast_info': forecast_info,
                    'diagnostics': diagnostics,
                    'changepoint_prior_scale': changepoint_prior_scale
                }
            else:
                continue_loop = False

        print('min_mape=', min_mape)
        print('best changepoint_prior_scale=', result['changepoint_prior_scale'])
        return result

    def save_tuning_progress(self, changepoint_prior_scale, seasonality_prior_scale):
        path = os.path.realpath(self.cache_obj_file_path)
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        with FileLock(path + '.lock', timeout=10):
            entries = []
            if os.path.isfile(path):
                with open(path, encoding='utf-8') as source:
                    entries = json.load(source)
            if not isinstance(entries, list) or any(
                not isinstance(entry, dict) or not isinstance(entry.get('ticker'), str)
                for entry in entries
            ):
                raise ValueError('The ticker file must contain a list of entries with ticker symbols.')
            settings = {
                'changepoint_prior_scale': float(changepoint_prior_scale),
                'seasonality_prior_scale': float(seasonality_prior_scale),
            }
            matched = False
            for entry in entries:
                if entry['ticker'].strip().lower() == self.ticker.strip().lower():
                    entry.update(settings)
                    matched = True
            if not matched:
                entries.append({'ticker': self.ticker.strip().lower(), **settings})
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory,
                                                 prefix='.ticker-settings-', suffix='.json',
                                                 delete=False) as destination:
                    temporary_path = destination.name
                    json.dump(entries, destination, indent=4, allow_nan=False)
                    destination.write('\n')
                    destination.flush()
                    os.fsync(destination.fileno())
                os.replace(temporary_path, path)
            finally:
                if temporary_path is not None and os.path.exists(temporary_path):
                    os.unlink(temporary_path)

    def make_forecast_finding_best_params(self):
        """
        Find the best changepoint_prior_scale and seasonality_prior_scale using grid search.
        According to the fphropet manual, the changepoint prior scale is probably the most
        impactful parameter, and the seasonality prior scale is also important.
        The changepoint prior scale: "It determines the flexibility of the trend, and in particular
        how much the trend changes at the trend changepoints. If it is too small, the trend
        will be underfit and variance that should have been modeled with trend changes will
        instead end up being handled with the noise term. If it is too large, the trend will
        overfit and in the most extreme case you can end up with the trend capturing yearly
        seasonality. The default of 0.05 works for many time series, but this could be tuned;
        a range of [0.001, 0.5] would likely be about right. Parameters like this
        (regularization penalties; this is effectively a lasso penalty) are often tuned on a
        log scale."
        The seasonality prior scale: "Similar to the changepoint prior scale, this parameter
        controls the flexibility of the seasonality model. Larger values allow the seasonality
        to fit larger seasonal fluctuations, smaller values dampen the seasonality. The
        default of 10.0 works for many time series, but this could be tuned; a range of
        [0.1, 50.0] would likely be about right."
        This method evaluates a changepoint prior scale from 0.01 until 0.5 with a 0.01 step,
        choosing the value producing the minimum Mean Absolute Percent Error (MAPE). As it
        evaluates every single point it finds the absolute minimum in the range evaluated
        at the cost of speed.
        Inputs:
        historical_data - is the historical stock data
        periods - is the number of days to forecast
        """
        start = time.time()
        # Test the model using up to 25% of historical data as the horizon, max 365 days
        horizon_days = min(365, int(len(self.stock_info['historical_data']) * 0.25))

        # Define ranges
        changepoint_scales = np.arange(0.01, 0.51, 0.01) # From 0.01 to 0.5 with a 0.01 step
        seasonality_scales = [0.1, 1.0, 10.0, 50.0]  # Common values

        result_min_mape = {'mape': float('inf')}

        for cps in changepoint_scales:
            for sps in seasonality_scales:
                forecast_info = self.make_forecast(cps, sps)
                forecast_info['params_info']['horizon_days'] = horizon_days

                diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
                mape = diagnostics['df_performance'].tail(1).mape.values[0]

                print(self.ticker + f' - cps={cps}, sps={sps}, mape={mape}')

                if math.isfinite(mape) and mape < result_min_mape['mape']:
                    self.save_tuning_progress(cps, sps)
                    print(f'{self.ticker}: saved best settings so far to {self.cache_obj_file_path}', flush=True)
                    result_min_mape = {
                        'forecast_info': forecast_info,
                        'diagnostics': diagnostics,
                        'changepoint_prior_scale': cps,
                        'seasonality_prior_scale': sps,
                        'mape': mape
                    }

        if not math.isfinite(result_min_mape['mape']):
            raise ValueError('No finite validation error was found during retuning.')
        print(f'Best: cps={result_min_mape["changepoint_prior_scale"]}, sps={result_min_mape["seasonality_prior_scale"]}, mape={result_min_mape["mape"]}')
        return result_min_mape

    def make_forecast_finding_best_changepoint_prior_scale3(self):
        """
        Find the best changepoint prior scale to use, returning the forecast.
        According to the fphropet manual, the changepoint prior scale is probably the most
        impactful parameter: "It determines the flexibility of the trend, and in particular
        how much the trend changes at the trend changepoints. If it is too small, the trend
        will be underfit and variance that should have been modeled with trend changes will
        instead end up being handled with the noise term. If it is too large, the trend will
        overfit and in the most extreme case you can end up with the trend capturing yearly
        seasonality. The default of 0.05 works for many time series, but this could be tuned;
        a range of [0.001, 0.5] would likely be about right. Parameters like this
        (regularization penalties; this is effectively a lasso penalty) are often tuned on a
        log scale."

        This method use a ternary search to find the a change point prior scale value that
        produces a minimum Mean Absolute Percent Error (MAPE). As the evaluated function is
        not necessarily and unimodal function, the minimum value found might not be necesarily
        be the minimum value.

        With some modifications this method can also use the golden section search algorithm

        Inputs:
        historical_data - is the historical stock data
        periods - is the number of days to forecast

        @todo refactor the method to remove double code
        """
        start = time.time()
        # Test the model using 25% of historical data as the horizon
        horizon_days = int(len(self.stock_info['historical_data']) * 0.25)

        stats = []

        left_cps = 0.001 # left changepoint prior scale
        right_cps = 0.5 # right changepoint prior scale
        precision = 0.01

        golden_ratio = (math.sqrt(5) +1) / 2

        while abs(right_cps - left_cps) >= precision:
            # left_cps_third = left_cps + (right_cps - left_cps) / 3
            # right_cps_third = right_cps - (right_cps - left_cps) / 3
            left_cps_third = right_cps - (right_cps - left_cps) / golden_ratio
            right_cps_third = left_cps + (right_cps - left_cps) / golden_ratio

            print('evaluating left ', left_cps_third)

            forecast_info = self.make_forecast(left_cps_third)
            forecast_info['params_info']['horizon_days'] = horizon_days

            diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
            forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
            left_mape = diagnostics['df_performance'].tail(1).mape.values[0]

            stat = {
                'changepoint_prior_scale': left_cps_third,
                'mape': left_mape
            }
            stats.append(stat)
            print(pd.DataFrame(stats).reindex(
                columns=['changepoint_prior_scale', 'mape']))

            print('evaluating right ', right_cps_third)
            forecast_info = self.make_forecast(right_cps_third)
            forecast_info['params_info']['horizon_days'] = horizon_days

            diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
            forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
            right_mape = diagnostics['df_performance'].tail(1).mape.values[0]

            stat = {
                'changepoint_prior_scale': right_cps_third,
                'mape': right_mape
            }
            stats.append(stat)
            print(pd.DataFrame(stats).reindex(
                columns=['changepoint_prior_scale', 'mape']))

            if left_mape > right_mape:
                left_cps = left_cps_third
            else:
                right_cps = right_cps_third

            print('time=', time.time() - start)

        best_changepoint_prior_scale = (left_cps + right_cps) / 2
        print('evaluating best ', best_changepoint_prior_scale)
        forecast_info = self.make_forecast(best_changepoint_prior_scale)
        forecast_info['params_info']['horizon_days'] = horizon_days

        diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
        forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
        mape = diagnostics['df_performance'].tail(1).mape.values[0]

        stat = {
            'changepoint_prior_scale': best_changepoint_prior_scale,
            'mape': mape
        }
        stats.append(stat)
        print(pd.DataFrame(stats).reindex(
            columns=['changepoint_prior_scale', 'mape']))
        print('time=', time.time() - start)

        result_min_mape = {
            'forecast_info': forecast_info,
            'diagnostics': diagnostics,
            'changepoint_prior_scale': best_changepoint_prior_scale,
            'mape': mape
        }

        return result_min_mape

    def make_forecast_finding_best_changepoint_prior_scale4(self):
        """
        Find the best changepoint prior scale to use, returning the forecast.
        According to the fphorpet manual, the changepoint prior scale is probably the most
        impactful parameter: "It determines the flexibility of the trend, and in particular
        how much the trend changes at the trend changepoints. If it is too small, the trend
        will be underfit and variance that should have been modeled with trend changes will
        instead end up being handled with the noise term. If it is too large, the trend will
        overfit and in the most extreme case you can end up with the trend capturing yearly
        seasonality. The default of 0.05 works for many time series, but this could be tuned;
        a range of [0.001, 0.5] would likely be about right. Parameters like this
        (regularization penalties; this is effectively a lasso penalty) are often tuned on a
        log scale."

        Similarly to the third method above, this method uses the a Golden section search
        algorithm to find the a change point prior scale value that produces a minimum Mean
        Absolute Percent Error (MAPE). As the evaluated function is not necessarily and unimodal
        function, the minimum value found might not be necesarily the minimum value. The
        difference with the third method above is that this method reuses function evaluations,
        saving evaluations per iteration, saving time in the process.

        Inputs:
        historical_data - is the historical stock data
        periods - is the number of days to forecast

        @todo refactor the method to remove double code
        """

        start = time.time()
        # Test the model using 25% of historical data as the horizon
        horizon_days = int(len(self.stock_info['historical_data']) * 0.25)

        stats = []
        #result_min_mape = {'mape': 100}

        left_cps = 0.001 # left changepoint prior scale
        right_cps = 0.5 # right changepoint prior scale
        precision = 0.01

        # based on golden ratio (math.sqrt(5) +1) / 2
        inv_phi = (math.sqrt(5)-1) / 2 # 1 / phi
        inv_phi2 = (3 - math.sqrt(5)) / 2 # 1 / phi^2

        distance = right_cps - left_cps
        max_steps = int(math.ceil(math.log(precision/distance) / math.log(inv_phi)))

        left_cps_tmp = left_cps + inv_phi2 * distance
        right_cps_tmp = left_cps + inv_phi * distance

        print('evaluating left ', left_cps_tmp)
        forecast_info = self.make_forecast(left_cps_tmp)
        forecast_info['params_info']['horizon_days'] = horizon_days

        diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
        forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
        left_mape = diagnostics['df_performance'].tail(1).mape.values[0]

        stat = {
            'changepoint_prior_scale': left_cps_tmp,
            'mape': left_mape
        }
        stats.append(stat)
        print(pd.DataFrame(stats).reindex(
            columns=['changepoint_prior_scale', 'mape']))

        print('evaluating right ', right_cps_tmp)
        forecast_info = self.make_forecast(right_cps_tmp)
        forecast_info['params_info']['horizon_days'] = horizon_days

        diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
        forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
        right_mape = diagnostics['df_performance'].tail(1).mape.values[0]

        stat = {
            'changepoint_prior_scale': right_cps_tmp,
            'mape': right_mape
        }
        stats.append(stat)
        print(pd.DataFrame(stats).reindex(
            columns=['changepoint_prior_scale', 'mape']))

        for i in range(max_steps):
            if left_mape < right_mape:
                right_cps = right_cps_tmp
                right_cps_tmp = left_cps_tmp
                right_mape = left_mape
                distance = inv_phi * distance
                left_cps_tmp = left_cps + inv_phi2 * distance

                print('evaluating left ', left_cps_tmp)
                forecast_info = self.make_forecast(left_cps_tmp)
                forecast_info['params_info']['horizon_days'] = horizon_days

                diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
                forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
                left_mape = diagnostics['df_performance'].tail(1).mape.values[0]

                stat = {
                    'changepoint_prior_scale': left_cps_tmp,
                    'mape': left_mape
                }
                stats.append(stat)
                print(pd.DataFrame(stats).reindex(
                    columns=['changepoint_prior_scale', 'mape']))
                print('time=', time.time() - start)
            else:
                left_cps = left_cps_tmp
                left_cps_tmp = right_cps_tmp
                left_mape = right_mape
                distance = inv_phi * distance
                right_cps_tmp = left_cps + inv_phi * distance

                print('evaluating right ', right_cps_tmp)
                forecast_info = self.make_forecast(right_cps_tmp)
                forecast_info['params_info']['horizon_days'] = horizon_days

                diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
                forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
                right_mape = diagnostics['df_performance'].tail(1).mape.values[0]

                stat = {
                    'changepoint_prior_scale': right_cps_tmp,
                    'mape': right_mape
                }
                stats.append(stat)
                print(pd.DataFrame(stats).reindex(
                    columns=['changepoint_prior_scale', 'mape']))
                print('time=', time.time() - start)

        if left_mape < right_mape:
            print('range ', left_cps, right_cps_tmp)
            print('best ', (left_cps + right_cps_tmp) / 2)
            best_changepoint_prior_scale = (left_cps + right_cps_tmp) / 2
        else:
            print('range ', left_cps_tmp, right_cps)
            print('best ', (left_cps_tmp + right_cps) / 2)
            best_changepoint_prior_scale = (left_cps_tmp + right_cps) / 2

        print('evaluating best ', best_changepoint_prior_scale)
        forecast_info = self.make_forecast(best_changepoint_prior_scale)
        forecast_info['params_info']['horizon_days'] = horizon_days

        diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
        forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
        mape = diagnostics['df_performance'].tail(1).mape.values[0]

        stat = {
            'changepoint_prior_scale': best_changepoint_prior_scale,
            'mape': mape
        }
        stats.append(stat)
        print(pd.DataFrame(stats).reindex(
            columns=['changepoint_prior_scale', 'mape']))
        print('time=', time.time() - start)

        result_min_mape = {
            'forecast_info': forecast_info,
            'diagnostics': diagnostics,
            'changepoint_prior_scale': best_changepoint_prior_scale,
            'mape': mape
        }

        return result_min_mape

    def make_forecast(self, changepoint_prior_scale=0.05, seasonality_prior_scale=10.0):
        """
        Forecast the price of the stock on a future number of days
        Inputs:
        historical_data - is the historical stock data
        periods - is the number of days to forecast
        """

        # Prophet requires the dates (ds) and adjusted closing prices (y)
        # Create new data frame with the required data
        self.periods = self.validate_periods(self.periods)
        df_historical_data = pd.DataFrame()
        if 'Close' not in self.stock_info['historical_data']:
            raise ValueError('Historical closing prices are unavailable for this ticker.')
        df_historical_data['ds'] = pd.DatetimeIndex(
            self.stock_info['historical_data'].index
        ).tz_localize(None).normalize()
        df_historical_data['y'] = self.stock_info['historical_data']['Close'].values
        df_historical_data = df_historical_data.sort_values('ds').reset_index(drop=True)
        if (len(df_historical_data) < 2 or df_historical_data['ds'].isna().any()
                or df_historical_data['ds'].duplicated().any()
                or not np.isfinite(df_historical_data['y']).all()
                or (df_historical_data['y'] <= 0).any()):
            raise ValueError('Historical data must contain distinct dates and positive finite closing prices.')

        # Set minimum posible value
        # df_historical_data['floor'] = 0

        # Determine if yearly seasonality should be enabled based on data length
        data_length = len(df_historical_data)
        enable_yearly_seasonality = data_length >= 366

        # Create a Prophet model
        # As there is one single closing price daily, disable the daily seasonality
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=enable_yearly_seasonality,
            seasonality_prior_scale=seasonality_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale
        )

        #model = Prophet(
        #    growth='logistic',
        #    #seasonality_mode='multiplicative',
        #    daily_seasonality=False,
        #    changepoint_prior_scale=changepoint_prior_scale
        #)
        #df_historical_data['floor'] = 0
        #df_historical_data['cap'] = 1.2 * df_historical_data['y'].max()

        #df_historical_data['y'] = np.log(1 + df_historical_data['y'])
        #model = Prophet(
        #    daily_seasonality=False,
        #    changepoint_prior_scale=changepoint_prior_scale
        #)

        market_country = self.get_market_country()
        if market_country:
            try:
                model.add_country_holidays(country_name=market_country)
            except Exception as e:
                print(f"Warning: could not add holidays for country {market_country}: {e}")

        model.fit(df_historical_data)

        # Start forecast from the last historical date
        last_ds = df_historical_data['ds'].max()
        total_future = pd.DataFrame({'ds': pd.date_range(
            start=last_ds + pd.Timedelta(days=1), periods=self.periods, freq='D'
        )})

        #total_future['floor'] = 0
        #total_future['cap'] = 1.2 * df_historical_data['y'].max()

        # Check if there is data only on business days and if so remove weekends in the future
        # For ASX tickers (.ax) or market country tickers, treat as stock markets (remove weekends)
        # For others, check if historical data includes weekends
        is_asx_ticker = self.ticker.endswith('.ax')
        if is_asx_ticker or market_country is not None:
            has_weekend_data = False  # Stock market behaviour
        else:
            is_crypto = any(
                self.ticker.lower().endswith(suffix)
                for suffix in ('-usd', '-aud', '-eur', '-gbp')
            )
            has_weekend_data = is_crypto or any(df_historical_data['ds'].dt.dayofweek >= 5)
        if market_country == 'AU':
            calendar = market_calendars.get_calendar('ASX')
            sessions = calendar.valid_days(
                start_date=last_ds + pd.Timedelta(days=1),
                end_date=last_ds + pd.Timedelta(days=self.periods)
            ).tz_localize(None)
            future_days = pd.DataFrame({'ds': sessions})
        elif not has_weekend_data:
            # As the stock exchange is closed on weekends, remove weekends in the future
            future_days = total_future[total_future['ds'].dt.dayofweek < 5]
        else:
            # Some markets are open all the time e.g. crypto
            future_days = total_future

        # Recalculate number of available periods to display in case that some days were removed
        future_weekdays_count = len(future_days)
        if future_weekdays_count == 0:
            raise ValueError('No trading dates fall within the requested forecast period.')

        full_forecast = model.predict(future_days)

        # Predict on historical data for components
        hist_forecast = model.predict(df_historical_data)

        #model.history['y'] = np.exp(model.history['y']) -1
        #df_historical_data['y'] = np.exp(df_historical_data['y']) -1
        #for col in ['yhat', 'yhat_lower', 'yhat_upper', 'trend']:
        #    full_forecast[col] = np.exp(full_forecast[col]) -1

        # Return requested period
        # forecast = full_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(available_periods+1)
        forecast = full_forecast

        result = {
            'historical_data': df_historical_data,
            'full_forecast': full_forecast,
            'hist_forecast': hist_forecast,
            'forecast': forecast,
            'model': model,
            'params_info': {
                'periods': self.periods,
                'historical_periods': len(self.stock_info['historical_data']),
                'weekday_periods': future_weekdays_count,
                'origin': last_ds,
                'origin_price': float(df_historical_data.loc[
                    df_historical_data['ds'].idxmax(), 'y'
                ]),
                'requested_endpoint': last_ds + pd.Timedelta(days=self.periods),
                'forecast_endpoint': future_days['ds'].iloc[-1],
                'elapsed_days': (future_days['ds'].iloc[-1] - last_ds).days,
                'changepoint_prior_scale': changepoint_prior_scale,
                'seasonality_prior_scale': seasonality_prior_scale
            }
        }

        return result

    def diagnose_model(self, horizon_days, model):
        """
        Diagnose the model

        Inputs:
        horizon_days - is the number of days to use when testing the model
        model - is the Phropet model
        """

        horizon = str(horizon_days) + ' days'
        print('horizon', horizon)

        # Adaptive initial: at least the max seasonality period (366 for yearly), but not exceeding data constraints
        data_length = len(self.stock_info['historical_data'])
        max_seasonality_period = 366  # For yearly seasonality
        initial_days = min(data_length - horizon_days - 1, max(max_seasonality_period, max(1, int(horizon_days * 0.5))))
        initial = str(initial_days) + ' days'
        print('initial', initial)

        df_cross_validation = cross_validation(model, horizon=horizon, initial=initial, parallel=None)

        df_performance = performance_metrics(df_cross_validation)
        # print(df_performance)

        return {'df_cross_validation': df_cross_validation, 'df_performance': df_performance}

    def make_graphs(self, stock_data):
        """
        Create and save graphs on a directory for the browser to pickup

        Inputs:
        ticker - Is the ticker/quote of the stock as defined by Yahoo Finance
        stock_data - Dictionary containing information about the stock
        """
        
        # Helper to convert to clean serializable list
        def to_list(vals, is_date=False):
            if is_date:
                return pd.to_datetime(vals).dt.strftime('%Y-%m-%d %H:%M:%S').tolist()

            # Cleanly handle nan/inf for JSON
            cleaned = []
            for v in vals.values.astype(float):
                if math.isnan(v) or math.isinf(v): cleaned.append(None)
                else: cleaned.append(v)
            return cleaned

        # 1. Price Forecast Graph (Manual Construction)
        fig_price = go.Figure()
        fcst = stock_data['full_forecast']
        hist = stock_data['historical_data']
        
        hist_ds = to_list(hist['ds'], is_date=True)
        fcast_ds = to_list(fcst['ds'], is_date=True)

        # Layers (Bottom to Top)
        # Trend
        fig_price.add_trace(go.Scatter(x=fcast_ds, y=to_list(fcst['trend']), name='Trend', line=dict(color='red', width=1.5)))
        
        # Historical
        fig_price.add_trace(go.Scatter(x=hist_ds, y=to_list(hist['y']), name='Historical Price', line=dict(color='black', width=2)))
        
        # Forecast
        fig_price.add_trace(go.Scatter(x=fcast_ds, y=to_list(fcst['yhat']), name='Forecast', line=dict(color='#1f77b4', width=2.5)))
        
        # Uncertainty
        y_up, y_low = to_list(fcst['yhat_upper']), to_list(fcst['yhat_lower'])
        fig_price.add_trace(go.Scatter(x=fcast_ds + fcast_ds[::-1], y=y_up + y_low[::-1], fill='toself', 
                                      fillcolor='rgba(31, 119, 180, 0.2)', line=dict(color='rgba(0,0,0,0)'), 
                                      hoverinfo="skip", showlegend=False, name='Uncertainty'))

        # Today Line
        fig_price.add_vline(x=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), line_width=1, line_dash="dot", line_color="silver")
        
        # All Changepoints (Subtle Light Salmon)
        for cp in stock_data['model'].changepoints:
            cp_str = pd.Timestamp(cp).strftime('%Y-%m-%d %H:%M:%S')
            fig_price.add_vline(x=cp_str, line_width=1, line_dash="dot", line_color="lightsalmon", opacity=0.4)
            
        # Significant Changepoints (Bold Red)
        signif_changepoint_threshold = 0.01
        delta_means = np.abs(np.nanmean(stock_data['model'].params['delta'], axis=0))
        signif_changepoints = stock_data['model'].changepoints[delta_means >= signif_changepoint_threshold] if len(stock_data['model'].changepoints) > 0 else []
        for scp in signif_changepoints:
            scp_str = pd.Timestamp(scp).strftime('%Y-%m-%d %H:%M:%S')
            fig_price.add_vline(x=scp_str, line_width=2, line_dash="dot", line_color="red")

        fig_price.update_layout(
            title=dict(text=f"{self.ticker.upper()} - Close Price & Forecast", font=dict(size=20)),
            xaxis_title="Day (ds)",
            yaxis_title="Price (y)",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=60, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # 2. Components Graph (Manual construction for accurate scaling and trend bands)
        # Using manual construction because native plot_components_plotly fails to correctly 
        # isolate seasonality (producing linear artifacts).
        from plotly.subplots import make_subplots
        comps = []
        if 'trend' in stock_data['full_forecast'].columns: comps.append('trend')
        if 'weekly' in stock_data['full_forecast'].columns: comps.append('weekly')
        if 'yearly' in stock_data['full_forecast'].columns: comps.append('yearly')

        fig_components = make_subplots(rows=len(comps), cols=1, subplot_titles=[c.title() for c in comps], vertical_spacing=0.10)

        # Data reuse
        fcast_ds = to_list(stock_data['full_forecast']['ds'], is_date=True)
        hist_ds = to_list(stock_data['hist_forecast']['ds'], is_date=True)
        combined_ds = hist_ds + fcast_ds
        combined_trend = to_list(stock_data['hist_forecast']['trend']) + to_list(stock_data['full_forecast']['trend'])

        for i, comp in enumerate(comps):
            if comp == 'trend':
                # Add Trend Band (Forecast Range)
                if 'trend_lower' in stock_data['full_forecast'].columns and 'trend_upper' in stock_data['full_forecast'].columns:
                    t_up = to_list(stock_data['full_forecast']['trend_upper'])
                    t_low = to_list(stock_data['full_forecast']['trend_lower'])
                    fig_components.add_trace(go.Scatter(x=fcast_ds + fcast_ds[::-1], y=t_up + t_low[::-1], fill='toself', 
                                                      fillcolor='rgba(0, 114, 178, 0.2)', line=dict(color='rgba(0,0,0,0)'), 
                                                      hoverinfo="skip", showlegend=False, name='Trend Uncertainty'), row=i+1, col=1)
                
                fig_components.add_trace(go.Scatter(x=combined_ds, y=combined_trend, name='Trend', line=dict(color='#0072B2', width=2.5)), row=i+1, col=1)
            
            elif comp == 'weekly':
                # Exact value parity using synthetic Sunday-start baseline
                days = pd.date_range(start='2017-01-01', periods=7, freq='D') # 2017-01-01 was a Sunday
                w_fcst = stock_data['model'].predict(pd.DataFrame({'ds': days}))
                fig_components.add_trace(go.Scatter(x=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], 
                                                   y=to_list(w_fcst['weekly']), name='Weekly', line=dict(color='#0072B2', width=2.5)), row=i+1, col=1)
            
            elif comp == 'yearly':
                # Clean year extraction
                days = pd.date_range(start='2017-01-01', periods=366, freq='D')
                y_fcst = stock_data['model'].predict(pd.DataFrame({'ds': days}))
                y_labels = days.strftime('%B %d').tolist()
                fig_components.add_trace(go.Scatter(x=list(range(366)), y=to_list(y_fcst['yearly']), text=y_labels, hoverinfo="text+y", line=dict(color='#0072B2', width=2.5)), row=i+1, col=1)
                fig_components.update_xaxes(tickvals=list(range(0, 366, 60)), ticktext=[y_labels[k] for k in range(0, 366, 60)], row=i+1, col=1)
            
            if comp != 'trend': fig_components.add_hline(y=0, row=i+1, col=1, line_dash="dash", line_color="gray")

        fig_components.update_layout(title=dict(text=f"{self.ticker.upper()} - Model Components", font=dict(size=20)), 
                                     height=350 * len(comps) + 100, template="plotly_white", showlegend=False, margin=dict(l=50, r=20, t=80, b=50))

        # 3. MAPE Analysis (Manual - No native Plotly equivalent)
        fig_mape = go.Figure()
        if stock_data['df_cross_validation'] is not None:
             df_cv = stock_data['df_cross_validation'].copy()
             df_cv['mape'] = np.abs((df_cv['y'] - df_cv['yhat']) / df_cv['y'])
             df_cv['horizon'] = (df_cv['ds'] - df_cv['cutoff']).dt.days
             
             def _clean_mape(vals):
                 return [None if math.isnan(v) or math.isinf(v) else float(v) for v in vals]
             
             fig_mape.add_trace(go.Scatter(x=df_cv['horizon'].tolist(), y=_clean_mape(df_cv['mape'].values), mode='markers', name='Errors', marker=dict(color='gray', size=4, opacity=0.5)))
             
             if stock_data['df_performance'] is not None:
                  df_p = stock_data['df_performance']
                  fig_mape.add_trace(go.Scatter(x=df_p['horizon'].dt.days.tolist(), y=_clean_mape(df_p['mape'].values), mode='lines', name='Mean MAPE', line=dict(color='#0072B2', width=3)))
             
             fig_mape.update_layout(title=dict(text=f"{self.ticker.upper()} - Cross Validation MAPE", font=dict(size=20)), xaxis_title="Horizon (Days)", yaxis_title="MAPE", template="plotly_white", margin=dict(l=50, r=20, t=80, b=50))

        return {
            'plotly_price': fig_price.to_json(),
            'plotly_components': fig_components.to_json(),
            'plotly_mape': fig_mape.to_json()
        }
