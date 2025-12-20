import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from prophet.plot import add_changepoints_to_plot
from prophet.plot import plot_components_plotly
from datetime import datetime
import numpy as np
from diskcache import FanoutCache
# from diskcache import Cache

import os.path
import json

import time
import math
import plotly.graph_objs as go
import plotly.utils


class StockPredictor:
    def __init__(self, ticker='', periods=365):
        self.ticker = ticker

        self.cache = FanoutCache(directory='./tmp', timeout=20, shards=4)
        # self.cache = FanoutCache(directory='./tmp', timeout=20, eviction_policy='none')
        # self.cache = Cache(directory='./tmp')
        self.cache.clear()
        self.cache_expire = 60 * 60 * 12 # 12 hours

        self.cache_obj_file_path = './tmp/tickers_change_point_prior_scale.json'
        self.cache_obj = []
        self.prime_cache()

        if ticker:
            self.stock_info = self.get_stock_info()
            self.stock_info['now'] = datetime.now()

            self.periods = self.restrict_max_periods(periods)

            self.forecaster()

    def preload(self, periods=365):
        """
        Read previously requested tickers. make forecast on all of them and set cache
        """

        if not os.path.isfile(self.cache_obj_file_path):
            print('File ' + self.cache_obj_file_path + ' was not found')
            return

        with open(self.cache_obj_file_path, 'r', encoding='utf-8') as json_file:
            tickers = json.load(json_file)

        for index in range(len(tickers)):
            self.cache.clear()
            self.cache_obj = []

            self.ticker = tickers[index]['ticker']
            print('Forecasting ' + self.ticker)
            self.stock_info = self.get_stock_info()
            self.stock_info['now'] = datetime.now()

            self.periods = self.restrict_max_periods(periods)

            self.forecaster()

    def prime_cache(self):
        if not os.path.isfile(self.cache_obj_file_path):
            print('File ' + self.cache_obj_file_path + ' was not found')
            return

        with open(self.cache_obj_file_path, 'r', encoding='utf-8') as json_file:
            self.cache_obj = json.load(json_file)

        print('self.cache_obj=', self.cache_obj)

        for index in range(len(self.cache_obj)):
            self.cache.set(
                self.cache_obj[index]['ticker'] + '_best_changepoint_prior_scale',
                self.cache_obj[index]['changepoint_prior_scale'],
                expire=self.cache_expire
            )

    def write_cache(self, changepoint_prior_scale):
        found = False

        # Prepopulate self.cache_obj in case there are multiple concurrent runs
        self.prime_cache()

        for index in range(len(self.cache_obj)):
            if self.cache_obj[index]['ticker'] == self.ticker:
                found = True
                self.cache_obj[index]['changepoint_prior_scale'] = changepoint_prior_scale

        if not found:
            self.cache_obj.append({'ticker': self.ticker, 'changepoint_prior_scale': changepoint_prior_scale})

        with open(self.cache_obj_file_path, 'w') as json_file:
            #json.dump(self.cache_obj, json_file, indent=4)
            # Compact format the json file
            json_file.write(json.dumps(self.cache_obj).replace('[{', '[\n\t{').replace('}, ', '},\n\t').replace('}]', '}\n]'))

    def restrict_max_periods(self, periods):
        # Make sure the new number of periods to use is not bigger than 36% of the historical periods
        periods = int(periods)
        historical_periods_count = len(self.stock_info['historical_data'])
        # Estimate the number of maximum periods allowed. This was derived by trial and error
        max_periods = int(historical_periods_count * 0.36) + 1

        return max_periods if periods > max_periods else periods

    def forecaster(self):
        """
        Forecast the given ticker/quote a number of days into the future from today

        Inputs:
        ticker - is the ticker/quote of the stock as defined by Yahoo Finance
        periods - is the number of days into the future to forecast
        """

        cache_changepoint_prior_scale = self.ticker + '_best_changepoint_prior_scale'
        if not cache_changepoint_prior_scale in self.cache:
            print('No optimal changepoint_prior_scale was found in cache')
            optimal_forecast = self.make_forecast_finding_best_changepoint_prior_scale2()
            print('Results were for ticker', self.ticker)
            self.write_cache(optimal_forecast['changepoint_prior_scale'])
            self.cache.set(
                cache_changepoint_prior_scale,
                optimal_forecast['changepoint_prior_scale'],
                expire = self.cache_expire
            )

            # Calculate deltas
            delta = optimal_forecast['forecast_info']['forecast']['yhat'].pct_change()
            optimal_forecast['forecast_info']['forecast'] = optimal_forecast['forecast_info']['forecast'].assign(delta = delta.values)

            # Ensure df_performance is in forecast_info for make_graphs
            optimal_forecast['forecast_info']['df_performance'] = optimal_forecast['diagnostics']['df_performance']

            fig_paths = self.make_graphs(optimal_forecast['forecast_info'])
            result = {
                'params_info': optimal_forecast['forecast_info']['params_info'],
                'forecast': optimal_forecast['forecast_info']['forecast'],
                'performance': optimal_forecast['diagnostics']['df_performance']
            }
        else:
            print('Using old changepoint_prior_scale found in cache')
            changepoint_prior_scale = self.cache.get(cache_changepoint_prior_scale)

            # Test the model using 25% of historical data as the horizon
            # TODO review why all of a sudden I have to do this for some tickers as I'm getting
            # ValueError: Less data than horizon after initial window. Make horizon or initial shorter.
            #horizon_days = int(len(self.stock_info['historical_data']) * 0.25)
            testPercentage = 0.25
            if self.ticker in ['ada-aud', 'btc-aud', 'doge-aud', 'dot-aud', 'eth-aud', 'ftm-aud', 'ksm-aud', 'luna1-aud', 'matic-aud', 'sol-aud']:
                testPercentage = 0.24

            horizon_days = int(len(self.stock_info['historical_data']) * testPercentage)

            forecast_info = self.make_forecast(changepoint_prior_scale)
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
        result['returns'] = {}
        result['returns']['requested_period'] = (result['forecast'].tail(1)['yhat'].values[0] / result['stock_info']['info']['currentPrice']) -1
        result['returns']['annualised'] = result['returns']['requested_period'] / result['params_info']['periods'] * 365

        self.result = result

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
        info['currentPrice'] = stock_data.history('1d')['Close'][0]
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

        # Remove outliers. That is any close price value that is greater than 8 standard deviations
        outliers = historical_data[np.abs(historical_data.Close-historical_data.Close.mean()) > (8*historical_data.Close.std())].Close
        print('Close price values removed as outliers from historical data:', outliers)
        historical_data = historical_data[np.abs(historical_data.Close-historical_data.Close.mean()) <= (8*historical_data.Close.std())]

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

    def make_forecast_finding_best_changepoint_prior_scale2(self):
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

        This method evaluates all point prior scale from 0.01 until 0.5 with a 0.01 step,
        choosing the value producing the minimum Mean Absolute Percent Error (MAPE). As it
        evaluates every sinlge point it finds the absolute minimum in the range evaluated
        at the cost of speed.

        Inputs:
        historical_data - is the historical stock data
        periods - is the number of days to forecast
        """

        start = time.time()
        # Test the model using 25% of historical data as the horizon
        # TODO review why all of a sudden I have to do this for some tickers as I'm getting
        # ValueError: Less data than horizon after initial window. Make horizon or initial shorter.
        #horizon_days = int(len(self.stock_info['historical_data']) * 0.25)
        testPercentage = 0.25
        if self.ticker in ['ada-aud', 'btc-aud', 'doge-aud', 'dot-aud', 'eth-aud', 'ftm-aud', 'ksm-aud', 'luna1-aud', 'matic-aud', 'sol-aud']:
            testPercentage = 0.24

        horizon_days = int(len(self.stock_info['historical_data']) * testPercentage)

        stats = []
        result_min_mape = {'mape': 9999999}

        # Loop from 0.01 to 0.5. n.arange doesn't include the stop, but the element before.
        for changepoint_prior_scale in np.arange(0.01, 0.51, 0.01):
            # for changepoint_prior_scale in np.arange(0.01, 0.02, 0.01):
            forecast_info = self.make_forecast(changepoint_prior_scale)
            forecast_info['params_info']['horizon_days'] = horizon_days

            diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
            forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
            mape = diagnostics['df_performance'].tail(1).mape.values[0]
            print('Evaluating ' + self.ticker + ' with changepoint_prior_scale=', changepoint_prior_scale)

            stat = {
                'changepoint_prior_scale': changepoint_prior_scale,
                'mape': mape
            }
            stats.append(stat)
            print(pd.DataFrame(stats).reindex(
                columns=['changepoint_prior_scale', 'mape']))

            if mape < result_min_mape['mape']:
                result_min_mape = {
                    'forecast_info': forecast_info,
                    'diagnostics': diagnostics,
                    'changepoint_prior_scale': changepoint_prior_scale,
                    'mape': mape
                }

            print('min mape so far =', result_min_mape['mape'])
            print('with changepoint_prior_scale=',
                result_min_mape['changepoint_prior_scale'])
            print('time=', time.time() - start)

        print('best changepoint_prior_scale=',
            result_min_mape['changepoint_prior_scale'])
        print('min mape=', result_min_mape['mape'])
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

    def make_forecast(self, changepoint_prior_scale=0.05):
        """
        Forecast the price of the stock on a future number of days
        Inputs:
        historical_data - is the historical stock data
        periods - is the number of days to forecast
        """

        # Prophet requires the dates (ds) and adjusted closing prices (y)
        # Create new data frame with the required data
        df_historical_data = pd.DataFrame()
        df_historical_data['ds'] = self.stock_info['historical_data'].index.values
        df_historical_data['y'] = self.stock_info['historical_data']['Close'].values

        # Set minimum posible value
        # df_historical_data['floor'] = 0

        # Create a Prophet model
        # As there is one single closing price daily, disable the daily seasonality
        model = Prophet(
            daily_seasonality=False,
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

        # model.add_country_holidays(country_name='AU')
        model.fit(df_historical_data)

        total_future = model.make_future_dataframe(self.periods, freq='D')

        #total_future['floor'] = 0
        #total_future['cap'] = 1.2 * df_historical_data['y'].max()

        # Check if there is data only on business days and if so remove weekends in the future
        if all(df_historical_data['ds'].dt.dayofweek.unique() < 5):
            # As the stock exchange is closed on weekends, remove weekends in the future
            future_days = total_future[total_future['ds'].dt.dayofweek < 5]
        else:
            # Some markets are open all the time e.g. crypto
            future_days = total_future

        # Recalculate number of available periods to display in case that some days were removed
        future_weekdays_count = self.periods - \
            (len(total_future) - len(future_days))

        full_forecast = model.predict(future_days)

        #model.history['y'] = np.exp(model.history['y']) -1
        #df_historical_data['y'] = np.exp(df_historical_data['y']) -1
        #for col in ['yhat', 'yhat_lower', 'yhat_upper', 'trend']:
        #    full_forecast[col] = np.exp(full_forecast[col]) -1

        # Return requested period
        # forecast = full_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(available_periods+1)
        forecast = full_forecast.tail(future_weekdays_count+1)

        result = {
            'historical_data': df_historical_data,
            'full_forecast': full_forecast,
            'forecast': forecast,
            'model': model,
            'params_info': {
                'periods': self.periods,
                'historical_periods': len(self.stock_info['historical_data']),
                'weekday_periods': future_weekdays_count,
                'changepoint_prior_scale': changepoint_prior_scale,
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
        print('horizon',horizon)

        df_cross_validation = cross_validation(model, horizon=horizon, parallel='processes')

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
        # Save graphs
        fig_location = '/static/img/figures/'
        fig_paths = {
            'price': fig_location + 'price_' + self.ticker + '.svg',
            'components': fig_location + 'components_' + self.ticker + '.svg',
            # 'forecast': fig_location + 'forecast_' + ticker + '.svg',
            'mape': fig_location + 'mape_' + self.ticker + '.svg'
        }

        date_now = datetime.now()

        # Price & Forecast
        fig_price = plt.figure(facecolor='w', figsize=(10, 6))

        plt.title(self.ticker + ' - close price & forecast', fontsize=10, pad=1)
        plt.xlabel('Day (ds)', fontsize=10)
        plt.ylabel('Price (y)', fontsize=10)

        # plot changes in price and significate changes in price
        for changepoint in stock_data['model'].changepoints:
            plt.axvline(changepoint, color="lightsalmon", linestyle=":")

        signif_changepoint_threshold = 0.01
        signif_changepoints = stock_data['model'].changepoints[np.abs(np.nanmean(
            stock_data['model'].params['delta'], axis=0)) >= signif_changepoint_threshold] if len(stock_data['model'].changepoints) > 0 else []
        for signif_changepoint in signif_changepoints:
            plt.axvline(signif_changepoint, color='r', linestyle=':')

        # plot trend
        plt.plot(stock_data['full_forecast']['ds'],
                stock_data['full_forecast']['trend'], color='r')

        # plot historical data
        plt.plot(stock_data['historical_data']['ds'],
                stock_data['historical_data']['y'], color='k', linewidth=1)

        # plot forecast
        #@todo find a way to limit lower forecasted value to zero
        # in the meantime limit the y axis to zero
        if stock_data['full_forecast']['yhat_lower'].min() < 0:
            plt.ylim(0, stock_data['full_forecast']['yhat_upper'].max())

        plt.plot(
            stock_data['full_forecast']['ds'],
            stock_data['full_forecast']['yhat']
        )
        plt.fill_between(
            stock_data['full_forecast']['ds'], stock_data['full_forecast']['yhat_lower'],
            stock_data['full_forecast']['yhat_upper'], color='#0072B2', alpha=0.2
        )

        # plot today line
        plt.axvline(date_now, color='silver', linestyle=':')

        # plot grid
        plt.grid(True, which='major', color='gray', linestyle='-', linewidth=1, alpha=0.2)

        fig_price.savefig('../app' + fig_paths['price'])

        # Forecast
        # fig_forecast = stock_data['model'].plot(stock_data['full_forecast'])
        # add_changepoints_to_plot(
        #     fig_forecast.gca(), stock_data['model'], stock_data['full_forecast'])
        # plt.margins(x=0)
        # plt.title(ticker + ' price forecast', fontsize=10, pad=1)
        # plt.xlabel('Day (ds)', fontsize=10)
        # plt.ylabel('Price (y)', fontsize=10)
        # plt.axvline(date_now, color='k', linestyle=':')
        # fig_forecast.savefig('../app' + fig_paths['forecast'])

        # Components
        stock_data['model'].plot_components(stock_data['full_forecast']).savefig(
            '../app' + fig_paths['components'])

        # Performance - Cross validation of the percentage error (MAPE)
        plot_cross_validation_metric(stock_data['df_cross_validation'], metric='mape').savefig(
            '../app' + fig_paths['mape'])

        plt.close('all')

        # --- PLOTLY GENERATION ---
        
        # Helper to convert Series/Index to list of clean float or string dates
        def to_list(vals, is_date=False):
            if is_date:
                return pd.to_datetime(vals).dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
            return vals.values.astype(float).tolist()

        # 1. Price vs Forecast Graph
        fig_price = go.Figure()
        
        hist_ds = to_list(stock_data['historical_data']['ds'], is_date=True)
        fcast_ds = to_list(stock_data['full_forecast']['ds'], is_date=True)

        # Trend (Bottom Layer)
        fig_price.add_trace(go.Scatter(
            x=fcast_ds,
            y=to_list(stock_data['full_forecast']['trend']),
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2)
        ))

        # Historical Price
        fig_price.add_trace(go.Scatter(
            x=hist_ds,
            y=to_list(stock_data['historical_data']['y']),
            mode='lines',
            name='Historical Price',
            line=dict(color='black', width=2)
        ))

        # Forecast
        fig_price.add_trace(go.Scatter(
            x=fcast_ds,
            y=to_list(stock_data['full_forecast']['yhat']),
            mode='lines',
            name='Forecast',
            line=dict(color='#1f77b4', width=2)
        ))

        # Uncertainty Interval
        yhat_upper = to_list(stock_data['full_forecast']['yhat_upper'])
        yhat_lower = to_list(stock_data['full_forecast']['yhat_lower'])
        fig_price.add_trace(go.Scatter(
            x=fcast_ds + fcast_ds[::-1],
            y=yhat_upper + yhat_lower[::-1],
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name='Uncertainty Interval'
        ))

        # Add vertical line for "Today"
        date_now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig_price.add_vline(x=date_now_str, line_width=1, line_dash="dot", line_color="silver")
        
        # Changepoints
        for changepoint in stock_data['model'].changepoints:
            cp_str = pd.Timestamp(changepoint).strftime('%Y-%m-%d %H:%M:%S')
            fig_price.add_vline(x=cp_str, line_width=1, line_dash="dot", line_color="lightsalmon", opacity=0.5)

        # Significant Changepoints
        signif_changepoint_threshold = 0.01
        signif_changepoints = stock_data['model'].changepoints[np.abs(np.nanmean(
            stock_data['model'].params['delta'], axis=0)) >= signif_changepoint_threshold] if len(stock_data['model'].changepoints) > 0 else []
        for signif_changepoint in signif_changepoints:
            scp_str = pd.Timestamp(signif_changepoint).strftime('%Y-%m-%d %H:%M:%S')
            fig_price.add_vline(x=scp_str, line_width=2, line_dash="dot", line_color="red")

        # Layout updates
        layout_args = dict(
            title=dict(text=f"{self.ticker.upper()} - Close Price & Forecast", font=dict(size=20)),
            xaxis_title="Day (ds)",
            yaxis_title="Price (y)",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Limit Y axis to 0 if lower bound is negative (redundant with rangemode='tozero' but keeps safety)
        if stock_data['full_forecast']['yhat_lower'].min() < 0:
            layout_args['yaxis'] = dict(range=[0, stock_data['full_forecast']['yhat_upper'].max()])

        fig_price.update_layout(**layout_args)

        # 2. Components Graph (Manual Construction to ensure data integrity)
        from plotly.subplots import make_subplots
        
        # Identify available components
        fcst = stock_data['full_forecast']
        available_components = []
        if 'trend' in fcst.columns: available_components.append('trend')
        if 'weekly' in fcst.columns: available_components.append('weekly')
        if 'yearly' in fcst.columns: available_components.append('yearly')
        
        if available_components:
            fig_components = make_subplots(
                rows=len(available_components), 
                cols=1, 
                subplot_titles=[c.title() for c in available_components],
                vertical_spacing=0.15 # Increased spacing
            )
            
            for i, comp in enumerate(available_components):
                if comp == 'trend':
                    x_data = to_list(fcst['ds'], is_date=True)
                    y_data = to_list(fcst['trend'])
                    
                    # Add Trend uncertainty band (trend_lower/upper)
                    if 'trend_lower' in fcst.columns and 'trend_upper' in fcst.columns:
                        t_upper = to_list(fcst['trend_upper'])
                        t_lower = to_list(fcst['trend_lower'])
                        fig_components.add_trace(
                            go.Scatter(
                                x=x_data + x_data[::-1],
                                y=t_upper + t_lower[::-1],
                                fill='toself',
                                fillcolor='rgba(0, 114, 178, 0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                hoverinfo="skip",
                                showlegend=False,
                                name='Trend Uncertainty'
                            ),
                            row=i+1, col=1
                        )

                    fig_components.add_trace(
                        go.Scatter(
                            x=x_data,
                            y=y_data,
                            name='Trend',
                            mode='lines',
                            line=dict(color='#0072B2', width=2)
                        ),
                        row=i+1, col=1
                    )

                elif comp == 'weekly':
                    # Extract pure weekly component using synthetic dates (Sunday to Saturday)
                    # Use a known Sunday (2017-01-01 was a Sunday)
                    days = pd.date_range(start='2017-01-01', periods=7, freq='D')
                    df_comp = pd.DataFrame({'ds': days})
                    # Use the model's prediction logic for exact component extraction
                    fcst_comp = stock_data['model'].predict(df_comp)
                    
                    days_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
                    # Prophet's result might be in day names or numeric; predict returns the component directly
                    y_vals = to_list(fcst_comp['weekly'])
                    
                    fig_components.add_trace(
                        go.Scatter(
                            x=days_order,
                            y=y_vals,
                            name='Weekly Seasonality',
                            mode='lines',
                            line=dict(color='#0072B2', width=2)
                        ),
                        row=i+1, col=1
                    )
                
                elif comp == 'yearly':
                    # Extract pure yearly component (Jan 1 to Dec 31)
                    days = pd.date_range(start='2017-01-01', periods=366, freq='D')
                    df_comp = pd.DataFrame({'ds': days})
                    fcst_comp = stock_data['model'].predict(df_comp)
                    
                    y_x_labels = days.strftime('%B %d').tolist()
                    y_y_vals = to_list(fcst_comp['yearly'])
                    
                    fig_components.add_trace(
                        go.Scatter(
                            x=list(range(len(y_x_labels))), 
                            y=y_y_vals,
                            name='Yearly Seasonality',
                            mode='lines',
                            line=dict(color='#0072B2', width=2),
                            text=y_x_labels,
                            hoverinfo="text+y"
                        ),
                        row=i+1, col=1
                    )
                    fig_components.update_xaxes(
                        tickvals=list(range(0, len(y_x_labels), 30)),
                        ticktext=[y_x_labels[k] for k in range(0, len(y_x_labels), 30)],
                        row=i+1, col=1
                    )

                if comp != 'trend':
                    fig_components.add_hline(y=0, row=i+1, col=1, line_dash="dash", line_color="gray")

            fig_components.update_layout(
                title=dict(text=f"{self.ticker.upper()} - Model Components (Interactive)", font=dict(size=20)),
                height=300 * len(available_components) + 100,
                template="plotly_white",
                showlegend=False,
                margin=dict(l=50, r=20, t=80, b=50)
            )
        else:
            fig_components = go.Figure()
            fig_components.update_layout(title="No components available")

        # 3. MAPE Analysis
        fig_mape = go.Figure()
        
        if stock_data['df_cross_validation'] is not None:
             df_cv = stock_data['df_cross_validation'].copy()
             # Calculate MAPE
             df_cv['mape'] = np.abs((df_cv['y'] - df_cv['yhat']) / df_cv['y'])
             # Calculate horizon
             df_cv['horizon'] = df_cv['ds'] - df_cv['cutoff']
             
             # Clean data for MAPE
             mape_x = df_cv['horizon'].dt.days.values.astype(int).tolist()
             mape_y = to_list(df_cv['mape'])
             mape_y = [None if math.isnan(v) or math.isinf(v) else v for v in mape_y]
             
             # Scatter plot of MAPE vs Horizon
             fig_mape.add_trace(go.Scatter(
                x=mape_x,
                y=mape_y,
                mode='markers',
                marker=dict(color='gray', size=5, opacity=0.6),
                name='Cross Validation Errors'
             ))
             
             # Mean MAPE Line
             if stock_data['df_performance'] is not None:
                  df_p = stock_data['df_performance']
                  p_x = df_p['horizon'].dt.days.values.astype(int).tolist()
                  p_y = to_list(df_p['mape'])
                  p_y = [None if math.isnan(v) or math.isinf(v) else v for v in p_y]
                  
                  fig_mape.add_trace(go.Scatter(
                        x=p_x,
                        y=p_y,
                        mode='lines',
                        line=dict(color='#0072B2', width=3),
                        name='Mean MAPE'
                  ))

             fig_mape.update_layout(
                title=dict(text=f"{self.ticker.upper()} - Cross Validation Error (MAPE) (Plotly)", font=dict(size=20)),
                xaxis_title="Horizon (Days)",
                yaxis_title="MAPE",
                template="plotly_white",
                margin=dict(l=50, r=20, t=50, b=50)
             )
        
        # Clean dict helper (Simple now as we pre-converted)
        def _clean_simple(obj):
            if isinstance(obj, dict):
                return {k: _clean_simple(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean_simple(x) for x in obj]
            if isinstance(obj, (float, np.floating)):
                if math.isnan(obj) or math.isinf(obj): return None
                return float(obj)
            return obj

        fig_price_dict = _clean_simple(fig_price.to_dict())
        fig_components_dict = _clean_simple(fig_components.to_dict())
        fig_mape_dict = _clean_simple(fig_mape.to_dict())

        # Standard serialization
        graphJSON_price = json.dumps(fig_price_dict, cls=plotly.utils.PlotlyJSONEncoder)
        graphJSON_components = json.dumps(fig_components_dict)
        graphJSON_mape = json.dumps(fig_mape_dict)

        # Add Plotly JSONs to fig_paths
        fig_paths['plotly_price'] = graphJSON_price
        fig_paths['plotly_components'] = graphJSON_components
        fig_paths['plotly_mape'] = graphJSON_mape

        return fig_paths
