import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.plot import add_changepoints_to_plot
from datetime import datetime
import numpy as np
from diskcache import FanoutCache
# from diskcache import Cache

import time

cache = FanoutCache(directory='./tmp', timeout=20, shards=4)
# cache = FanoutCache(directory='./tmp', timeout=20, eviction_policy='none')
# cache = Cache(directory='./tmp')
# cache.clear()
# Prime cache

"""my_cache = [
    {'ticker': 'a200.ax', 'changepoint_prior_scale': 0.07},
    # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.38},
    {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.5},
    # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.21},
    {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.03},
    {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.01},
    {'ticker': 'espo', 'changepoint_prior_scale': 0.34},
    {'ticker': 'ethi.ax', 'changepoint_prior_scale': 0.02},
    {'ticker': 'hack.ax', 'changepoint_prior_scale': 0.01},
    {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.14},
    {'ticker': 'ivv.ax', 'changepoint_prior_scale': 0.01},
    # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.37},
    {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.18},
    # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.01},
    {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.14},
    {'ticker': 'rbtz.ax', 'changepoint_prior_scale': 0.01},
    {'ticker': 'tech.ax', 'changepoint_prior_scale': 0.5}
]
expire = 60 * 60  # 1 hour
for index in range(len(my_cache)):
    cache.set(my_cache[index]['ticker'] + '_best_changepoint_prior_scale',
              my_cache[index]['changepoint_prior_scale'], expire=expire)
"""


def forecaster(ticker, periods):
    """
    Forecast the given ticker/quote a number of days into the future from today

    Inputs:
    ticker - is the ticker/quote of the stock as defined by Yahoo Finance
    periods - is the number of days into the future to forecast
    """

    stock_info = get_stock_info(ticker)
    stock_info['now'] = datetime.now()
    print(datetime.now())

    # Make sure the new number of periods to use is not bigger than 36% of the historical periods
    periods = int(periods)
    historical_periods_count = len(stock_info['historical_data'])
    # Estimate the number of maximum periods allowed. This was derived by trial and error
    max_periods = int(historical_periods_count * 0.36) + 1
    if periods > max_periods:
        periods = max_periods

    cache_changepoint_prior_scale = ticker + '_best_changepoint_prior_scale'
    if not cache_changepoint_prior_scale in cache:
        print('No optimal changepoint_prior_scale was found in cache')
        optimal_forecast = make_forecast_finding_best_changepoint_prior_scale2(
            stock_info['historical_data'], periods)
        # expire = 60 * 60 * 24 * 90 # 90 days
        expire = 60 * 60  # 1 hour
        cache.set(cache_changepoint_prior_scale,
                  optimal_forecast['changepoint_prior_scale'], expire=expire)
        fig_paths = make_graphs(ticker, optimal_forecast['forecast_info'])
        result = {
            'stock_info': stock_info,
            'params_info': optimal_forecast['forecast_info']['params_info'],
            'forecast': optimal_forecast['forecast_info']['forecast'],
            'performance': optimal_forecast['diagnostics']['df_performance'],
            'fig_paths': fig_paths
        }
    else:
        print('Using old changepoint_prior_scale found in cache')
        changepoint_prior_scale = cache.get(cache_changepoint_prior_scale)

        # Test the model using 25% of historical data as the horizon
        horizon_days = int(len(stock_info['historical_data']) * 0.25)

        forecast_info = make_forecast(
            stock_info['historical_data'], periods, changepoint_prior_scale)
        forecast_info['change_point_prior_scale'] = changepoint_prior_scale
        forecast_info['params_info']['horizon_days'] = horizon_days

        diagnostics = diagnose_model(horizon_days, forecast_info['model'])
        forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
        fig_paths = make_graphs(ticker, forecast_info)
        result = {
            'stock_info': stock_info,
            'params_info': forecast_info['params_info'],
            'forecast': forecast_info['forecast'],
            'performance': diagnostics['df_performance'],
            'fig_paths': fig_paths
        }
    return result


@cache.memoize(typed=True, expire=43200)  # cache for 12 hours
def get_stock_info(ticker):
    """
    Retrieves stock's information from Yahoo Finance

    Inputs:
    ticker - is the ticker/quote of the stock as defined by Yahoo Finance
    """

    print('Retrieving data from Yahoo Finance')

    # Get historical data from Yahoo Finance
    stock_data = yf.Ticker(ticker)

    info = stock_data.info
    # info = {'volume': 0}
    dividends = stock_data.dividends

    # Yahoo Finance allows to retrieve historical data for:
    # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    historical_data = stock_data.history('max', auto_adjust=True)

    return {'info': info, 'dividends': dividends, 'historical_data': historical_data}


def make_forecast_finding_best_changepoint_prior_scale(historical_data, periods):
    """
    Find the best changepoint prior scale to use and return the forecast
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

    Inputs:
    historical_data - is the historical stock data
    periods - is the number of days to forecast
    """
    min_mape = 100
    changepoint_prior_scale = 0
    continue_loop = True
    # Test the model using 25% of historical data as the horizon
    horizon_days = int(len(historical_data) * 0.25)

    while continue_loop:
        changepoint_prior_scale += 0.01
        forecast_info = make_forecast(
            historical_data, periods, changepoint_prior_scale)
        forecast_info['params_info']['horizon_days'] = horizon_days

        diagnostics = diagnose_model(horizon_days, forecast_info['model'])
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


def make_forecast_finding_best_changepoint_prior_scale2(historical_data, periods):
    start = time.time()
    # Test the model using 25% of historical data as the horizon
    horizon_days = int(len(historical_data) * 0.25)

    stats = []
    result_min_mape = {'mape': 100}

    # Loop from 0.01 to 0.5. n.arange doesn't include the stop, but the element before.
    for changepoint_prior_scale in np.arange(0.01, 0.51, 0.01):
        # for changepoint_prior_scale in np.arange(0.01, 0.02, 0.01):
        forecast_info = make_forecast(
            historical_data, periods, changepoint_prior_scale)
        forecast_info['params_info']['horizon_days'] = horizon_days

        diagnostics = diagnose_model(horizon_days, forecast_info['model'])
        forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
        mape = diagnostics['df_performance'].tail(1).mape.values[0]
        print('Evaluating changepoint_prior_scale=', changepoint_prior_scale)

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


def make_forecast(historical_data, periods, changepoint_prior_scale=0.05):
    """
    Forecast the price of the stock on a future number of days
    Inputs:
    historical_data - is the historical stock data
    periods - is the number of days to forecast
    """

    periods = int(periods)

    # Prophet requires the dates (ds) and adjusted closing prices (y)
    # Create new data frame with the required data
    df_historical_data = pd.DataFrame()
    df_historical_data['ds'] = historical_data.index.values
    df_historical_data['y'] = historical_data['Close'].values
    # Set minimum posible value
    # df_historical_data['floor'] = 0

    # Create a Prophet model
    # As there is one single closing price daily, disable the daily seasonality
    # model = Prophet(daily_seasonality=False)
    model = Prophet(daily_seasonality=False,
                    changepoint_prior_scale=changepoint_prior_scale)

    # model.add_country_holidays(country_name='AU')
    model.fit(df_historical_data)

    total_future = model.make_future_dataframe(periods, freq='D')

    # As the stock exchange is closed on weekends, remove weekends in the future
    future_weekdays = total_future[total_future['ds'].dt.dayofweek < 5]
    # As some days were removed, recalculate number of available periods to display
    future_weekdays_count = periods - \
        (len(total_future) - len(future_weekdays))

    full_forecast = model.predict(future_weekdays)

    # Return requested period
    # forecast = full_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(available_periods+1)
    forecast = full_forecast.tail(future_weekdays_count+1)

    result = {
        'historical_data': df_historical_data,
        'full_forecast': full_forecast,
        'forecast': forecast,
        'model': model,
        'params_info': {
            'periods': periods,
            'historical_periods': len(historical_data),
            'weekday_periods': future_weekdays_count,
            'changepoint_prior_scale': changepoint_prior_scale,
        }
    }

    return result

def diagnose_model(horizon_days, model):
    """
    Diagnose the model

    Inputs:
    horizon_days - is the number of days to use when testing the model
    model - is the Phropet model
    """

    horizon = str(horizon_days) + ' days'

    df_cross_validation = cross_validation(model, horizon=horizon, parallel='processes')

    df_performance = performance_metrics(df_cross_validation)
    # print(df_performance)

    return {'df_cross_validation': df_cross_validation, 'df_performance': df_performance}


def make_graphs(ticker, stock_data):
    """
    Create and save graphs on a directory for the browser to pickup

    Inputs:
    ticker - Is the ticker/quote of the stock as defined by Yahoo Finance
    stock_data - Dictionary containing information about the stock
    """
    # Save graphs
    fig_location = '/static/img/figures/'
    fig_paths = {
        'price': fig_location + 'price_' + ticker + '.png',
        'components': fig_location + 'components_' + ticker + '.png',
        # 'forecast': fig_location + 'forecast_' + ticker + '.png',
        'mape': fig_location + 'mape_' + ticker + '.png'
    }

    date_now = datetime.now()

    # Price & Forecast
    fig_price = plt.figure(facecolor='w', figsize=(10, 6))

    plt.title(ticker + ' - close price & forecast', fontsize=10, pad=1)
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
    plt.plot(stock_data['full_forecast']['ds'],
             stock_data['full_forecast']['yhat'])
    plt.fill_between(stock_data['full_forecast']['ds'], stock_data['full_forecast']
                     ['yhat_lower'], stock_data['full_forecast']['yhat_upper'], color='#0072B2', alpha=0.2)

    # plot today line
    plt.axvline(date_now, color='k', linestyle=':')

    # plot grid
    plt.grid(True, which='major', color='gray',
             linestyle='-', linewidth=1, alpha=0.2)

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

    return fig_paths
