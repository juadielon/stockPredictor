import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.plot import add_changepoints_to_plot
from datetime import datetime


def forecaster(ticker, periods):
    """
    Forecast the given ticker/quote a number of days into the future from today

    Inputs:
    ticker - is the ticker/quote of the stock as defined by Yahoo Finance
    periods - is the number of days into the future to forecast
    """

    stock_info = get_stock_info(ticker)

    changepoint_prior_scale = find_best_changepoint_prior_scale(
        stock_info['historical_data'], periods)
    model_info = {'change_point_prior_scale': changepoint_prior_scale}

    forecast_info = make_forecast(
        stock_info['historical_data'], periods, changepoint_prior_scale)
    diagnostics = diagnose_model(periods, forecast_info['model'])
    forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
    fig_paths = make_graphs(ticker, forecast_info)

    return {'stock_info': stock_info, 'forecast': forecast_info['forecast'], 'model_info': model_info, 'performance': diagnostics['df_performance'], 'fig_paths': fig_paths}


def get_stock_info(ticker):
    """
    Retrieves stock's information from Yahoo Finance

    Inputs:
    ticker - is the ticker/quote of the stock as defined by Yahoo Finance
    """
    # Get historical data from Yahoo Finance
    stock_data = yf.Ticker(ticker)

    info = stock_data.info
    dividends = stock_data.dividends

    # Yahoo Finance allows to retrieve historical data for:
    # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    historical_data = stock_data.history('max', auto_adjust=True)

    return {'info': info, 'dividends': dividends, 'historical_data': historical_data}


def find_best_changepoint_prior_scale(historical_data, periods):
    """
    Find the best changepoint prior scale to use
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
    while continue_loop:
        changepoint_prior_scale += 0.01
        forecast_info = make_forecast(
            historical_data, periods, changepoint_prior_scale)
        diagnostics = diagnose_model(periods, forecast_info['model'])
        mape = diagnostics['df_performance'].tail(1).mape.values
        print('mape=', mape)
        print('temp changepoint_prior_scale=', changepoint_prior_scale)
        if mape < min_mape:
            min_mape = mape
        else:
            continue_loop = False
            changepoint_prior_scale -= 0.01
    print('min_mape=', min_mape)
    print('best changepoint_prior_scale=', changepoint_prior_scale)
    return round(changepoint_prior_scale, 2)


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
    future = total_future[total_future['ds'].dt.dayofweek < 5]

    # As some days were removed, recalculate number of available periods to display
    available_periods = periods - (len(total_future) - len(future))
    full_forecast = model.predict(future)

    # Return requested period
    # forecast = full_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(available_periods+1)
    forecast = full_forecast.tail(available_periods+1)

    return {'historical_data': df_historical_data, 'full_forecast': full_forecast, 'forecast': forecast, 'model': model}
    # return {'forecast': forecast, 'performance': df_p, 'fig_paths': fig_paths}


def diagnose_model(periods, model):
    """
    Diagnose the model

    Inputs:
    periods - is the number of forcasted days
    model - is the Phropet model
    """

    horizon = str(periods) + ' days'
    df_cross_validation = cross_validation(
        model, horizon=horizon, parallel='processes')
    df_performance = performance_metrics(df_cross_validation)
    print(df_performance)
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
        'forecast': fig_location + 'forecast_' + ticker + '.png',
        'mape': fig_location + 'mape_' + ticker + '.png'
    }

    # Price
    fig_price = plt.figure(facecolor='w', figsize=(20, 6))

    # plot significate changes in price
    change_points = stock_data['model'].changepoints
    for change in change_points:
        plt.axvline(change, color="c", linestyle=":")

    plt.title(ticker + ' close price', fontsize=20)
    plt.ylabel('Price', fontsize=15)
    plt.plot(stock_data['historical_data']['ds'],
             stock_data['historical_data']['y'])
    fig_price.savefig('../app' + fig_paths['price'])

    # Forecast
    # model.plot(forecast).savefig('../app' + fig_paths['forecast'])
    fig_forecast = stock_data['model'].plot(stock_data['full_forecast'])
    add_changepoints_to_plot(
        fig_forecast.gca(), stock_data['model'], stock_data['full_forecast'])
    # plt.margins(x=0)
    plt.title(ticker + ' forecast', fontsize=10, pad=1)
    plt.xlabel('Day (ds)', fontsize=10)
    plt.ylabel('Price (y)', fontsize=10)
    date_now = datetime.now()
    plt.axvline(date_now, color="k", linestyle=":")
    fig_forecast.savefig('../app' + fig_paths['forecast'])

    # Components
    stock_data['model'].plot_components(stock_data['full_forecast']).savefig(
        '../app' + fig_paths['components'])

    # Performance - Cross validation of the percentage error (MAPE)
    plot_cross_validation_metric(stock_data['df_cross_validation'], metric='mape').savefig(
        '../app' + fig_paths['mape'])

    plt.close('all')

    return fig_paths
