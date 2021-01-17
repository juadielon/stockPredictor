import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from datetime import datetime


def forecaster(ticker, periods):
    """
    Forecast the given ticker/quote a number of days into the future from today

    Inputs:
    ticker - is the ticker/quote of the stock as defined by Yahoo Finance
    periods - is the number of days into the future to forecast
    """

    periods = int(periods)

    # Get historical data from Yahoo Finance
    stock_data = yf.Ticker(ticker)
    # Yahoo Finance allows to retrieve historical data for:
    # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    historical_data = stock_data.history('max', auto_adjust=True)

    # Prophet requires the dates (ds) and adjusted closing prices (y)
    # Create new data frame with the required data
    df = pd.DataFrame()
    df['ds'] = historical_data.index.values
    df['y'] = historical_data['Close'].values
    # Set minimum posible value
    # df['floor'] = 0

    # Create a Prophet model
    # As there is one single closing price daily, disable the daily seasonality
    model = Prophet(daily_seasonality=False)

    # m.add_country_holidays(country_name='AU')
    model.fit(df)

    future = model.make_future_dataframe(periods, freq='D')
    # As the stock exchange is closed on weekends, remove weekends in the future
    new_future = future[future['ds'].dt.dayofweek < 5]
    # As some days were removed, recalculate number of available periods to display
    available_periods = periods - (len(future) - len(new_future))
    forecast = model.predict(new_future)

    # Diagnostics
    horizon = str(periods) + ' days'
    df_cv = cross_validation(model, horizon=horizon)
    df_p = performance_metrics(df_cv)
    print(df_p)

    # Save graphs
    fig_location = '/static/img/figures/'
    fig_paths = {
        'price': fig_location + 'price_' + ticker + '.png',
        'components': fig_location + 'components_' + ticker + '.png',
        'forecast': fig_location + 'forecast_' + ticker + '.png'
    }

    # Price
    fig_price = plt.figure(facecolor='w', figsize=(20, 6))
    plt.title(ticker + ' close price', fontsize=20)
    plt.ylabel('Price', fontsize=15)
    plt.plot(df['ds'], df['y'])
    fig_price.savefig('../app' + fig_paths['price'])

    # Components
    model.plot_components(forecast).savefig('../app' + fig_paths['components'])

    # Forecast
    # model.plot(forecast).savefig('../app' + fig_paths['forecast'])
    fig_forecast = model.plot(forecast)
    # plt.margins(x=0)
    plt.title(ticker + ' forecast', fontsize=10, pad=1)
    plt.xlabel('Day (ds)', fontsize=10)
    plt.ylabel('Price (y)', fontsize=10)
    date_now = datetime.now()
    plt.axvline(date_now, color="k", linestyle=":")
    fig_forecast.savefig('../app' + fig_paths['forecast'])

    # Return requested period
    # forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(available_periods+1)
    forecast = forecast.tail(available_periods+1)

    return {'forecast': forecast, 'performance': df_p, 'fig_paths': fig_paths}
