import yfinance as yf
import pandas as pd
from fbprophet import Prophet

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
    m = Prophet(daily_seasonality=False)

    # m.add_country_holidays(country_name='AU')
    m.fit(df)

    future = m.make_future_dataframe(periods, freq='D')

    forecast = m.predict(future)

    m.plot_components(forecast)
    m.plot(forecast)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods+1)
    
