from flask import render_template, redirect
from app import app
from app.ticker_form import TickerForm
from app.stock_predictor import forecaster


@app.route('/')
def home():
    form = TickerForm()
    return render_template('home.html', form=form)


@app.route('/ticker', methods=['POST'])
def ticker():
    form = TickerForm()

    if not form.validate_on_submit():
        return redirect('/')

    forecast_info = forecaster(form.ticker.data, form.days.data)
    stock_info = forecast_info['stock_info']['info']
    now = forecast_info['stock_info']['now']
    params_info = forecast_info['params_info']
    dividends = forecast_info['stock_info']['dividends']
    forecast = forecast_info['forecast'].itertuples()
    performance = forecast_info['performance'].itertuples()
    returns = forecast_info['returns']
    return render_template('results.html', ticker=form.ticker.data, days=form.days.data, stock_info=stock_info, now=now, params_info=params_info, dividends=dividends, forecast=forecast, performance=performance, returns=returns, fig_paths=forecast_info['fig_paths'])
