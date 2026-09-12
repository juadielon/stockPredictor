from flask import render_template
from app import app
from app.ticker_form import TickerForm
from app.stock_predictor import StockPredictor

import os.path

@app.route('/')
def home():
    form = TickerForm()
    return render_template('home.html', form=form)

@app.route('/ticker', methods=['POST'])
def ticker():
    form = TickerForm()

    if not form.validate_on_submit():
        return render_template('home.html', form=form), 400

    try:
        forecast_info = StockPredictor(form.ticker.data, form.days.data).result
    except ValueError:
        app.logger.warning('Forecast rejected for %s', form.ticker.data, exc_info=True)
        return render_template(
            'home.html', form=form,
            error='Unable to forecast this ticker and period. Check the ticker, available history and trading dates.'
        ), 422

    return render_template(
        'results.html',
        ticker = form.ticker.data,
        days = form.days.data,
        stock_info = forecast_info['stock_info']['info'],
        now = forecast_info['stock_info']['now'],
        params_info = forecast_info['params_info'],
        dividends = forecast_info['stock_info']['dividends'],
        forecast = forecast_info['forecast'].itertuples(),
        performance = forecast_info['performance'].itertuples(),
        returns = forecast_info['returns'],
        fig_paths = forecast_info['fig_paths']
    )

@app.route('/preload')
def preload():
    """
    Read previously requested tickers and cache them
    """
    StockPredictor().preload()
    return 'Working ... Check the container logs (docker logs -f stock_predictor)'