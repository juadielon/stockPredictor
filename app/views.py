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

   forecast = forecaster(form.ticker.data, form.days.data)   
   print(forecast)
   forecast = forecast.itertuples()
   
   return render_template('results.html', ticker=form.ticker.data, days=form.days.data, forecast=forecast)
