from flask import render_template
from app import app

@app.route('/')
def home():
  form = TickerForm()
  return render_template('home.html', form=form)

@app.route('/template')
def template():
    return render_template('home.html')
