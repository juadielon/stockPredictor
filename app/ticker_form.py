from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class TickerForm(FlaskForm):
    ticker = StringField('ticker', validators=[DataRequired()])
    days = StringField('days', validators=[DataRequired()])
    submit = SubmitField('submit')