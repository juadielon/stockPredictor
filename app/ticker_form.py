from flask_wtf import FlaskForm
from wtforms import IntegerField, StringField, SubmitField
from wtforms.validators import DataRequired, InputRequired, Length, NumberRange

class TickerForm(FlaskForm):
    ticker = StringField('ticker', validators=[DataRequired(), Length(max=32)],
                         filters=[lambda value: value.strip().lower() if value else value])
    days = IntegerField('days', validators=[InputRequired(), NumberRange(min=1, max=730)])
    submit = SubmitField('submit')