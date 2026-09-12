import pytest
from app import app
from app.ticker_form import TickerForm

@pytest.fixture(autouse=True)
def disable_csrf():
    """Disable CSRF protection during unit testing."""
    previous_csrf = app.config.get('WTF_CSRF_ENABLED', True)
    app.config['WTF_CSRF_ENABLED'] = False
    yield
    app.config['WTF_CSRF_ENABLED'] = previous_csrf

def test_ticker_form_valid():
    """Verify that valid ticker and days inputs pass validation."""
    with app.test_request_context(method='POST', data={'ticker': 'aapl', 'days': '365'}):
        form = TickerForm()
        assert form.validate() is True
        assert form.ticker.data == 'aapl'
        assert form.days.data == 365

@pytest.mark.parametrize('days', ['abc', '0', '-1', '1.5', '731'])
def test_ticker_form_invalid_days(days):
    with app.test_request_context(method='POST', data={'ticker': 'aapl', 'days': days}):
        form = TickerForm()
        assert form.validate() is False
        assert 'days' in form.errors

def test_ticker_form_normalises_ticker():
    with app.test_request_context(method='POST', data={'ticker': '  CBA.AX  ', 'days': '30'}):
        form = TickerForm()
        assert form.validate() is True
        assert form.ticker.data == 'cba.ax'

def test_ticker_form_missing_ticker():
    """Verify that missing ticker raises a validation error."""
    with app.test_request_context(method='POST', data={'days': '365'}):
        form = TickerForm()
        assert form.validate() is False
        assert 'ticker' in form.errors

def test_ticker_form_missing_days():
    """Verify that missing forecast days raises a validation error."""
    with app.test_request_context(method='POST', data={'ticker': 'aapl'}):
        form = TickerForm()
        assert form.validate() is False
        assert 'days' in form.errors

def test_ticker_form_empty():
    """Verify that empty form submission is invalid."""
    with app.test_request_context(method='POST', data={}):
        form = TickerForm()
        assert form.validate() is False

