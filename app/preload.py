import click
from filelock import Timeout

from app.stock_predictor import StockPredictor


@click.command('preload')
@click.option('--days', default=365, show_default=True, type=click.IntRange(1, 730))
@click.option('--ticker', 'tickers', multiple=True, help='Refresh only these tickers; repeat for more than one.')
@click.option('--retune', is_flag=True, help='Run the expensive parameter search before saving forecasts.')
def preload_command(days, tickers, retune):
    """Refresh saved forecasts, reusing model settings unless --retune is given."""
    predictor = None
    try:
        predictor = StockPredictor()
        summary = predictor.preload(days, retune=retune, tickers=tickers or None)
    except Timeout as error:
        raise click.ClickException('Another preload batch is already running.') from error
    except Exception as error:
        raise click.ClickException(str(error)) from error
    finally:
        if predictor is not None:
            predictor.cache.close()
    click.echo(f"Finished: {len(summary['succeeded'])} refreshed, {len(summary['failed'])} failed.")
    if summary['failed']:
        raise click.ClickException('Failed tickers: ' + ', '.join(summary['failed']))