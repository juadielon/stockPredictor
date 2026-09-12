import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from diskcache import Cache
from filelock import FileLock


class ForecastCache:
    model_version = 'prophet-v2'
    forecast_lifetime = 12 * 60 * 60
    parameter_lifetime = 30 * 24 * 60 * 60
    retention = 7 * 24 * 60 * 60

    def __init__(self, directory='./tmp/forecasts-v2'):
        self.directory = Path(directory)
        self.store = Cache(str(self.directory), timeout=20)

    def key(self, kind, ticker, periods):
        return (self.model_version, kind, ticker.strip().lower(), periods)

    def lock(self, ticker, periods):
        identity = repr(self.key('work', ticker, periods)).encode('utf-8')
        name = hashlib.sha256(identity).hexdigest()
        return FileLock(str(self.directory / (name + '.lock')), timeout=0)

    def parameters(self, ticker, periods):
        entry = self.store.get(self.key('parameters', ticker, periods))
        if entry is None or entry['expires_at'] <= time.time():
            return None
        return entry

    def save_parameters(self, ticker, periods, cps, sps, data_cutoff=None, source='tuned'):
        cps, sps = float(cps), float(sps)
        if not all(math.isfinite(value) and value > 0 for value in (cps, sps)):
            raise ValueError('Model parameters must be positive and finite.')
        now = time.time()
        entry = {
            'cps': cps, 'sps': sps, 'source': source,
            'tuned_at': now if source == 'tuned' else None,
            'imported_at': now if source == 'legacy' else None,
            'data_cutoff': data_cutoff, 'expires_at': now + self.parameter_lifetime,
        }
        self.store.set(self.key('parameters', ticker, periods), entry)
        return entry

    def import_legacy(self, entries):
        marker = (self.model_version, 'legacy-imported')
        with self.store.transact():
            if self.store.get(marker):
                return
            for entry in entries:
                if 'changepoint_prior_scale' not in entry:
                    continue
                self.save_parameters(
                    entry['ticker'], None, entry['changepoint_prior_scale'],
                    entry.get('seasonality_prior_scale', 10.0), source='legacy'
                )
            self.store.set(marker, True)

    def remember_ticker(self, ticker):
        with self.store.transact():
            tickers = set(self.store.get('requested-tickers', []))
            tickers.add(ticker.strip().lower())
            self.store.set('requested-tickers', sorted(tickers))

    def tickers(self, path):
        entries = []
        if Path(path).is_file():
            with open(path, encoding='utf-8') as source:
                entries = json.load(source)
        tickers = {entry['ticker'].strip().lower() for entry in entries}
        return sorted(tickers | set(self.store.get('requested-tickers', [])))

    def result(self, ticker, periods, allow_stale=False):
        entry = self.store.get(self.key('result', ticker, periods))
        if entry is None:
            return None
        stale = entry['cache_info']['expires_at'] <= time.time()
        if stale and not allow_stale:
            return None
        entry['cache_info']['stale'] = stale
        return entry

    def save_result(self, ticker, periods, result):
        now = time.time()
        parameters = result['cache_info'].get('parameters', {})
        expires_at = min(now + self.forecast_lifetime,
                         parameters.get('expires_at', now + self.forecast_lifetime))
        result['cache_info'].update({
            'generated_at': now, 'expires_at': expires_at,
            'generated_at_label': datetime.fromtimestamp(now, timezone.utc).strftime('%d/%m/%Y %H:%M UTC'),
            'stale': False, 'model_version': self.model_version,
            'requested_days': periods,
        })
        self.store.set(self.key('result', ticker, periods), result, expire=self.retention)

    def close(self):
        self.store.close()