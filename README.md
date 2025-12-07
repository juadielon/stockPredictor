# Stock Predictor

Time series analysis to predict future stock prices.

## How to run

To install everything that is required, the first time you run it, make sure Docker has access to at least 2.5GB of memory

To install and run the container simply do:

```
./start.sh
```

Then visit http://localhost

## Update after code changes

The application is configured to automatically reload when code changes are detected (Hot Reloading). You do not need to manually restart the service for Python code changes.

However, if you need to fully restart the container (e.g., after changing `requirements` or `Dockerfile`), use:

```
./restart.sh
```

## Useful commands

Access the docker container

```
docker exec -it stock_predictor bash
```

Restart nginx

```
service nginx reload
or
supervisorctl restart nginx
```

## Scripts

### `./start.sh`
Builds the docker image, removes any previous container instances, and runs the new container on port 80. It also prunes unused docker system resources.

### `./restart.sh`
Restarts the `stock_predictor` container. Use this if you want to restart the application without rebuilding the image.

### `./preload.sh`
Triggers the pre-calculation of forecasts for tickers defined in the cache configuration. It sends a request to the `/preload` endpoint and follows the logs to show progress.

### `./prestart.sh`
Internal script executed automatically by the container during startup. It cleans up Nginx logs. You do not need to run this manually.

## Cache Configuration

### `tmp/tickers_change_point_prior_scale.json`
This file contains a list of tickers and their pre-calculated optimal `changepoint_prior_scale` parameters. The application uses this file to:
1.  Speed up forecasts by using cached parameters instead of recalculating them.
2.  Define the list of tickers to cache when running `./preload.sh`.

Example structure:
```json
[
    {
        "ticker": "ndq.ax",
        "changepoint_prior_scale": 0.01
    }
]
```
