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

Uwsgi restarts everytime the uwsgi.ini file is changed. So everytime a file is changed simply do the following for the changes to take effect:

```
touch uwsgi.ini
```

Alternatively, the docker container can be restarted with the following script:

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
