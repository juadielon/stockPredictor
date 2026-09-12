# Stock Predictor

Time series analysis to predict future stock prices.

## How forecasts work

- Enter 1 to 730 calendar days. The app may shorten this using its existing limit of roughly 36% of the available historical records. That limit does not guarantee accuracy.
- Predictions start after the last date in the price history, not necessarily today. For example, seven days after a Friday ends the following Friday.
- ASX forecasts skip weekends and exchange holidays. Recognised crypto tickers include every day. Other shares currently skip weekends only.
- If the end date is not a trading day, the forecast ends on the previous trading day. Requests with no future trading days are rejected.
- Returns compare the predicted final price with the last historical close adjusted for splits and dividends. The annualised figure compounds that change over a year; it is not a separate prediction or a total investment return.

**Limitations:** Data may be stale or include an unfinished day's price. Saved model settings may be outdated. The forecasts have not yet been shown to beat simply using the latest price.

## How to run

To install everything that is required, the first time you run it, make sure Docker has access to at least 2.5GB of memory

To install and run the container simply do:

```
./start.sh
```

Then visit http://localhost

## Update after code changes

The application is configured to automatically reload when code changes are detected (Hot Reloading). You do not need to manually restart the service for Python code changes.

To restart the existing container without rebuilding it, use:

```
./restart.sh
```

Changes to dependencies or the Dockerfile require rebuilding the image and recreating the app container; a restart alone will not apply them.

## Run tests in Docker

Run these commands from the project folder with Docker running. They do not stop or replace the running app.

1. Build a test image. This installs dependencies and runs the tests; the build fails if a test fails.

    ```sh
    docker build -t stock_predictor:test .
    ```

2. Run the tests again without network access, using the code in that image. The temporary container is removed afterwards.

    ```sh
    docker run --rm --network none --entrypoint pytest stock_predictor:test tests/ -p no:cacheprovider -q
    ```

Repeat both steps after code or dependency changes. The build may download packages; the second step uses local test data only. These tests check that the code works, not how accurately it predicts real prices.

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

### `./test.sh`
Executes the unit test suite inside the running container using `pytest`. Note that unit tests are also automatically executed inside Docker as a pre-build gate during `docker build` (in `./start.sh`).


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
