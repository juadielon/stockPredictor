# Stock Predictor

Time series analysis to predict future stock prices.

## How forecasts work

Forecasts use Prophet, a statistical time-series model originally developed at Facebook. It fits trends and seasonal patterns to historical prices. It is not a chatbot or generative AI model, and it does not guarantee accurate returns.

- Enter 1 to 730 calendar days. The app may shorten this using its existing limit of roughly 36% of the available historical records. That limit does not guarantee accuracy.
- Predictions start after the last date in the price history, not necessarily today. For example, seven days after a Friday ends the following Friday.
- ASX forecasts skip weekends and exchange holidays. Recognised crypto tickers include every day. Other shares currently skip weekends only.
- If the end date is not a trading day, the forecast ends on the previous trading day. Requests with no future trading days are rejected.
- Returns compare the predicted final price with the last historical close adjusted for splits and dividends. The annualised figure compounds that change over a year; it is not a separate prediction or a total investment return.

**Limitations:** Data may be stale or include an unfinished day's price. Cached forecasts can be up to 12 hours old. The forecasts have not yet been shown to beat simply using the latest price.

## How to run

Start Docker and give it at least 2.5 GB of memory. Open a terminal in the project folder.

The Docker commands below work in **PowerShell and Bash**. Run them one at a time and only continue if each succeeds. You do not need Python installed on your computer.

If you use **Git Bash on Windows**, run `export MSYS_NO_PATHCONV=1` first to prevent it from changing Docker paths. This is not needed in PowerShell or Linux/WSL Bash.

1. Build the app image. The build also runs the tests.

    ```sh
    docker build -t stock_predictor .
    ```

2. Start the app. Keep the quotes around the mount argument so paths with spaces work.

    ```sh
    docker run -d --name stock_predictor -p 80:80 --mount "type=bind,source=${PWD},target=/app" stock_predictor
    ```

Then visit http://localhost

If a container named `stock_predictor` already exists, follow the rebuild steps below. If port 80 is busy, use `-p 8083:80` and visit http://localhost:8083 instead.

## Update after code changes

Python code changes reload automatically when you run the app with the project folder mounted as shown above.

To restart the existing container without rebuilding it, use:

```sh
docker restart stock_predictor
```

After changing dependencies or the Dockerfile, rebuild first. **Only continue if the build succeeds:**

```sh
docker build -t stock_predictor .
```

Then replace the existing app container. This briefly stops the app but leaves the project files and saved forecasts on your computer unchanged.

```sh
docker stop stock_predictor
docker rm stock_predictor
docker run -d --name stock_predictor -p 80:80 --mount "type=bind,source=${PWD},target=/app" stock_predictor
```

Keep any custom port mapping you used when first starting the app. If the new container fails to start, the old one is not restored automatically; fix the reported error before trying again.

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

Repeat both steps after code or dependency changes. Docker may reuse a cached test step if its inputs have not changed; the second command always runs the tests. The build may download packages; the second step uses local test data only. These tests check that the code works, not how accurately it predicts real prices.

## Useful commands

Open a Bash terminal inside the running container:

```
docker exec -it stock_predictor bash
```

Reload nginx from your usual terminal:

```sh
docker exec stock_predictor nginx -s reload
```

Run tests in the running app container:

```sh
docker exec stock_predictor pytest tests/ -v
```

## Refresh saved forecasts

The app container must be running. Choose one command; these are alternatives, not steps.

Refresh all saved tickers for 365 days:

```sh
docker exec stock_predictor flask --app app preload
```

Refresh all saved tickers for 90 days:

```sh
docker exec stock_predictor flask --app app preload --days 90
```

Refresh only selected tickers:

```sh
docker exec stock_predictor flask --app app preload --days 90 --ticker ndq.ax --ticker btc-usd
```

Refresh reuses model settings. To search for new settings too, add `--retune`. This is much slower: it tests 200 parameter combinations per ticker.

```sh
docker exec stock_predictor flask --app app preload --days 90 --ticker ndq.ax --retune
```

A 90-day refresh prepares 90-day requests, not every possible forecast period. Without `--ticker`, it includes configured tickers and those with a previous successful forecast.

Only one batch runs at a time. A failed ticker does not stop the others. The command prints a final summary and exits with a non-zero status if any ticker failed. The old `/preload` URL returns HTTP 410 and does not start work.

## Optional Bash scripts

These `.sh` shortcuts require Bash; they do not run directly in PowerShell. On Windows, use Git Bash with Docker Desktop or WSL with Docker integration enabled. The direct Docker commands above work in PowerShell too.

Run the examples from the project folder. If executable permissions are unavailable, use `bash ./start.sh` (or the corresponding script name).

### `./start.sh`
Builds the image, then creates or replaces the `stock_predictor` container on port 80. A failed build leaves the existing app running. Replacement briefly stops the app; if startup fails afterwards, there is no automatic rollback.

The script locates the project folder itself, mounts it at `/app` and keeps saved forecasts on your computer. It handles Git Bash path conversion automatically and uses Docker's default DNS. By default, it removes only the existing `stock_predictor` container.

To use another host port:

```sh
HOST_PORT=8083 ./start.sh
```

Then visit http://localhost:8083. Supply the same `HOST_PORT` each time you recreate the container; otherwise it defaults to 80. Values must be integers from 1 to 65535. Restarting an existing container keeps its port mapping.

To also remove stopped containers after successful startup:

```sh
bash ./start.sh --prune
```

**Warning:** `--prune` deletes all stopped containers, including other projects' containers and data in their writable layers, without another confirmation. It does not prune volumes, images, networks or build cache. Running containers are not pruned. Omit this option to leave other containers alone.

Use `bash ./start.sh --help` for usage. You can combine options, for example `HOST_PORT=8083 bash ./start.sh --prune`.

### `./restart.sh`
Restarts the `stock_predictor` container. Use this if you want to restart the application without rebuilding the image.

### `./preload.sh`
Refreshes saved forecasts inside the running `stock_predictor` container. It reuses model settings, prints progress and exits when finished. It no longer calls a web endpoint.

```sh
./preload.sh --days 90 --ticker ndq.ax --ticker btc-usd
```

It accepts the same options as the Docker preload command above.

### `./test.sh`
Runs `pytest tests/ -v` inside the running `stock_predictor` container without requiring an interactive terminal. Extra arguments are passed to pytest, for example:

```sh
./test.sh -k cache
```

This is a quick check using the running app's code and installed dependencies. Use "Run tests in Docker" above for a fresh test image and a separate network-disabled test run. Tests also run as a step during image builds, unless Docker reuses that cached step.

Restart, preload and test commands return a non-zero status if Docker or the command inside the container fails. Preload and tests require a running container; restart requires an existing one.

## Saved forecasts and settings

Forecasts, diagnostics and charts are stored in `tmp/forecasts-v2/`. A matching ticker and requested horizon can reuse the saved result for up to 12 hours without downloading data or fitting the model again. Preload forces a refresh even within that period.

Model settings expire after 30 days. Refresh uses suitable saved settings or Prophet defaults; only `--retune` runs the parameter search. The results page shows which source was used. Expiry is based on when settings were saved, not when they were last read.

If refresh fails, the app can show the last successful forecast with a warning. These fallback results are kept for up to seven days, subject to cache eviction. Cached quotes and forecasts retain their original timestamps. Cache age does not guarantee that Yahoo's data is current; new prices are checked on refresh, not on a cache hit.

The cache stores the model version, data cutoff and a fingerprint of the downloaded history. Runtime cache files are excluded from Git and Docker images. Keep `tmp/` mounted when recreating the container to retain saved results; the existing start script mounts the project folder.

### `tmp/tickers_change_point_prior_scale.json`
This is the configured ticker list. Existing parameter values are imported once into the new cache and labelled `legacy`, with no known tuning date. They expire 30 days after import; rereading the file does not extend that period. To replace those settings, run preload with `--retune`.

The app does not rewrite this file. Newly requested tickers are remembered separately after a successful forecast. The old cache files are left untouched. A ticker entry without parameters uses Prophet defaults until retuned.

Example structure:
```json
[
    {
        "ticker": "ndq.ax",
        "changepoint_prior_scale": 0.01
    }
]
```
