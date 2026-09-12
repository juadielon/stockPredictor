# Stock Predictor

Time series analysis to predict future stock prices.

## How forecasts work

Forecasts use Prophet, a statistical time-series model originally developed at Facebook. It fits trends and seasonal patterns to historical prices. It is not a chatbot or generative AI model, and it does not guarantee accurate returns.

- Enter 1 to 730 calendar days. The app may shorten this using its existing limit of roughly 36% of the available historical records. That limit does not guarantee accuracy.
- Predictions start after the last date in the price history, not necessarily today. For example, seven days after a Friday ends the following Friday.
- ASX forecasts skip weekends and exchange holidays. Recognised crypto tickers include every day. Other shares currently skip weekends only.
- If the end date is not a trading day, the forecast ends on the previous trading day. Requests with no future trading days are rejected.
- Returns compare the predicted final price with the last historical close adjusted for splits and dividends. The annualised figure compounds that change over a year; it is not a separate prediction or a total investment return.

**Limitations:** Data may be stale or include an unfinished day's price. Saved forecasts are normally reused for up to 12 hours; older results may be shown with a warning if a refresh fails. The forecasts have not yet been shown to beat simply using the latest price.

## How to run

Start Docker and give it at least 2.5 GB of memory. Open a terminal in the project folder.

You do not need to choose model settings or run preload before using the app. Start it and search for a ticker. The first search may take a few minutes while the forecast is prepared.

If you want to prepare forecasts in advance, choose your tickers and forecast period using the [preload guide](#prepare-forecasts-with-preload) below. You can keep the supplied ticker file, edit its list, or select tickers directly in the command.

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

### Quick check in the running app

To test the code and dependencies in the existing app container:

```sh
docker exec stock_predictor pytest tests/ -v
```

In Bash, `bash ./test.sh` runs the same check. You can pass extra pytest options, for example `bash ./test.sh -k cache` to run tests with "cache" in their name. The app container must be running; no interactive terminal is needed. A failed test returns a non-zero exit status.

This shortcut does not rebuild the image or run the tests without network access. Use the two steps above for that.

## Useful commands

Open a Bash terminal inside the running container:

```
docker exec -it stock_predictor bash
```

Reload nginx from your usual terminal:

```sh
docker exec stock_predictor nginx -s reload
```

## Optional Bash shortcuts for startup

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

Restart requires an existing container. These scripts return a non-zero exit status if a Docker command fails.

## Prepare forecasts with preload

Preload does the forecasting work before someone searches. It downloads price history, fits the model, creates charts and saves the results. A later search for the same ticker and period can then use the saved forecast without repeating that work.

It is optional. The website can prepare forecasts on demand, but the first search will usually be slower. Preload also refreshes forecasts that have already been saved, even if they are still recent.

### Choose your tickers and period

Before running preload, decide which tickers you want and how many calendar days to forecast. You do not need to pick the model's numerical settings yourself.

There are two ways to choose tickers:

- Add `--ticker` to the command to process only those tickers. You can repeat the option.
- Leave out `--ticker` to process the list in [tmp/tickers_change_point_prior_scale.json](tmp/tickers_change_point_prior_scale.json), plus tickers that have previously produced a successful forecast.

Each entry must have a `ticker`. When adding a new ticker, this is all you need:

```json
[
    {"ticker": "ndq.ax"},
    {"ticker": "btc-aud"}
]
```

The supplied file also has numbers beside some tickers. These are optional initial model settings. Prophet uses both settings, but you do not need to choose their values yourself: `--retune` can do that. You can keep existing entries unchanged and add new entries with just a ticker. See [how model settings are saved](#how-model-settings-are-saved) below.

Use Yahoo Finance symbols: `.ax` for ASX listings and, for example, `btc-aud` for Bitcoin priced in Australian dollars. Keep the file valid JSON, with double quotes and no trailing comma or comments.

Adding a ticker does not generate a forecast immediately. It adds it to the next preload run that has no `--ticker` option. Removing a ticker from the file does not remove it from the app's remembered list; use explicit `--ticker` options when you want full control over what runs.

Choose a period with `--days`, from 1 to 730. Without it, preload uses 365 days. A 90-day forecast is saved separately from a 365-day forecast, so choose the period you expect to search for. The app may shorten the actual forecast when there is not enough history.

### Run a normal refresh

Start the app first, then open Bash in the project folder. On Windows, use Git Bash or WSL with Docker integration enabled. The script uses the running `stock_predictor` container; it does not start it for you.

Choose the command that matches what you want to refresh. You do not need to run all three.

Refresh all configured and remembered tickers for 365 days:

```sh
bash ./preload.sh
```

Refresh that same list for 90 days:

```sh
bash ./preload.sh --days 90
```

Refresh only NDQ and Bitcoin for 90 days:

```sh
bash ./preload.sh --days 90 --ticker ndq.ax --ticker btc-aud
```

Preload works through the tickers one at a time and prints progress in the terminal. Wait for the final summary. If one ticker fails, it continues with the others and tells you which failed. It does not count an old saved forecast as a successful refresh. The command returns a non-zero exit status if anything fails.

Only one preload batch can run at a time. Run preload from the terminal; the `/preload` web address returns HTTP 410 and does not start a refresh.

### What does `--retune` change?

A **forecast** is the result: predicted prices, return estimates, charts and supporting information. **Model settings** control how Prophet fits the price history.

A normal preload downloads data and fits the model again using suitable saved settings. It is not just loading a file, and it can still take a while. It skips the much slower search for new settings.

With `--retune`, the app tests 200 combinations of `changepoint_prior_scale` and `seasonality_prior_scale` against historical data. Every time a combination has a lower validation error than the best one tested so far in that run, it immediately writes both values to the ticker's entry in the JSON file. It does not wait for the ticker or the whole batch to finish. Equal or worse scores do not change the file.

The first combination with a valid score starts the record for each run. "Better" means better than the other combinations tested in that run, not necessarily better than settings from a different run with different data. Retuning prints a message whenever it saves a new best pair.

| Command | What it does | When to use it |
| --- | --- | --- |
| `bash ./preload.sh` | Refreshes forecasts using saved settings, or Prophet defaults if none are valid. | Routine updates. |
| `bash ./preload.sh --retune` | Searches for new settings, then saves updated forecasts. | When you want to try new settings and have time for a longer run. |

Both examples cover all configured and remembered tickers for 365 days. For your first retuning run, start with one ticker and the period you use:

```sh
bash ./preload.sh --days 90 --ticker ndq.ax --retune
```

Once the forecast succeeds, the final settings and complete forecast are also saved in the cache for that ticker and requested period. Retuning is much slower and does not guarantee more accurate future predictions. You do not need to retune before every search or refresh.

**Using PowerShell instead?** Replace `bash ./preload.sh` with `docker exec stock_predictor flask --app app preload` and keep the same options. The examples above use Bash throughout.

### How model settings are saved

**Both settings are used by Prophet.** The JSON file supplies initial values and records the best pair found so far during retuning, like this:

```json
{"ticker": "acdc.ax", "changepoint_prior_scale": 0.08, "seasonality_prior_scale": 1.0}
```

This is the same ticker-list format shown earlier, with two optional settings added. **Only `ticker` is required.** Keep the numbers if you already have them; for a new ticker, you can leave them out and let the app use saved settings or defaults.

The optional numbers mean:

| Field | Meaning |
| --- | --- |
| `changepoint_prior_scale` | How freely the model can change its trend. Higher values allow more changes; lower values favour a smoother trend. |
| `seasonality_prior_scale` | How strongly the model can fit repeating weekly or yearly patterns. Higher values allow stronger patterns. |

These numbers are model controls, not predicted prices, percentages or accuracy scores. Higher is not automatically better. For a new ticker, a ticker-only entry is enough. The app uses Prophet defaults (`0.05` for trend changes and `10.0` for seasonal patterns) when no suitable saved settings exist.

The app imports initial settings from this file into its cache once. An entry needs `changepoint_prior_scale` to be included in that import. If `seasonality_prior_scale` is missing, it uses `10.0`. Ticker-only entries still belong to the preload list; they simply have no initial settings to import.

**After that one-time import, editing the JSON numbers does not change the cached settings.** Changes to the ticker list are still read by preload. A normal refresh does not write to the JSON file; `--retune` updates it as better combinations are found.

Retuning keeps the other tickers and extra fields in the file. If the ticker is missing, it adds an entry when it finds the first valid combination. Each update replaces the file in one step, so the app does not leave half-written JSON. Avoid editing the file by hand while retuning is running.

If retuning stops or fails, the JSON file keeps the best pair saved so far. That does not mean the full search or forecast finished. The previously completed forecast and settings in the cache remain unchanged until a new forecast succeeds. Running `--retune` again starts a new search; it does not resume where it stopped.

The JSON file holds one pair per ticker, so retuning another period replaces that ticker's pair. The cache in `tmp/forecasts-v2/` keeps completed settings and forecasts separately for each requested period, along with the remembered ticker list.

### How long are forecasts and settings kept?

- **Forecasts:** normally reused for up to 12 hours for the same ticker and requested period. They may expire sooner if their settings expire. Preload forces a refresh even before that time is up.
- **Tuned settings:** valid for 30 days after they are saved. Using them for another refresh does not restart the clock.
- **Settings imported from JSON:** valid for 30 days after import. The file does not record when those values were selected.

When choosing settings, the app first looks for valid tuned settings for that ticker and period, then valid settings imported from JSON, then Prophet defaults. Expiry does not automatically trigger retuning; only `--retune` does that.

The results page labels the source as `tuned` (selected by retuning), `legacy` (imported from JSON), or `default` (Prophet defaults).

If an ordinary website search cannot refresh a forecast, it may show the last successful result with a warning. These fallback results are kept for up to seven days, but storage cleanup can remove them sooner. Preload reports a failed refresh instead of silently using that fallback.

A recently generated forecast does not necessarily contain today's market prices. Check the **Data through** date on the results page; Yahoo's data can be delayed or incomplete.

Keep the project's `tmp/` folder if you want to retain saved forecasts and settings. The startup commands and script above mount the project folder, so those files stay on your computer when you replace the container.
