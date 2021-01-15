# Stock Predictor

Time series analysis to predict future stock prices.

## How to run

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
restart.sh
```
