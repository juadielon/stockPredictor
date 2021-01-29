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
