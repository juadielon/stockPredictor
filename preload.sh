#! /usr/bin/env bash
curl -s 'http://localhost/preload' &
docker exec -it stock_predictor tail -f /var/log/uwsgi/uwsgi.log
