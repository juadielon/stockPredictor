#! /usr/bin/env bash
curl -s 'http://localhost/preload' &
docker logs -f stock_predictor 2>&1 | grep -v INFO
