#!/bin/bash
# Run test suite inside the docker container
docker exec -it stock_predictor pytest tests/ -v
