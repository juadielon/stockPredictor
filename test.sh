#!/usr/bin/env bash
set -eu
exec docker exec stock_predictor pytest tests/ -v "$@"
