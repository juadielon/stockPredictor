#!/bin/bash
app="stock_predictor"
docker build -t ${app} .

# Stop and remove previous version
docker stop ${app}
docker rm ${app}

docker run -d -p 80:80 --name=${app} -v ${PWD}:/app ${app}

# Remove:
# - all stopped containers
# - all networks not used by at least one container
# - all dangling images
# - all dangling build cache
docker system prune -f
