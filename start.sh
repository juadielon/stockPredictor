#!/usr/bin/env bash
# Usage: bash ./start.sh [--prune]
# Use HOST_PORT=8083 bash ./start.sh to choose another host port (default: 80).
# Builds the image before replacing stock_predictor; saved project files stay mounted.
# --prune deletes ALL stopped containers after startup, including other projects
# and data in their writable layers. Volumes, images and networks are not pruned.
# Without --prune, only the existing stock_predictor container is removed.
set -euo pipefail

prune_containers=false
for argument in "$@"; do
	case "$argument" in
		--prune) prune_containers=true ;;
		-h|--help)
			printf '%s\n' \
				'Usage: bash ./start.sh [--prune]' \
				'Build and recreate stock_predictor. HOST_PORT sets the host port (default: 80).' \
				'Example: HOST_PORT=8083 bash ./start.sh --prune' \
				'--prune: after startup, delete ALL stopped containers, including other projects and their writable data.' \
				'Volumes, images and networks are not pruned. No automatic rollback is provided.'
			exit 0
			;;
		*)
			printf 'Unknown option: %s. Use --help for usage.\n' "$argument" >&2
			exit 2
			;;
	esac
done

app="stock_predictor"
host_port="${HOST_PORT-80}"
if [[ ! "$host_port" =~ ^[0-9]{1,5}$ ]]; then
	printf 'HOST_PORT must be an integer from 1 to 65535.\n' >&2
	exit 1
fi

host_port=$((10#$host_port))
if ((host_port < 1 || host_port > 65535)); then
	printf 'HOST_PORT must be an integer from 1 to 65535.\n' >&2
	exit 1
fi

project_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
mount_directory="$project_directory"
case "${OSTYPE:-}" in
	msys*)
		export MSYS_NO_PATHCONV=1
		mount_directory="$(cd -- "$project_directory" && pwd -W)"
		;;
esac

docker info >/dev/null
docker build -t "$app" "$mount_directory"

container_id="$(docker container ls -aq --filter "name=^/${app}$")"
if [[ -n "$container_id" ]]; then
	running="$(docker inspect --format '{{.State.Running}}' "$container_id")"
	if [[ "$running" == "true" ]]; then
		docker stop "$container_id"
	fi
	docker rm "$container_id"
fi

if docker run -d --name "$app" -p "${host_port}:80" \
	--mount "type=bind,source=${mount_directory},target=/app" "$app"; then
	printf 'Container started. Open http://localhost:%s once the app is ready.\n' "$host_port"
else
	printf 'Could not start the app. Any previous container has already been removed; no automatic rollback was performed.\n' >&2
	exit 1
fi

if [[ "$prune_containers" == true ]]; then
	printf 'Removing all stopped containers, including those from other projects.\n'
	docker container prune -f
fi
