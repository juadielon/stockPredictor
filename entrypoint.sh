#!/bin/bash
set -e

# Run prestart script if it exists
if [ -f /app/prestart.sh ]; then
    echo "Running prestart.sh..."
    . /app/prestart.sh
else
    echo "No prestart.sh found."
fi

# Start Supervisor
exec /usr/bin/supervisord
