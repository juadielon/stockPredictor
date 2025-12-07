#!/bin/bash
set -e

# specific cleanup for logs
rm -f /var/log/nginx/* /var/log/supervisor/*

# Symlink nginx.conf from the mounted volume to apply changes on restart
if [ -f /app/nginx.conf ]; then
   ln -sf /app/nginx.conf /etc/nginx/nginx.conf
fi



# Start Supervisor
exec /usr/bin/supervisord
