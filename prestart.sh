#! /usr/bin/env bash
# Remove previously created symlinks to stdoutput, enabling nginx logs in files
rm -f /var/log/nginx/* /var/log/supervisor/*

# Symlink nginx.conf from the mounted volume to apply changes on restart
ln -sf /app/nginx.conf /etc/nginx/nginx.conf
