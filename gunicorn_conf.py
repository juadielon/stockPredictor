import multiprocessing

# Bind to localhost port 5000
bind = "0.0.0.0:5000"

# Worker Options
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'

# Timeout
timeout = 900  # Matching the previous uwsgi_read_timeout 900s

# Logging configuration
accesslog = '-'  # Log to stdout
errorlog = '-'   # Log to stderr
loglevel = 'info'

# Reload on code changes (Development mode)
reload = True
