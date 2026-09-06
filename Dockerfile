FROM python:3.13-slim

# Prevent writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    nginx \
    supervisor \
    vim \
    apt-utils \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --upgrade pip \
    && pip install flask-wtf yfinance \
    && pip install prophet \
    && pip install diskcache plotly \
    && pip install gunicorn \
    && pip install pytest pytest-mock

# Set up Nginx
RUN rm /etc/nginx/sites-enabled/default
COPY nginx.conf /etc/nginx/nginx.conf

# Set up Supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set up Gunicorn config
COPY gunicorn_conf.py /app/gunicorn_conf.py

# Copy application code
WORKDIR /app
COPY . /app

# Run unit tests before finalising the image
RUN pytest tests/

# Set environment variables for static files (used by the app logic if needed, though Nginx handles serving now)
ENV STATIC_URL=/static
ENV STATIC_PATH=/app/static

# Set the timezone
RUN ln -fs /usr/share/zoneinfo/Australia/Brisbane /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata

# Create directory for supervisor logs if not exists
RUN mkdir -p /var/log/supervisor

# Expose port 80
EXPOSE 80

# Make scripts executable
RUN chmod +x /app/*.sh /app/entrypoint.sh

# Run entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]