FROM tiangolo/uwsgi-nginx-flask:python3.8-alpine
RUN apk --update add bash vim
ENV STATIC_URL /static
ENV STATIC_PATH /app/static
RUN pip install --upgrade pip \
    && pip install Flask==1.1.2
