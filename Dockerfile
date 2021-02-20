FROM tiangolo/uwsgi-nginx-flask:python3.8

#RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \ 
    && apt-get install -y \
    apt-utils

RUN apt-get upgrade -y \
    && apt-get install -y \
    vim

# Clean after apt-get update and installs
RUN rm -rf /var/lib/apt/lists/* \
    && rm -rf /src/*.deb

ENV STATIC_URL /static
ENV STATIC_PATH /app/static

# Set the timezone
RUN ln -fs /usr/share/zoneinfo/Australia/Brisbane /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata

RUN echo "uwsgi_read_timeout 900s;" > /etc/nginx/conf.d/uwsgi_timeout.conf
RUN mkdir /var/log/uwsgi
RUN pip install --upgrade pip \
    && pip install flask-wtf yfinance \
    && pip install pystan convertdate lunarcalendar holidays tqdm \
    && pip install fbprophet \
    && pip install diskcache