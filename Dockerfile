FROM tiangolo/uwsgi-nginx-flask:python3.8
RUN apt update \
    && apt upgrade -y \
    && apt install vim -y
ENV STATIC_URL /static
ENV STATIC_PATH /app/static
RUN mkdir /var/log/uwsgi
RUN pip install --upgrade pip \
    && pip install flask-wtf yfinance \
    && pip install pystan convertdate lunarcalendar holidays tqdm \
    && pip install fbprophet