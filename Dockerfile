FROM python:3.9.13
COPY  . /app
WORKDIR  /app
RUN  pio install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app

