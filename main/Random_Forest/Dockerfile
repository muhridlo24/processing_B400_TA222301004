#Dockerfile, Image, Container

FROM python:3.11

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app