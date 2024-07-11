FROM apache/airflow:latest


USER root
RUN apt-get update && \
    apt-get  -y install git && \
    apt-get clean

USER airflow

WORKDIR project

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY ./ ./
