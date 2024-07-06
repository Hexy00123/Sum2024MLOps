# The base image - Use the same version of the package `apache-airflow` that you installed via pip
FROM apache/airflow:2.7.3-python3.11
#FROM apache/airflow:latest-python3.11
# FROM apache/airflow:2.9.2-python3.11
# Why python3.11? the explanation is later below

# Set CWD inside the container
WORKDIR /project
# This will be the project folder inside the container

# Copy requirements file
COPY airflow.requirements.txt .

RUN pip install --upgrade pip

# Install requirements.txt
RUN pip install -r airflow.requirements.txt --upgrade

# Switch to root user
USER root

# Install some more CLIs
RUN apt-get update \
&& apt-get install -y --no-install-recommends vim curl git rsync unzip \
&& apt-get autoremove -y \
&& apt-get clean

EXPOSE 8080

# Switch to regular user airflow
USER airflow

# Run this command when we start the container
CMD ["airflow", "standalone"]