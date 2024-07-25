#!/bin/bash

# docker login before running this script

cd api

docker build -t my_ml_service .
docker run -d -p 5123:8080 my_ml_service

docker tag my_ml_service sashhhak0/my_ml_service
docker tag my_ml_service sashhhak0/my_ml_service:v1.0
docker push sashhhak0/my_ml_service:latest