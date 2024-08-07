[![Test code](https://github.com/Hexy00123/Sum2024MLOps/actions/workflows/test-code.yaml/badge.svg?branch=dev&event=push)](https://github.com/Hexy00123/Sum2024MLOps/actions/workflows/test-code.yaml)
[![Validate model](https://github.com/Hexy00123/Sum2024MLOps/actions/workflows/validate-model.yaml/badge.svg)](https://github.com/Hexy00123/Sum2024MLOps/actions/workflows/validate-model.yaml)

# MLOps Sum 2024 project repository

## **Team Members**

| Team Member              | Telegram ID         |
|--------------------------|---------------------|
| Alexandra Vabnits        | @sashhhaka          |   
| Ruslan Izmailov          | @Nooth1ng           | 
| Bulat Akhmatov           | @bulatik1337        | 

## Running scripts
All scripts should be run from the repository root, to ensure correct paths initialization in scripts.

Initial whole dataset zameen-updated.csv should be stored in folder "data".

[Zameen.com House Price Prediction Dataset](https://www.kaggle.com/datasets/howisusmanali/house-price-prediction-zameencom-dataset)

## Running documentation
### Install requirements:
```sh
pip install -r requirements.txt
```
### Set env:
```sh
export PROJECT_DIR=$PWD
export PYTHONPATH=$PROJECT_DIR/src
export ZENML_CONFIG_PATH=$PWD/services/zenml
export MLFLOW_TRACKING_URI="http://localhost:5000"
```
### Download dataset from Kaggle:
```sh
python src/download_kaggle_dataset
```
### How to start ZenML pipeline:
```sh
python pipelines/data_prepare.py
```
### How to start Airflow pipeline:
```sh
airflow standalone
```
### MLFlow experiments:
```sh
mlflow server
mlflow run . --env-manager=local
```
### Giskard validation:
```sh
python src/validate.py
```
### Run tests:
```sh
pytest tests
```
## Deploy
### Deploy Docker container:
```sh
cd api

docker build -t my_ml_service .
docker run -d -p 5123:8080 my_ml_service

docker tag my_ml_service sashhhak0/my_ml_service
docker tag my_ml_service sashhhak0/my_ml_service:v1.0
docker push sashhhak0/my_ml_service:latest
```
### Deploy website:
Start mlflow server:
```
mlflow ui --port 5000
```
Deploy Flask API:
```
python3 api/app.py
```
Deploy Gradio UI:
```
python3 src/app.py
```


