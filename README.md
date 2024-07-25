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
### Set venv:
```sh
export PROJECT_DIR=$PWD
export PYTHONPATH=$PROJECT_DIR/src
export ZENML_CONFIG_PATH=$PWD/services/zenml
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
