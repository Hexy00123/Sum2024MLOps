import os

import giskard
from datetime import datetime
import mlflow
# src/validate.py
import pandas as pd
from hydra import compose, initialize
from data import read_datastore, preprocess_data
from model import retrieve_model_with_version

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.get_tracking_uri()
print(datetime.now())


# ------------------------------
# 1. Wrap raw dataset
with initialize(config_path="../configs"):
    cfg = compose(config_name="giskard")



print("Wrapping dataset")
version = cfg.test_data_version

df = read_datastore(cfg)
print(df.columns)
if 'price' not in df.columns:
    raise ValueError("The 'price' column is missing from the DataFrame.")
# Specify categorical columns and target column
TARGET_COLUMN = cfg.data.target_cols[0]

# CATEGORICAL_COLUMNS = list(cfg.data.cat_cols)

dataset_name = cfg.dataset.name

giskard_dataset = giskard.Dataset(
    df=df,  # A pandas.DataFrame containing raw data (before pre-processing) and including ground truth variable.
    target="price",  # Ground truth variable
    name=dataset_name,
)


print("Wrapping model")
# ------------------------------
# 2. Wrap model
model_name = cfg.model.best_model_name

model_alias = cfg.model.best_model_alias

model_version = cfg.model.best_model_version

print("Model name: ", model_name)
# model: mlflow.pyfunc.PyFuncModel = retrieve_model_with_alias(model_name, model_alias = model_alias)  
model: mlflow.pyfunc.PyFuncModel = retrieve_model_with_version(model_name, model_version)
print("Model loaded successfully")

client = mlflow.MlflowClient()

# mv = client.get_model_version_by_alias(name = model_name, alias=model_alias)
mv = client.get_model_version(name=model_name, version=str(model_version))

model_version = mv.version


print("Model input schema: ", model.metadata.get_input_schema())


schema = model.metadata.get_input_schema()
print(type(schema))
columns = []
for col in schema:
    column = col.name
    columns.append(column)
print(type(columns[0]))
print(columns)


# def transform_data(df):
#     print("Original DataFrame columns: ", df.columns)

#     X = preprocess_data(
#         df=df,
#         X_only=True,
#     )    
#     print("DataFrame after preprocessing: ", X.columns)

#     input_schema = columns
#     # print("Missing from input schema:")
#     for col in X.columns:
#         if col not in input_schema:
#             X.drop(col, axis=1, inplace=True)
            
#     # now fill the missing columns with 0
#     # print("Missing from input data:")
#     for col in input_schema:
#         if col not in X.columns:
#             X[col] = float(0)
#     # transform all to float
#     X = X.astype(float)
#     X = pd.DataFrame(X)
#     return X


data = preprocess_data(df[df.columns].head(), X_only=True)
data

data2 = preprocess_data(df[df.columns].head(), X_only=True)
data2




# ------------------------------
# Initial prediction
print("Initial prediction")
def predict(raw_df):
    print(raw_df)
    data = preprocess_data(df=raw_df, X_only=True)
    X = data.drop('price', axis=1)
    return model.predict(X)


predictions = predict(df[df.columns].head())
print(f"Predictions: {predictions}")



X, y = df.drop('price', axis=1), df['price']


# ------------------------------

print("Create giskard model")
# 4. Create Giskard model
giskard_model = giskard.Model(
    model=predict,
    model_type="regression",
    feature_names=X.columns,
    name=model_name,
)



print(giskard_model.feature_names)
print(giskard_dataset.df.columns)



# ------------------------------
# 3. Scanning model
print("Scanning model")
from sklearn.metrics import r2_score


wrapped_predict = giskard_model.predict(giskard_dataset)
wrapped_test_metric = r2_score(y, wrapped_predict.prediction)

print(f'Wrapped Test R2-score: {wrapped_test_metric:.2f}')


scan_results = giskard.scan(giskard_model, giskard_dataset)
scan_results_path = f"reports/validation_results_{model_name}_{model_version}_{dataset_name}_{version}.html"
scan_results.to_html(scan_results_path)


# 4. Create a test suite
suite_name = f"test_suite_{model_name}_{model_version}_{dataset_name}_{version}"
test_suite = giskard.Suite(name = suite_name)

# TODO: create a custom giskard function with MAPE

from sklearn.metrics import mean_absolute_percentage_error
from giskard import demo, test, Dataset, TestResult, testing


test1 = giskard.testing.test_rmse(model=giskard_model,
                                 dataset=giskard_dataset,
                                 threshold=1.0)

test_suite.add_test(test1)

test_results = test_suite.run()
if (test_results.passed):
  print("Passed model validation!")
else:
  print("Model has vulnerabilities!")