# src/validate.py
from copy import deepcopy

from giskard import demo, test, Dataset, TestResult, testing
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import pandas as pd
from datetime import datetime

from data import read_datastore, preprocess_data, read_features
from model import retrieve_model_with_version
import giskard
from hydra import compose, initialize
import mlflow
import zenml

mlflow.set_tracking_uri("http://localhost:5000")
print(datetime.now())


# ------------------------------
# 1. Wrap raw dataset
with initialize(config_path="../configs"):
    cfg = compose(config_name="main")


print("DBG: Wrapping dataset")
version = cfg.test_data_version

df = read_datastore()
print('DBG: read datastore')
if cfg.data.target_cols[0] not in df.columns:
    raise ValueError("The 'price' column is missing from the DataFrame.")

dataset_name = cfg.dataset.name

# Wrap your Pandas DataFrame with giskard.Dataset (validation or test set)
giskard_dataset = giskard.Dataset(
    # A pandas.DataFrame containing raw data (before pre-processing) and including ground truth variable.
    df=df,
    target=cfg.data.target_cols[0],  # Ground truth variable
    name=dataset_name,  # Optional: Give a name to your dataset
    # cat_columns=CATEGORICAL_COLUMNS  # List of categorical columns. Optional, but improves quality of results if available.
)

print("DBG: Wrapping model")
print("DBG dataframe", df.columns)
# ------------------------------
# 2. Wrap model
model_name = cfg.model.best_model_name
model_alias = cfg.model.best_model_alias
model_version = cfg.model.best_model_version

print("DBG: Model name: ", model_name)
# model: mlflow.pyfunc.PyFuncModel = retrieve_model_with_alias(model_name, model_alias = model_alias)
model: mlflow.pyfunc.PyFuncModel = retrieve_model_with_version(
    model_name, model_version)
print("DBG: Model loaded successfully")

client = mlflow.MlflowClient()

# mv = client.get_model_version_by_alias(name = model_name, alias=model_alias)
mv = client.get_model_version(name=model_name, version=str(model_version))

model_version = mv.version

print("DBG: Model input schema: ", model.metadata.get_input_schema())
schema = model.metadata.get_input_schema()
print(type(schema))
columns = []
for col in schema:
    column = col.name
    columns.append(column)
print(type(columns[0]))
print(columns)

# data = preprocess_data(df[df.columns].head(), X_only=True)
# data
#
# data2 = preprocess_data(df[df.columns].head(), X_only=True)
# data2
# ------------------------------
# 3. Initial prediction

print("DBG dataframe", df.columns)


def predict(raw_df):
    column_name = cfg.data.target_cols[0]
    scaler = zenml.load_artifact(f"{column_name}_scaler")

    # TODO: price
    print("DBG preprocess: call predict")
    data = preprocess_data(df=deepcopy(raw_df), X_only=True)
    predictions = model.predict(data)
    predictions = scaler.inverse_transform(
        predictions.reshape(-1, 1)).reshape(-1)

    print("DBG return:", predictions)
    # X = data.drop('price', axis=1)
    return predictions


print("DBG dataframe", df.columns)

predictions = predict(df)
print(f"DBG: Predictions: {len(predictions)}\n{predictions}")

X, y = df.drop(cfg.data.target_cols[0], axis=1), df[cfg.data.target_cols[0]]

print("DBG dataframe", df.columns)

# ------------------------------
print("DBG: Create giskard model")
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
print("DBG: Scanning model")

wrapped_predict = giskard_model.predict(giskard_dataset)
wrapped_test_metric = r2_score(y, wrapped_predict.prediction)

print(f'DBG: Wrapped Test R2-score: {wrapped_test_metric:.2f}')

scan_results = giskard.scan(giskard_model, giskard_dataset)
scan_results_path = f"reports/validation_results_{model_name}_{model_version}_{dataset_name}_{version}.html"
scan_results.to_html(scan_results_path)

# 4. Create a test suite
suite_name = f"test_suite_{model_name}_{model_version}_{dataset_name}_{version}"
test_suite = giskard.Suite(name=suite_name)

test1 = giskard.testing.test_r2(model=giskard_model,
                                  dataset=giskard_dataset,
                                  threshold=0.4)

test_suite.add_test(test1)

test_results = test_suite.run()
if (test_results.passed):
    print("Passed model validation!")
else:
    print("Model has vulnerabilities!")
