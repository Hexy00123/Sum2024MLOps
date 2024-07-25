# src/validate.py
from copy import deepcopy
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score
import giskard
from giskard import Dataset, Model, Suite, testing
import mlflow
import zenml
from data import read_datastore, preprocess_data
from model import retrieve_model_with_alias, retrieve_model_with_version
from hydra import compose, initialize
import mlflow.sklearn
import pandas as pd
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
print(datetime.now())

# ------------------------------
# 1. Initialize Configuration
with initialize(config_path="../configs"):
    cfg = compose(config_name="main")

print("DBG: Wrapping dataset")
version = cfg.test_data_version

# Read datastore
df = read_datastore()
print('DBG: read datastore')
if cfg.data.target_cols[0] not in df.columns:
    raise ValueError(f"The '{cfg.data.target_cols[0]}' column is missing from the DataFrame.")

dataset_name = cfg.dataset.name

# Wrap your Pandas DataFrame with giskard.Dataset (validation or test set)
giskard_dataset = Dataset(
    df=df,
    target=cfg.data.target_cols[0],
    name=dataset_name,
)

print("DBG: Wrapping model")
# print("DBG dataframe", df.columns)

# ------------------------------
# 2. Retrieve Model by finding the model with highest r2 score
model_name = cfg.model.best_model_name
model_tag_key = cfg.model.model_tag_key
model_tag_value = cfg.model.model_tag_value
model_alias = cfg.model.best_model_alias
model_version = cfg.model.best_model_version

# Initialize MLflow Client
client = mlflow.MlflowClient()

mlflow.set_tracking_uri("http://localhost:5000")


with initialize(config_path="../configs", version_base=None):
    cfg = compose(config_name="main")

model = mlflow.pyfunc.load_model(os.path.join("api", "model_dir"))

print(model)



# df = mlflow.search_runs(filter_string="metrics.r2 < 1")
# print(df)

# run_id = df.loc[df['metrics.rmse'].idxmax()]['run_id']
# print(run_id)

# model = mlflow.sklearn.load_model("runs:/" + run_id + "/model")

# reatrieve a 

# Search for the model version that matches the tag {'source': 'champion'}
model_version = str(cfg.model.best_model_version)
# for mv in client.search_model_versions(f"name='{model_name}'"):
#     # Check if the specific tag key-value pair exists
#     if model_tag_key in mv.tags and mv.tags[model_tag_key] == model_tag_value:
#         model_version = mv.version
#         break

# if model_version is None:
#     raise ValueError(f"No model found with name '{model_name}' and tag '{model_tag_key}': '{model_tag_value}'")

# # Assign alias 'champion' to the model version found
# client.set_registered_model_alias(name=model_name, version=model_version, alias=model_alias)

# print("DBG: Model name: ", model_name)
# print("DBG: Model tag: ", model_tag_key, '=', model_tag_value)
# print("DBG: Model version: ", model_version)

# model: mlflow.pyfunc.PyFuncModel = retrieve_model_with_alias(model_name, model_alias = model_alias)  
# model: mlflow.pyfunc.PyFuncModel = retrieve_model_with_version(model_name, model_version = model_version)
# print("Model loaded successfully")

# mv = client.get_model_version(name = model_name, version=model_version)
# mv = client.get_model_version_by_alias(name = model_name, alias=model_alias)
model_version = mv.version
# print("Model input schema: ", model.metadata.get_input_schema())

schema = model.metadata.get_input_schema()
# print(type(schema))
columns = []
for col in schema:
    column = col.name
    columns.append(column)
# print(type(columns[0]))
# print(columns)

# ------------------------------
# 3. Initial Prediction

def predict(raw_df):
    column_name = cfg.data.target_cols[0]
    scaler = zenml.load_artifact(f"{column_name}_scaler")

    # print("DBG preprocess: call predict")
    data = preprocess_data(df=deepcopy(raw_df), X_only=True)
    predictions = model.predict(data)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1)

    # print("DBG return:", predictions)
    return predictions

predictions = predict(df)
print(f"DBG: Predictions: {len(predictions)}\n{predictions}")

X, y = df.drop(cfg.data.target_cols[0], axis=1), df[cfg.data.target_cols[0]]

# ------------------------------
# 4. Create Giskard Model
print("DBG: Create giskard model")

giskard_model = Model(
    model=predict,
    model_type="regression",
    feature_names=X.columns,
    name=model_name,
)

# print(giskard_model.feature_names)
# print(giskard_dataset.df.columns)

# ------------------------------
# 5. Scanning Model
print("DBG: Scanning model")

wrapped_predict = giskard_model.predict(giskard_dataset)
wrapped_test_metric = r2_score(y, wrapped_predict.prediction)

print(f'DBG: Wrapped Test R2-score: {wrapped_test_metric:.2f}')

# scan_results = giskard.scan(giskard_model, giskard_dataset)
# scan_results_path = f"reports/validation_results_{model_name}_{model_version}_{dataset_name}_{version}.html"
# scan_results.to_html(scan_results_path)

# 6. Create a Test Suite
suite_name = f"test_suite_{model_name}_{model_version}_{dataset_name}_{version}"
test_suite = Suite(name=suite_name)

test1 = testing.test_r2(
    model=giskard_model,
    dataset=giskard_dataset,
    threshold=0.4
)

test_suite.add_test(test1)

test_results = test_suite.run()
if test_results.passed:
    print("Passed model validation!")
else:
    print("Model has vulnerabilities!")
