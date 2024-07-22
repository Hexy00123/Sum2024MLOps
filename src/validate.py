# src/validate.py
import pandas as pd

from data import read_datastore, preprocess_data, read_features
from model import retrieve_model_with_alias, retrieve_model_with_version
import giskard
import hydra
from hydra import compose, initialize
import mlflow
mlflow.set_tracking_uri(uri="http://localhost:5000")


# ------------------------------
# 1. Wrap raw dataset
with initialize(config_path="../configs"):
    cfg = compose(config_name="main")


print("DBG:", cfg)
version = cfg.test_data_version

df = read_datastore()
print('DBG: read datastore')

# Specify categorical columns and target column
TARGET_COLUMN = cfg.data.target_cols[0]
# CATEGORICAL_COLUMNS = list(cfg.data.cat_cols)

dataset_name = cfg.dataset.name

# Wrap your Pandas DataFrame with giskard.Dataset (validation or test set)
giskard_dataset = giskard.Dataset(
    # A pandas.DataFrame containing raw data (before pre-processing) and including ground truth variable.
    df=df,
    target=TARGET_COLUMN,  # Ground truth variable
    name=dataset_name,  # Optional: Give a name to your dataset
    # cat_columns=CATEGORICAL_COLUMNS  # List of categorical columns. Optional, but improves quality of results if available.
)

# ------------------------------
# 2. Wrap model
model_name = cfg.model.best_model_name

# You can sweep over challenger aliases using Hydra
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

# ------------------------------
# 3. Initial prediction

df_50 = df.sample(frac=0.01, random_state=42)


def predict(raw_df):
    raw_df = raw_df.drop(columns=cfg.data.target_cols)
    X = preprocess_data(df=raw_df, X_only=True)
    X = pd.DataFrame(X)
    return model.predict(X)


predictions = predict(df_50)
print(f"DBG: Predictions: {len(predictions)}\n{predictions}")


# 3. Validation with raw dataset

# X, y = preprocess_data(
#     df=df_50,
#     # df=df.head(),
#     only_X=True
# )
# X = pd.DataFrame(X)
# predictions = model.predict(X)
# print(predictions)


# Validation with ZenML dataset
# train_data_version = str(cfg.train_data_version)
# test_data_version = str(cfg.test_data_version)
# #
# X_test, y_test = read_features(name="features_target",
#                                    version=test_data_version)
#
# predictions = model.predict(X_test)
# print(predictions)
# print(y_test)


# ------------------------------
# 4. Create Giskard model
# giskard_model = giskard.Model(
#   model=predict,
#   model_type = "classification", # regression
#   classification_labels=list(cfg.data.labels),  # The order MUST be identical to the prediction_function's output order
#   feature_names = df.columns, # By default all columns of the passed dataframe
#   name=model_name, # Optional: give it a name to identify it in metadata
#   # classification_threshold=0.5, # Optional: Default: 0.5
# )
