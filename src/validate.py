# src/validate.py
import pandas as pd

from data import read_datastore, preprocess_data
from model import retrieve_model_with_alias, retrieve_model_with_version
import giskard
import hydra
from hydra import compose, initialize
import mlflow

# 1. Wrap raw dataset
with initialize(config_path="../configs"):
    cfg = compose(config_name="giskard")

version = cfg.test_data_version

df = read_datastore()

# Specify categorical columns and target column
TARGET_COLUMN = cfg.data.target_cols[0]

# CATEGORICAL_COLUMNS = list(cfg.data.cat_cols)

dataset_name = cfg.dataset.name

# Wrap your Pandas DataFrame with giskard.Dataset (validation or test set)
giskard_dataset = giskard.Dataset(
    df=df,  # A pandas.DataFrame containing raw data (before pre-processing) and including ground truth variable.
    target=TARGET_COLUMN,  # Ground truth variable
    name=dataset_name,  # Optional: Give a name to your dataset
    # cat_columns=CATEGORICAL_COLUMNS  # List of categorical columns. Optional, but improves quality of results if available.
)

# 2. Wrap model
model_name = cfg.model.best_model_name

# You can sweep over challenger aliases using Hydra
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


# custom predict function
# transformer_version = cfg.data_transformer_version


def predict(raw_df):
    X, y = preprocess_data(
        df=raw_df
    )
    X = pd.DataFrame(X)

    return model.predict(X)


# print(type(df))
# print(type(df[df.columns].head()))
# print(predict(df))


# predictions = predict(df[df.columns].head())
# predictions = predict(df.head())
# print(predictions)

# from data import read_features
# train_data_version = str(cfg.train_data_version)
# test_data_version = str(cfg.test_data_version)
#
# X_test, y_test = read_features(name="features_target",
#                                    version=test_data_version)
#
# print(X_test.shape, y_test.shape)
# print(X_test.head())
# X_zenml = X_test.head()

# take half of the data df
df_50 = df.sample(frac=0.5, random_state=88)

X, y = preprocess_data(
    # df=df,
    df=df.head(),
    only_X=True
)
#
#
X = pd.DataFrame(X)
print(type(X))
#
print(X)
#
predictions = model.predict(X)
print(predictions)
