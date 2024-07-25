from hydra import compose, initialize

import os
import sys
from typing_extensions import Tuple, Annotated

import pandas as pd
from zenml import step, pipeline, ArtifactConfig

BASE_PATH = os.getenv("PROJECT_DIR")
print(BASE_PATH)
sys.path.append(BASE_PATH)


from data import read_datastore, preprocess_data, load_features
from data_expectations import validate_features
# from src.data_expectations import validate_features
# from src.data import preprocess_data, read_datastore, load_features


@step(enable_cache=False)
def extract() -> Tuple[
    Annotated[pd.DataFrame,
              ArtifactConfig(name="extracted_data",
                             tags=["data_preparation"]
                             )
              ],
    Annotated[int,
              ArtifactConfig(name="data_version",
                             tags=["data_preparation"])]
]:
    print("Extracting version...")
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="main")
        version = cfg.index
        print(f'Version is: {version}, type: {type(version)}')

    print("Extracting data...")
    df = read_datastore()
    print(df)

    return df, version


@step(enable_cache=False)
def transform(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,
              ArtifactConfig(name="input_features",
                             tags=["data_preparation"])],
    Annotated[pd.Series,
              ArtifactConfig(name="input_target",
                             tags=["data_preparation"])]
]:
    print("Preprocessing data...")
    X, y = preprocess_data(df)
    print(X.info())
    print(X.head())
    print(y.head())
    return X, y


@step(enable_cache=False)
def validate(X: pd.DataFrame, y: pd.Series) -> Tuple[
    Annotated[pd.DataFrame,
              ArtifactConfig(name="valid_input_features",
                             tags=["data_preparation"])],
    Annotated[pd.Series,
              ArtifactConfig(name="valid_target",
                             tags=["data_preparation"])]
]:
    print("Validating features...")
    validate_features(X, y)
    return X, y


@step(enable_cache=False)
def load(X: pd.DataFrame, y: pd.Series, version: int) -> Tuple[
    Annotated[pd.DataFrame,
              ArtifactConfig(name="features",
                             tags=["data_preparation"])],
    Annotated[pd.Series,
              ArtifactConfig(name="target",
                             tags=["data_preparation"])]
]:
    print("Loading features...")
    load_features(X, y, version)
    return X, y


@pipeline()
def prepare_data_pipeline():
    df, version = extract()
    X, y = transform(df)
    X, y = validate(X, y)
    X, y = load(X, y, version)


if __name__ == "__main__":
    run = prepare_data_pipeline()
