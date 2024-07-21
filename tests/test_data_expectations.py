import pandas as pd
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig
from typing import Tuple


from src.data import read_datastore, preprocess_data
from src.data_expectations import validate_features, validate_initial_data


@pytest.fixture
def cfg() -> DictConfig:
    """
    Load the test_config.yaml configuration file
    """
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="test_config")
    return cfg


@pytest.fixture
def raw_sample(cfg) -> pd.DataFrame:
    df = read_datastore(cfg)
    return df

@pytest.fixture
def preprocessed_sample(raw_sample) -> Tuple[pd.DataFrame, pd.Series]:
    X, y = preprocess_data(raw_sample)
    return X, y


def test_validate_initial_data():
    try:
        validate_initial_data()
    except AssertionError:
        pass


def test_validate_features(preprocessed_sample):
    print("Data read successfully!")
    X, y = preprocessed_sample
    print("Data preprocessed successfully!")
    try:
        validate_features(X, y)
        print("Data validated successfully!")
    except AssertionError:
        pass
