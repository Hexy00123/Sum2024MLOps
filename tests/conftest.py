from typing import Tuple

import pandas as pd
import pytest
from hydra import initialize, compose
from omegaconf import DictConfig

from src.data import (
    read_datastore,
    preprocess_data,
)


@pytest.fixture
def cfg() -> DictConfig:
    """
    Load the test_config.yaml configuration file
    """
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="test_config")
    return cfg


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    df = read_datastore()
    return df


@pytest.fixture
def preprocessed_sample(raw_sample) -> Tuple[pd.DataFrame, pd.Series]:
    X, y = preprocess_data(raw_sample)
    return X, y
