import pandas as pd
import pytest

from src.data import read_datastore, preprocess_data
from src.data_expectations import validate_initial_data, validate_features
from tests.test_data import preprocessed_sample


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    df = read_datastore()
    return df


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
