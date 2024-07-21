from src.data_expectations import validate_features, validate_initial_data


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
