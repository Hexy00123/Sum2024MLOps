import os
import re
import subprocess

import hydra
import mlflow
import numpy as np
import pandas as pd
import zenml
import zenml.client
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from hydra import compose, initialize


import os


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def sample_data(cfg: DictConfig):
    filename = "sample.csv" if cfg.test is False else "test_sample.csv"
    data_path = hydra.utils.to_absolute_path("data/" + cfg.dataset.url)
    data = pd.read_csv(data_path)

    data = data.sort_values(by="date_added")

    index = cfg.index
    total_length = len(data)
    num_splits = cfg.num_samples
    split_size = total_length // num_splits
    start_idx = (index - 1) * split_size
    end_idx = start_idx + split_size

    if index == num_splits:
        end_idx = total_length
    sampled_data = data.iloc[start_idx:end_idx]

    output_dir = os.path.join(
        hydra.utils.get_original_cwd(), "data", "samples")

    os.makedirs(output_dir, exist_ok=True)
    sample_file = os.path.join(output_dir, filename)
    sampled_data.to_csv(sample_file, index=False)
    print(f"Sampled data for stage {index} saved to {sample_file}")


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def refactor_sample_data(cfg: DictConfig):
    filename = "sample.csv" if cfg.test is False else "test_sample.csv"

    data_path = hydra.utils.to_absolute_path("data/samples/" + filename)

    df = pd.read_csv(data_path)

    pattern = re.compile(r"^[0-9]+(,[0-9]+)?\s*(Kanal|Marla)$")
    regex = r"^[0-9]+(\.[0-9])?\s*(Kanal|Marla)$"
    pattern_perfect = re.compile(regex)

    # Function to replace commas with dots and round the number to one decimal place
    def format_area(value):
        match = pattern.match(value)
        perfect_match = pattern_perfect.match(value)
        if match:
            new_value = value.replace(",", ".")
            numeric_part, unit = new_value.split()
            rounded_value = f"{round(float(numeric_part), 1)} {unit}"
            return rounded_value
        elif perfect_match:
            return value
        else:
            return None

    df["area"] = df["area"].apply(format_area)

    df = df.dropna(subset=["area"])

    df = df[df["baths"] <= 30]

    # filtering data by coordinates
    df = df[df['latitude'] > 22][df['latitude'] < 38]
    df = df[df['longitude'] > 59][df['longitude'] < 79]

    # print('data filtered')

    output_dir = os.path.join(
        hydra.utils.get_original_cwd(), "data", "samples")
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(data_path, index=False)
    print(f"Refactored data saved to {data_path}")


def read_datastore():
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="main")

    version = cfg.test_data_version if cfg.test else cfg.index
    try:
        subprocess.run(["dvc", "fetch"], check=True)
        subprocess.run(["dvc", "pull"], check=True)
        subprocess.run(
            ["git", "checkout", f"v{version}.0", f"{cfg.dvc_file_path}"], check=True)
        subprocess.run(["dvc", "pull"], check=True)
        subprocess.run(["dvc", "checkout", f"{cfg.dvc_file_path}"], check=True)

        sample_path = cfg.sample_path
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"File {sample_path} not found.")
        sample = pd.read_csv(sample_path)
        return sample
    finally:
        # Return to the HEAD state
        subprocess.run(["git", "checkout", "HEAD",
                       f"{cfg.dvc_file_path}"], check=True)
        subprocess.run(["dvc", "pull"], check=True)
        subprocess.run(["dvc", "checkout", f"{cfg.dvc_file_path}"], check=True)


def one_hot_encode_feature(data: pd.DataFrame, column_name: str, X_only: bool = False) -> pd.DataFrame:
    if X_only:
        ohe = zenml.load_artifact(f"{column_name}_encoder")
        print("DBG preprocess: encoder was loaded:", column_name)
    else:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(
            data[[column_name]])
        zenml.save_artifact(data=ohe, name=f"{column_name}_encoder")
        print("DBG preprocess: encoder was saved:", column_name)

    encoded_df = pd.DataFrame(ohe.transform(
        data[[column_name]]), columns=ohe.get_feature_names_out([column_name]))

    data = data.drop(column_name, axis=1)

    data = data.reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)

    data = pd.concat([data, encoded_df], axis=1)

    return data


def scale_feature(data: pd.DataFrame, column_name: str, strategy: str = 'std', X_only: bool = False) -> pd.DataFrame:
    scalers = {
        'std': StandardScaler,
        'minmax': MinMaxScaler,
    }
    if strategy not in scalers:
        raise NotImplementedError(f'Scaling is not implemented for {strategy}')

    if X_only:
        scaler = zenml.load_artifact(f"{column_name}_scaler")
        print("DBG preprocess: scaler was loaded:", column_name)
    else:
        scaler = scalers[strategy]().fit(data[[column_name]])
        zenml.save_artifact(data=scaler, name=f"{column_name}_scaler")
        print("DBG preprocess: scaler was saved:", column_name)

    data[column_name] = scaler.transform(data[[column_name]])

    return data


def cyclic_encoding(data: pd.DataFrame, column_name: str, max_value: int):
    data[column_name + '_sin'] = np.sin(2 *
                                        np.pi * data[column_name] / max_value)
    data[column_name + '_cos'] = np.sin(2 *
                                        np.pi * data[column_name] / max_value)
    data = data.drop(column_name, axis=1)
    return data


def preprocess_data(df: pd.DataFrame, X_only: bool = False):
    print('Retrieving config...')
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="categories")
    print(f'Config retrieved: {cfg}')

    try:
        # Preprocess datetime features
        df['day'] = df['date_added'].apply(lambda x: int(x.split('-')[1]))
        df['month'] = df['date_added'].apply(lambda x: int(x.split('-')[0]))
        df['year'] = df['date_added'].apply(lambda x: int(x.split('-')[2]))

        print('datetime features processed')

        # Cyclic datetime encoding
        df = cyclic_encoding(df, 'day', 31)
        df = cyclic_encoding(df, 'month', 12)

        print('cyclic encoding completed')

        # Convert metrics
        area_marla = np.where(df['Area Type'] == 'Kanal',
                              df['Area Size'] * 20, df['Area Size'])
        df['area'] = area_marla

        print('metrics converted')

        # Drop unnecessary/raw columns
        data = df.drop(['Area Type', 'Area Size', 'Area Category',
                        'property_id', 'page_url', 'date_added', 'agency', 'agent', 'location'], axis=1)

        print('unnecessary columns dropped')

        # Encoding features
        # columns_to_one_hot = ['property_type',
        #                       'city', 'province_name', 'purpose']
        print("DBG:", cfg.keys())
        columns_to_one_hot = cfg.keys()
        for column in columns_to_one_hot:
            data = one_hot_encode_feature(data, column, X_only=X_only)

        print('features encoded')

        # Scaling features
        columns_to_minmax = ['latitude', 'longitude', 'location_id', 'year']
        if X_only:
            columns_to_std = ['area', 'baths', 'bedrooms']
        else:
            columns_to_std = ['area', 'baths', 'bedrooms', 'price']

        for column in columns_to_minmax:
            data = scale_feature(
                data, column, strategy='minmax', X_only=X_only)
        for column in columns_to_std:
            data = scale_feature(data, column, X_only=X_only)

        print('features scaled')

        # scale one-hot encoded features
        for column in columns_to_one_hot:
            print(f"Scaling one-hot encoded features for {column}")
            columns = [
                col for col in data.columns if col.startswith(f"{column}_")]
            for col in columns:
                data = scale_feature(data, col, X_only=X_only)

        print('one-hot encoded features scaled')

        if X_only:
            # check that all columns from cfg are present in data, otherwise add them with 0 values
            for column in columns_to_one_hot:
                columns = [
                    col for col in data.columns if col.startswith(f"{column}_")]
                if set(columns) != set(cfg[column]):
                    for col in cfg[column]:
                        if col not in columns:
                            data[col] = float(0)

            print(data)

            print('preprocessing done')
            return data
        else:
            X, y = data.drop('price', axis=1), data['price']

            with open(hydra.utils.to_absolute_path(f'configs/categories.yaml'), 'w') as f:
                for column in columns_to_one_hot:
                    columns = [
                        col for col in X.columns if col.startswith(f"{column}_")]
                    f.write(f"{column}: {columns}\n")

            return X, y

    except Exception as e:
        print(f"Preprocessing failed: {e}")


def load_features(X: pd.DataFrame, y: pd.Series, version: int) -> None:
    y.rename('price', inplace=True)
    df = pd.concat([X, y], axis=1)
    tag = str(version)
    zenml.save_artifact(data=df, name="features_target", tags=[tag])


def read_features(name, version, size=1, logs=False):
    client = zenml.client.Client()
    artifacts = client.list_artifact_versions(
        name=name, tag=version, sort_by="updated").items

    df = artifacts[-1].load()
    df = df.sample(frac=size, random_state=88)

    if logs:
        print("size of df is ", df.shape)
        print("df columns: ", df.columns)

    X = df[df.columns[:-1]]
    y = df.price

    if logs:
        print("shapes of X,y = ", X.shape, y.shape)

    return X, y


def transform_data(df: pd.DataFrame, model: mlflow.pyfunc.PyFuncModel) -> pd.DataFrame:
    """
    Function for preprocessing data for inference
    :param df: A dataframe with raw data
    :param model: A model object
    :return: Transformed data
    """
    schema = model.metadata.get_input_schema()
    print(type(schema))
    columns = []
    for col in schema:
        column = col.name
        columns.append(column)
    print(type(columns[0]))
    print(columns)
    print("Original DataFrame columns: ", df.columns)

    X = preprocess_data(
        df=df,
        X_only=True,
    )
    print("DataFrame after preprocessing: ", X.columns)

    input_schema = columns
    # print("Missing from input schema:")
    for col in X.columns:
        if col not in input_schema:
            X.drop(col, axis=1, inplace=True)

    # fill the missing columns with 0
    for col in input_schema:
        if col not in X.columns:
            X[col] = float(0)
    X = X.astype(float)
    X = pd.DataFrame(X)
    return X


if __name__ == "__main__":
    # read_features('features_target', 1, size=0.01, logs=True)
    sample_data()
    refactor_sample_data()
