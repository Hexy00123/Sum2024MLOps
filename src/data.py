import os
import re

import hydra
import numpy as np
import pandas as pd
import zenml
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import zenml.client


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

    output_dir = os.path.join(
        hydra.utils.get_original_cwd(), "data", "samples")
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(data_path, index=False)
    print(f"Refactored data saved to {data_path}")


def read_datastore():
    # TODO: get versions with git and dvc checkouts
    sample_path = "data/samples/sample.csv"
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"File {sample_path} not found.")
    sample = pd.read_csv(sample_path)
    return sample


def one_hot_encode_feature(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(
        data[[column_name]])

    encoded_df = pd.DataFrame(ohe.transform(
        data[[column_name]]), columns=ohe.get_feature_names_out([column_name]))

    data = data.drop(column_name, axis=1)

    data = data.reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)

    data = pd.concat([data, encoded_df], axis=1)

    return data


def scale_feature(data: pd.DataFrame, column_name: str, strategy: str = 'std') -> pd.DataFrame:
    scalers = {
        'std': StandardScaler,
        'minmax': MinMaxScaler
    }
    if strategy not in scalers:
        raise NotImplementedError(f'Scaling is not implemented for {strategy}')

    scaler = scalers[strategy]().fit(data[[column_name]])

    data[column_name] = scaler.transform(data[[column_name]])
    return data


def cyclic_encoding(data: pd.DataFrame, column_name: str, max_value: int):
    data[column_name + '_sin'] = np.sin(2 *
                                        np.pi * data[column_name] / max_value)
    data[column_name + '_cos'] = np.sin(2 *
                                        np.pi * data[column_name] / max_value)
    data = data.drop(column_name, axis=1)
    return data


def preprocess_data(df: pd.DataFrame, only_X: bool = False):
    try:
        print(df)
        # Filter data
        df = df[df['latitude'] > 22][df['latitude'] < 38]
        df = df[df['longitude'] > 59][df['longitude'] < 79]

        print('data filtered')

        # put '-' instead of missing values in agent and agency
        df['agent'] = df['agent'].fillna('-')
        df['agency'] = df['agency'].fillna('-')

        print('missing values filled')

        # Preprocess datetime features
        df['day'] = df['date_added'].apply(lambda x: int(x.split('-')[1]))
        df['month'] = df['date_added'].apply(lambda x: int(x.split('-')[0]))
        df['year'] = df['date_added'].apply(lambda x: int(x.split('-')[2]))

        print('datetime features processed')

        # Convert metrics
        area_marla = np.where(df['Area Type'] == 'Kanal',
                              df['Area Size'] * 20, df['Area Size'])
        df['area'] = area_marla

        print('metrics converted')

        # Drop unnecessary/raw columns
        data = df.drop(['Area Type', 'Area Size', 'Area Category',
                       'property_id', 'page_url', 'date_added'], axis=1)

        print('unnecessary columns dropped')

        # Encoding features
        columns_to_one_hot = ['property_type', 'city',
                              'province_name', 'purpose', 'agency', 'agent', 'location']
        for column in columns_to_one_hot:
            data = one_hot_encode_feature(data, column)

        print('features encoded')

        # Scaling features
        columns_to_minmax = ['latitude', 'longitude', 'location_id', 'year']
        columns_to_std = ['area', 'baths', 'bedrooms', 'price']
        for column in columns_to_minmax:
            data = scale_feature(data, column, strategy='minmax')
        for column in columns_to_std:
            data = scale_feature(data, column)

        print('features scaled')

        # scale one-hot encoded features
        for column in columns_to_one_hot:
            print(f"Scaling one-hot encoded features for {column}")
            columns = [col for col in data.columns if col.startswith(f"{column}_")]
            for col in columns:
                data = scale_feature(data, col)

        print('one-hot encoded features scaled')
        print(data)

        # PCA for too large categorical features
        columns_to_pca = ['agency', 'agent', 'location']
        for column in columns_to_pca:
            print(f"Applying PCA to {column}")
            dummy_cols = [col for col in data.columns if col.startswith(f"{column}_") and col != 'location_id']
            dummies = data[dummy_cols]
            print(dummies)
            n_components = min(500, len(dummy_cols))
            print(n_components)
            pca_result = PCA(n_components=n_components).fit_transform(dummies)

            pca_df = pd.DataFrame(pca_result, columns=[
                                  f"{column}_{i}" for i in range(n_components)])

            if n_components < 500:
                for i in range(n_components, 500):
                    pca_df[f"{column}_{i}"] = 0

            data = data.reset_index(drop=True)
            data = pd.concat([data, pca_df], axis=1)
            data = data.drop(dummy_cols, axis=1)

        print('PCA applied')

        # Cyclic datetime encoding
        data = cyclic_encoding(data, 'day', 31)
        data = cyclic_encoding(data, 'month', 12)

        print('datetime encoded')

        X, y = data.drop('price', axis=1), data['price']

        print('preprocessing done')

        return X if only_X else X, y
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
        name=name, tag=version, sort_by="version").items

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


if __name__ == "__main__":
    sample_data()
    refactor_sample_data()
    
