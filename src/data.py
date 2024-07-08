import os
import hydra
import pandas as pd
from omegaconf import DictConfig

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

# TODO:
# 2. preprocess_data @Nooth1ng
# 3. load_features


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def sample_data(cfg: DictConfig):
    filename = "sample.csv" if cfg.test is False else "test_sample.csv"

    # Read data
    data_path = hydra.utils.to_absolute_path('data/' + cfg.dataset.url)
    data = pd.read_csv(data_path)

    # Sort data by 'date_added'
    data = data.sort_values(by='date_added')

    # Read project stage from config
    index = cfg.index

    # Calculate start and end indices for the sample
    total_length = len(data)
    num_splits = cfg.num_samples
    split_size = total_length // num_splits

    start_idx = (index - 1) * split_size
    end_idx = start_idx + split_size

    # Handle the case when index is the last one and may not divide evenly
    if index == num_splits:
        end_idx = total_length

    # Take the slice of data based on start and end indices
    sampled_data = data.iloc[start_idx:end_idx]

    # Ensure the output directory exists
    output_dir = os.path.join(
        hydra.utils.get_original_cwd(), "data", "samples")
    os.makedirs(output_dir, exist_ok=True)

    # Save sampled data
    sample_file = os.path.join(output_dir, filename)
    sampled_data.to_csv(sample_file, index=False)
    print(f"Sampled data for stage {index} saved to {sample_file}")


def read_datastore():
    version = 3
    sample_path = "../data/samples/sample.csv"
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"File {sample_path} not found.")
    sample = pd.read_csv(sample_path)
    return sample, version


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
    data[column_name+'_sin'] = np.sin(2 * np.pi * data[column_name]/max_value)
    data[column_name+'_cos'] = np.sin(2 * np.pi * data[column_name]/max_value)
    data = data.drop(column_name, axis=1)
    return data


def preprocess_data(df: pd.DataFrame):
    # Filter data
    df = df[df['latitude'] > 22][df['latitude'] < 38]
    df = df[df['longitude'] > 59][df['longitude'] < 79]

    # Preprocess datetime features
    df['day'] = df['date_added'].apply(lambda x: int(x.split('-')[1]))
    df['month'] = df['date_added'].apply(lambda x: int(x.split('-')[0]))
    df['year'] = df['date_added'].apply(lambda x: int(x.split('-')[2]))

    # Convert mertics
    area_marla = np.where(df['Area Type'] == 'Kanal',
                          df['Area Size'] * 20, df['Area Size'])
    df['area'] = area_marla

    # Drop unnecessary/raw columns
    data = df.drop(['Area Type', 'Area Size', 'Area Category', 'agency',
                    'agent', 'property_id', 'page_url', 'date_added'], axis=1)

    # Encoding features
    data = one_hot_encode_feature(data, 'property_type')
    data = one_hot_encode_feature(data, 'location')
    data = one_hot_encode_feature(data, 'city')
    data = one_hot_encode_feature(data, 'province_name')
    data = one_hot_encode_feature(data, 'purpose')

    # Scaling features
    data = scale_feature(data, 'latitude', strategy='minmax')
    data = scale_feature(data, 'longitude', strategy='minmax')
    data = scale_feature(data, 'area')
    data = scale_feature(data, 'location_id', strategy='minmax')
    data = scale_feature(data, 'baths')
    data = scale_feature(data, 'bedrooms')
    data = scale_feature(data, 'year', strategy='minmax')

    data = scale_feature(data, 'price')

    # Cyclic datetime encoding
    data = cyclic_encoding(data, 'day', 31)
    data = cyclic_encoding(data, 'month', 12)

    X, y = data.drop('price', axis=1), data['price']

    print(X.describe())

    return X, y


def load_features():
    # zenml
    ...


if __name__ == "__main__":
    sample_data()
