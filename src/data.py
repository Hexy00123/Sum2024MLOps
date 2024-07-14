import os
import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import zenml
import re
import argparse


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def sample_data(cfg: DictConfig):
    filename = "sample.csv" if cfg.test is False else "test_sample.csv"
    data_path = hydra.utils.to_absolute_path('data/' + cfg.dataset.url)
    data = pd.read_csv(data_path)

    data = data.sort_values(by='date_added')

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
    
    data_path = hydra.utils.to_absolute_path('data/samples/' + filename)
    
    df = pd.read_csv(data_path)
    
    pattern = re.compile(r"^[0-9]+(,[0-9]+)?\s*(Kanal|Marla)$")
    regex=r"^[0-9]+(\.[0-9])?\s*(Kanal|Marla)$"
    pattern_perfect = re.compile(regex)
    
    # Function to replace commas with dots and round the number to one decimal place
    def format_area(value):
        match = pattern.match(value)
        perfect_match = pattern_perfect.match(value)
        if match:
            new_value = value.replace(',', '.')
            numeric_part, unit = new_value.split()
            rounded_value = f"{round(float(numeric_part), 1)} {unit}"
            return rounded_value
        elif perfect_match:
            return value  
        else:
            return None          
    
    df['area'] = df['area'].apply(format_area)

    df = df.dropna(subset=['area'])

    df = df[df['baths'] <= 30]

    output_dir = os.path.join(hydra.utils.get_original_cwd(), "data", "samples")
    os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(data_path, index=False)
    print(f"Refactored data saved to {data_path}")


def read_datastore():
    version = 3
    sample_path = "data/samples/sample.csv"
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
    try:
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

        # PCA for too large categorical features
        columns_to_pca = ['agency', 'agent', 'location']
        for column in columns_to_pca:
            dummies = pd.get_dummies(df[column])
            n_components = min(500, dummies.shape[1])
            pca_result = PCA(n_components=n_components).fit_transform(dummies)

            pca_df = pd.DataFrame(pca_result, columns=[f"{column}_{i}" for i in range(n_components)])

            if n_components < 500:
                for i in range(n_components, 500):
                    pca_df[f"{column}_{i}"] = 0

            df = df.reset_index(drop=True)
            df = pd.concat([df, pca_df], axis=1)
            df = df.drop(column, axis=1)

        print('PCA applied')

        # Drop unnecessary/raw columns
        data = df.drop(['Area Type', 'Area Size', 'Area Category', 'property_id', 'page_url', 'date_added'], axis=1)

        print('unnecessary columns dropped')

        # Encoding features
        data = one_hot_encode_feature(data, 'property_type')
        data = one_hot_encode_feature(data, 'city')
        data = one_hot_encode_feature(data, 'province_name')
        data = one_hot_encode_feature(data, 'purpose')

        print('features encoded')

        # Scaling features
        data = scale_feature(data, 'latitude', strategy='minmax')
        data = scale_feature(data, 'longitude', strategy='minmax')
        data = scale_feature(data, 'area')
        data = scale_feature(data, 'location_id', strategy='minmax')
        data = scale_feature(data, 'baths')
        data = scale_feature(data, 'bedrooms')
        data = scale_feature(data, 'year', strategy='minmax')
        data = scale_feature(data, 'price')

        print('features scaled')

        # Cyclic datetime encoding
        data = cyclic_encoding(data, 'day', 31)
        data = cyclic_encoding(data, 'month', 12)

        print('datetime encoded')

        X, y = data.drop('price', axis=1), data['price']

        print('preprocessing done')

        return X, y
    except Exception as e:
        print(f"Preprocessing failed: {e}")



def load_features(X: pd.DataFrame, y: pd.Series, version: int) -> None:
    y.rename('price', inplace=True)
    df = pd.concat([X, y], axis=1)
    tag = str(version)
    zenml.save_artifact(data=df, name="features_target", tags=[tag])


if __name__ == "__main__":
    sample_data()
    refactor_sample_data()
