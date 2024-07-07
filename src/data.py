import os
import hydra
import pandas as pd
from omegaconf import DictConfig

#TODO:
# 2. preprocess_data @Bulat
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
    output_dir = os.path.join(hydra.utils.get_original_cwd(), "data", "samples")
    os.makedirs(output_dir, exist_ok=True)

    # Save sampled data
    sample_file = os.path.join(output_dir, filename)
    sampled_data.to_csv(sample_file, index=False)
    print(f"Sampled data for stage {index} saved to {sample_file}")


@hydra.main(config_path="../configs", config_name="data_version", version_base=None)
def read_datastore(cfg: DictConfig):
    # which returns the sample as a dataframe/tensor.
    version = cfg.data_version
    sample_path = hydra.utils.to_absolute_path(cfg.sample_path)
    sample = pd.read_csv(sample_path)
    return sample, version


def preprocess_data():
    ...


def load_features():
    # zenml
    ...


if __name__ == "__main__":
    sample_data()
