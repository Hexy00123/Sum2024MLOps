# run with: python src/data.py

import os
import pandas as pd
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def sample_data(cfg: DictConfig):
    # Read data
    data_path = hydra.utils.to_absolute_path(cfg.dataset.url)
    data = pd.read_csv(data_path)

    # Sort data by 'date_added'
    data = data.sort_values(by='date_added')

    # Read number of samples from config
    num_samples = cfg.num_samples

    # Split the data into num_samples equal parts
    splits = [data.iloc[i::num_samples, :] for i in range(num_samples)]

    # Ensure the output directory exists
    output_dir = os.path.join(hydra.utils.get_original_cwd(), "data\\samples")
    os.makedirs(output_dir, exist_ok=True)

    # Save each split
    for i, split in enumerate(splits):
        sample_file = os.path.join(output_dir, f"sample_part_{i + 1}.csv")
        split.to_csv(sample_file, index=False)
        print(f"Sampled data saved to {sample_file}")


if __name__ == "__main__":
    sample_data()
