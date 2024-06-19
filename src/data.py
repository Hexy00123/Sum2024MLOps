import os
import pandas as pd
import hydra
from omegaconf import DictConfig
import dvc.api


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def sample_data(cfg: DictConfig):
    # Read data
    data_path = hydra.utils.to_absolute_path(cfg.dataset.url)
    data = pd.read_csv(data_path)

    # Sort data by 'date_added'
    data = data.sort_values(by='date_added')

    # Read project stage from config
    project_stage = cfg.project_stage

    # Calculate the number of samples to take based on project stage
    num_samples = int(len(data) * project_stage / cfg.num_samples)

    # Take the first num_samples based on project_stage
    sampled_data = data.head(num_samples)

    # Ensure the output directory exists
    output_dir = os.path.join(hydra.utils.get_original_cwd(), "data", "samples")
    os.makedirs(output_dir, exist_ok=True)

    # Save sampled data
    sample_file = os.path.join(output_dir, f"sample.csv")
    sampled_data.to_csv(sample_file, index=False)
    print(f"Sampled data for stage {project_stage} saved to {sample_file}")


if __name__ == "__main__":
    sample_data()
