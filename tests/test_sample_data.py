import os

import hydra
import pandas as pd
import pytest
from hydra import initialize, compose
from omegaconf import DictConfig

from src.data import sample_data


@pytest.fixture
def cfg() -> DictConfig:
    """
    Load the test_config.yaml configuration file
    """
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="test_config")
    return cfg


def sample_data_stage(cfg: DictConfig, stage: int, sample_file: str):
    """
    Helper function to sample data for a specific project stage
    """
    buf = cfg.copy()
    buf.project_stage = stage
    sample_data(buf)
    sampled_data = pd.read_csv(sample_file)
    return sampled_data


class TestSampleData:
    """
    Test the sample_data function
    """

    def test_sample_length(self, monkeypatch, cfg):
        """
        Test the length of the sampled data
        """
        # Mock the hydra.utils.get_original_cwd() to return the current working directory
        monkeypatch.setattr(hydra.utils, 'get_original_cwd', os.getcwd)

        # Run the function
        sample_data(cfg)

        # Load the generated sample
        sample_file = os.path.join(os.getcwd(), 'data', 'samples', 'test_sample.csv')
        sampled_data = pd.read_csv(sample_file)

        dataset_path = cfg.dataset.url
        data = pd.read_csv(dataset_path)

        # Check the length of the sample
        expected_length = len(data) // cfg.num_samples  # Calculate expected length based on num_samples
        assert len(sampled_data) == expected_length

        # Clean up
        os.remove(sample_file)

    def test_different_samples(self, monkeypatch, cfg):
        """
        Test if the samples are different for different project stages
        """
        monkeypatch.setattr(hydra.utils, 'get_original_cwd', os.getcwd)

        sample_file = os.path.join(os.getcwd(), 'data', 'samples', 'test_sample.csv')

        sampled_data1 = sample_data_stage(cfg, 1, sample_file)
        sampled_data2 = sample_data_stage(cfg, 2, sample_file)

        # Check if samples are not identical
        assert not sampled_data1.equals(sampled_data2)

        # Clean up
        os.remove(sample_file)

    def test_samples_comparison(self, monkeypatch, cfg):
        """
        Test if each sample increases in size by 1/5 of the original data
        """
        monkeypatch.setattr(hydra.utils, 'get_original_cwd', os.getcwd)

        sample_file = os.path.join(os.getcwd(), 'data', 'samples', 'test_sample.csv')

        first_sample_len = len(sample_data_stage(cfg, 1, sample_file))
        for i in range(2, 6):
            sampled_data = sample_data_stage(cfg, i, sample_file)
            j = 1 if i == 5 else 0
            assert len(sampled_data) == i * first_sample_len + j

        # Clean up
        os.remove(sample_file)

    # TODO: Add a test that checks if the sampled data is sorted by 'date_added',
    #  and that new samples are supersets of previous samples.
    # That next sample date_added are greater or equal to the previous sample date_added.
