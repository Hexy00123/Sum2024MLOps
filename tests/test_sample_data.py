# TODO: fix working

import os
import pytest
import pandas as pd
from omegaconf import DictConfig
from hydra import initialize, compose
import hydra
from src.data import sample_data


# create a fixture to initialize the hydra config
@pytest.fixture
def cfg() -> DictConfig:
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="test_config")
    return cfg


class TestSampleData:
    """
    Test the sample_data function
    """
    def test_with_initialize(self, cfg) -> None:
        """
        Test the hydra config initialization
        """
        assert cfg == {
            "num_samples": 5,
            "dataset": {
                "url": "zameen-updated.csv",
                "name": "zameen-updated"
            },
            "project_stage": 1,
            "test": True
        }

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

        # Run the function twice with different project stages
        cfg1 = cfg.copy()
        cfg1.project_stage = 1
        sample_data(cfg1)
        sample_file = os.path.join(os.getcwd(), 'data', 'samples', 'test_sample.csv')
        sampled_data1 = pd.read_csv(sample_file)

        cfg2 = cfg.copy()
        cfg2.project_stage = 2
        sample_data(cfg2)
        sample_file = os.path.join(os.getcwd(), 'data', 'samples', 'test_sample.csv')

        # Load the generated samples
        sampled_data2 = pd.read_csv(sample_file)

        # Check if samples are not identical
        assert not sampled_data1.equals(sampled_data2)

        # Clean up
        os.remove(sample_file)

    def test_first_second_comparison(self, monkeypatch, cfg):
        """
        Test if the second sample has 2 times more data than the first sample
        """
        monkeypatch.setattr(hydra.utils, 'get_original_cwd', os.getcwd)

        # Run the function twice with different project stages
        cfg1 = cfg.copy()
        cfg1.project_stage = 1
        sample_data(cfg1)
        sample_file = os.path.join(os.getcwd(), 'data', 'samples', 'test_sample.csv')
        sampled_data1 = pd.read_csv(sample_file)

        cfg2 = cfg.copy()
        cfg2.project_stage = 2
        sample_data(cfg2)
        sample_file = os.path.join(os.getcwd(), 'data', 'samples', 'test_sample.csv')

        # Load the generated samples
        sampled_data2 = pd.read_csv(sample_file)

        # Check if samples are not identical
        assert len(sampled_data2) == 2 * len(sampled_data1)

        # Clean up
        os.remove(sample_file)
