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

def sample_data_stage(cfg: DictConfig, index: int, sample_file: str):
    """
    Helper function to sample data for a specific project stage
    """
    buf = cfg.copy()
    buf.index = index
    sample_data(buf)
    sampled_data = pd.read_csv(sample_file)
    return sampled_data

class TestSampleData:
    """
    Test the sample_data function
    """

    def test_sample_length(self, monkeypatch, cfg):
        """
        Test the length of the sampled data for each batch
        """
        # Mock the hydra.utils.get_original_cwd() to return the current working directory
        monkeypatch.setattr(hydra.utils, 'get_original_cwd', os.getcwd)

        # Load the dataset
        dataset_path = cfg.dataset.url
        data = pd.read_csv("data/" + dataset_path)
        total_length = len(data)
        split_size = total_length // cfg.num_samples

        sample_file = os.path.join(os.getcwd(), 'data', 'samples', 'test_sample.csv')

        for i in range(1, cfg.num_samples + 1):
            cfg.index = i
            sample_data(cfg)

            # Load the generated sample
            sampled_data = pd.read_csv(sample_file)

            # Calculate expected length of the sample
            if i == cfg.num_samples:
                expected_length = split_size + (total_length % cfg.num_samples)
            else:
                expected_length = split_size

            # Check the length of the sample
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

    def test_non_overlapping_samples(self, monkeypatch, cfg):
        """
        Test if the samples are non-overlapping for different project stages
        """
        monkeypatch.setattr(hydra.utils, 'get_original_cwd', os.getcwd)

        sample_file = os.path.join(os.getcwd(), 'data', 'samples', 'test_sample.csv')

        sampled_data1 = sample_data_stage(cfg, 1, sample_file)
        sampled_data2 = sample_data_stage(cfg, 2, sample_file)

        # Check if there is no overlap between samples
        intersection = pd.merge(sampled_data1, sampled_data2, how='inner', on=list(sampled_data1.columns))
        assert intersection.empty

        # Clean up
        os.remove(sample_file)

    def test_samples_order(self, monkeypatch, cfg):
        """
        Test if the samples are in the correct order and non-overlapping
        """
        monkeypatch.setattr(hydra.utils, 'get_original_cwd', os.getcwd)

        sample_file = os.path.join(os.getcwd(), 'data', 'samples', 'test_sample.csv')

        for i in range(1, cfg.num_samples + 1):
            sampled_data = sample_data_stage(cfg, i, sample_file)
            if i > 1:
                assert sampled_data['date_added'].min() >= previous_sample['date_added'].max()
            previous_sample = sampled_data

        # Clean up
        os.remove(sample_file)
