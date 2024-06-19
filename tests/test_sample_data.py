# TODO: fix working

import pytest
from omegaconf import OmegaConf, DictConfig
import hydra


@pytest.fixture(scope="session")
def cfg():
    # defaults:
    #   - _self_
    #
    # num_samples: 5
    #
    # dataset:
    #   url: "zameen-updated.csv"
    #   name: "zameen-updated"
    #
    # project_stage: 1
    #
    # test: True  # test mode
    cfg = DictConfig()
    return cfg

# tests/test_data.py

import os
import pandas as pd
import pytest

# Remove the hydra import from the test files

# You can keep the test functions as they are, but modify them to use the cfg fixture


def test_sample_length(cfg):
    from src.data import sample_data

    # Run the function
    sample_data(cfg)

    # Load the generated sample
    sample_file = os.path.join(os.getcwd(), 'data', 'samples', 'test_sample.csv')
    sampled_data = pd.read_csv(sample_file)

    dataset_path = cfg.dataset.url
    data = pd.read_csv(dataset_path)

    # Check the length of the sample
    expected_length = len(data) / cfg.num_samples  # Calculate expected length based on num_samples
    assert len(sampled_data) == expected_length

    # Clean up
    os.remove(sample_file)


def test_different_samples(cfg):
    from src.data import sample_data

    # Run the function twice with different project stages
    cfg1 = cfg.copy()
    cfg1.project_stage = 1
    sample_data(cfg1)
    sample_file1 = os.path.join(os.getcwd(), 'data', 'samples', 'test_sample1.csv')

    cfg2 = cfg.copy()
    cfg2.project_stage = 2
    sample_data(cfg2)
    sample_file2 = os.path.join(os.getcwd(), 'data', 'samples', 'test_sample2.csv')

    # Load the generated samples
    sampled_data1 = pd.read_csv(sample_file1)
    sampled_data2 = pd.read_csv(sample_file2)

    # Check if samples are not identical
    assert not sampled_data1.equals(sampled_data2)

    # Clean up
    os.remove(sample_file1)
    os.remove(sample_file2)
