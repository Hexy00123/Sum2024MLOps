import os

import pandas as pd

from src.download_kaggle_dataset import download_kaggle_dataset


def test_download_kaggle_dataset():
    dataset = "howisusmanali/house-price-prediction-zameencom-dataset"
    csv_name = "zameen-updated.csv"

    # Call the function to download the dataset
    download_kaggle_dataset(dataset, csv_name)

    # Verify the CSV file exists in the data directory
    csv_path = os.path.join("data", csv_name)

    # Ensure we are checking the correct path
    root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    full_csv_path = os.path.join(root_dir, csv_path)

    print("Full CSV path:", full_csv_path)
    assert os.path.exists(full_csv_path), f"File {full_csv_path} does not exist"

    # Load the CSV file into a DataFrame and verify its type
    df = pd.read_csv(full_csv_path)
    assert isinstance(df, pd.DataFrame)

    assert df.shape == (168446, 20)

    # Verify no additional zip or csv files exist in the repo
    src_dir = os.path.join(root_dir, "src")
    data_dir = os.path.join(root_dir, "data")
    assert not any(f.endswith(".zip") for f in os.listdir(src_dir)), "Zip files still exist in src"
    assert not any(f.endswith(".csv") for f in os.listdir(src_dir)), "CSV files still exist in src"
    assert not any(f.endswith(".zip") for f in os.listdir(data_dir)), "Zip files still exist in data"
