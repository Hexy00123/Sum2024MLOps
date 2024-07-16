import os
import subprocess
import zipfile


def download_kaggle_dataset(dataset: str, csv_name: str):
    """
    Download a Kaggle dataset using the Kaggle API.
    """
    src_folder = "src"
    data_folder = "data"

    os.chdir(src_folder)

    # Ensure the data folder exists
    data_path = os.path.join("..", data_folder)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    subprocess.run(["kaggle", "datasets", "download", "-d", dataset])

    # Find the downloaded zip file
    for file in os.listdir("."):
        if file.startswith(dataset.split("/")[1]) and file.endswith(".zip"):
            zip_path = file
            break

    # Unzip the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            os.rename(os.path.join(data_path, file), os.path.join(data_path, csv_name))

    os.remove(zip_path)

    print(
        f"Dataset downloaded, unzipped, and renamed to {csv_name} in the {data_folder} folder."
    )


if __name__ == "__main__":
    dataset = "howisusmanali/house-price-prediction-zameencom-dataset"
    csv_name = "zameen-updated.csv"
    download_kaggle_dataset(dataset, csv_name)
