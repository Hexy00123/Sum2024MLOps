# if running in wsl for the first time, running jupiter may be necessary
# jupyter execute notebooks/data_expectations.ipynb
# ./scripts/test_data.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Define the data sample file path
DATA_SAMPLE_PATH="data/samples"


# Step 1: Take a data sample
echo "Taking a data sample..."
python3 src/data.py project_stage=1

# Step 2: Validate the data sample
echo "Validating the data sample..."
if python3 tests/data_expectations.py; then
    echo "Data validation successful."
    # Step 3: Version the data sample with DVC
    echo "Versioning the data sample with DVC..."
    dvc add $DATA_SAMPLE_PATH
else
    echo "Data validation failed. Not versioning the data sample."
    exit 1
fi
