#!/bin/bash

# Check if project stage argument is provided, default to 1 if not
if [ -z "$1" ]; then
    echo "Project stage argument missing. Defaulting to stage 1."
    PROJECT_STAGE=1
else
    PROJECT_STAGE=$1
fi

set -e

# Define the data sample file path
DATA_SAMPLE_PATH="data/samples"

# Step 1: Take a data sample
echo "Taking a data sample..."
python3 src/data.py index=$PROJECT_STAGE

# Step 2: Validate the data sample
echo "Validating the data sample..."
if python3 src/data_expectations.py; then
    echo "Data validation successful."

    # Step 3: Version the data sample with DVC
    echo "Versioning the data sample with DVC..."
    dvc add $DATA_SAMPLE_PATH

    # Tagging the dataset based on the project stage
    TAG="v${PROJECT_STAGE}.0"

    # Git add and commit
    git add $DATA_SAMPLE_PATH.dvc
    git commit -m "Add data version $TAG"
    git push

    # Git tag and push tags
    git tag -a "$TAG" -m "Add data version $TAG"
    git push --tags

    # Push DVC changes
    dvc push

else
    echo "Data validation failed. Not versioning the data sample."
    exit 1
fi
