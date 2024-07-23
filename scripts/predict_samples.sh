#!/bin/bash
mlflow run . --env-manager local -e predict -P example_version=2 -P port=5123

echo "Prediction completed successfully."


# Define the base command for the MLflow project
base_command="mlflow run . --env-manager local -e predict"

# Loop through each version from 1 to 5
for version in {1..5}
do
  echo "Running prediction for version $version"
  
  # Run the MLflow project with the specified parameters
  $base_command -P example_version=$version -P port=5123

  # Check the status of the last executed command
  if [ $? -eq 0 ]; then
    echo "Prediction for version $version completed successfully."
  else
    echo "Prediction for version $version failed."
    exit 1
  fi
done

echo "All predictions completed successfully."
