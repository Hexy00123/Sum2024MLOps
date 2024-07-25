# api/app.py

from flask import Flask, request, jsonify, abort, make_response

import mlflow
import mlflow.pyfunc
import os
import sys
import pandas as pd
import zenml
from hydra import compose, initialize


mlflow.set_tracking_uri("http://localhost:5000")


with initialize(config_path="../configs", version_base=None):
    cfg = compose(config_name="main")

model = mlflow.pyfunc.load_model(os.path.join("api", "model_dir"))
column_name = cfg.data.target_cols[0]
scaler = zenml.load_artifact(f"{column_name}_scaler")

app = Flask(__name__)


@app.route("/info", methods=["GET"])
def info():

    response = make_response(str(model.metadata), 200)
    response.content_type = "text/plain"
    return response


@app.route("/", methods=["GET"])
def home():
    msg = """
	Welcome to our ML service to predict Customer satisfaction\n\n

	This API has two main endpoints:\n
	1. /info: to get info about the deployed model.\n
	2. /predict: to send predict requests to our deployed model.\n

	"""

    response = make_response(msg, 200)
    response.content_type = "text/plain"
    return response


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse the JSON request body
        data = request.get_json(force=True)
        print("Received Data:", data)

        # Check if 'data' field exists
        if 'data' not in data:
            return jsonify({"error": "No 'data' field in request"}), 400

        # Extract input data
        input_data = data['data']
        print("Input data:", input_data)

        # Check if input_data is a list
        if not isinstance(input_data, list):
            return jsonify({"error": "'data' should be a list of feature dictionaries"}), 400

        # Construct DataFrame
        if len(input_data) == 0:
            return jsonify({"error": "Empty data field"}), 400

        # Use keys of the first dictionary to set columns
        columns = list(input_data[0].keys())
        df = pd.DataFrame(input_data, columns=columns)

        # Make predictions
        predictions = model.predict(df)
        print("Predictions:", predictions)

        # Return predictions as JSON
        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        # Return error message as JSON
        return jsonify({"error": str(e)}), 400


# # This will run a local server to accept requests to the API.
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
