import json
import requests
import hydra
from data import read_features
import zenml
from hydra import compose, initialize
import numpy as np


with initialize(config_path="../configs", version_base=None):
    cfg = compose(config_name="main")


column_name = cfg.data.target_cols[0]
scaler = zenml.load_artifact(f"{column_name}_scaler")


# type: ignore
@hydra.main(config_path="../configs", config_name="deploy", version_base=None)
def predict(cfg=None):
    print(cfg)

    X, y = read_features(name="features_target",
                         version=cfg.example_version)

    example = X.iloc[0, :]
    example_target = y[0]
    example = json.dumps({
        "inputs": example.to_dict()
    })

    payload = example

    response = requests.post(
        url=f"http://localhost:{cfg.port}/invocations",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    
    
    # take the value from {'predictions': [-0.467059463262558]}
    prediction = response.json()["predictions"]
    print("encoded prediction:", prediction)
    
    prediction = np.array(prediction)
    prediction = scaler.inverse_transform(prediction.reshape(-1, 1))
    print("decoded prediction:", prediction)

    print("encoded target labels:", example_target)
    print("decoded target labels:", scaler.inverse_transform(np.array([example_target]).reshape(-1, 1)))


if __name__ == "__main__":
    predict()
