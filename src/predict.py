import json
import requests
import hydra
from data import read_features


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

    # print(predictions)
    # predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

    # print("Predictions:", predictions)
    
    print(response.json())
    print("encoded target labels: ", example_target)
    # print("target labels: ", list(cfg.data.labels)[example_target])


if __name__ == "__main__":
    predict()
