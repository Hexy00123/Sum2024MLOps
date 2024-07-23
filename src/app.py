import datetime
import json
from copy import deepcopy
import gradio as gr
import pandas as pd
import requests
from hydra import compose, initialize
from data import preprocess_data

# Initialize Hydra configuration
with initialize(config_path="../configs", version_base=None):
    cfg = compose(config_name="main")

def predict(
        property_id=None,
        location_id=None,
        page_url=None,
        property_type=None,
        location=None,
        city=None,
        province_name=None,
        latitude=None,
        longitude=None,
        baths=None,
        area=None,
        purpose=None,
        bedrooms=None,
        date_added=None,
        agency=None,
        agent=None,
        area_type=None,
        area_size=None,
        area_category=None
):
    # This will be a dict of column values for input data sample
    features = {
        "property_id": property_id,
        "location_id": location_id,
        "page_url": page_url,
        "property_type": property_type,
        "location": location,
        "city": city,
        "province_name": province_name,
        "latitude": latitude,
        "longitude": longitude,
        "baths": baths,
        "area": area,
        "purpose": purpose,
        "bedrooms": bedrooms,
        "date_added": date_added,
        "agency": agency,
        "agent": agent,
        "Area Type": area_type,
        "Area Size": area_size,
        "Area Category": area_category
    }

    # Build a dataframe of one row
    raw_df = pd.DataFrame(features, index=[0])

    # Preprocess data
    X = preprocess_data(df=deepcopy(raw_df), X_only=True)
    print("Preprocessed Data:", X)

    # Convert it into JSON payload
    payload = json.dumps({"data": X.to_dict(orient='records')})

    # Send POST request
    response = requests.post(
        url="http://localhost:5001/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    # Process the response
    print("Response from Flask API:", response.text)
    try:
        result = response.json()
        # Assuming result is a list of lists, extract the first value from the first list
        predictions = result.get("predictions", [])
        if predictions and isinstance(predictions[0], list) and len(predictions[0]) > 0:
            prediction = predictions[0][0]
            # Format the result into a readable string
            return f"Predicted value: {prediction:.2f}"
        else:
            return "Prediction could not be made."
    except json.JSONDecodeError:
        return "Invalid response from Flask API"

# Define the Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="property_id"),
        gr.Number(label="location_id"),
        gr.Textbox(label="page_url"),
        gr.Dropdown(label="property_type", choices=["House", "Flat", "Upper Portion", "Lower Portion", "Room", "Farm House", "Penthouse"]),
        gr.Text(label="location"),
        gr.Text(label="city"),
        gr.Text(label="province_name"),
        gr.Number(label="latitude"),
        gr.Number(label="longitude"),
        gr.Number(label="baths"), # slider
        gr.Text(label="area"),  # Marla or Kanal + size
        gr.Dropdown(label="purpose", choices=["For Sale", "For Rent"]),
        gr.Number(label="bedrooms"), # slider
        gr.Textbox(label="date_added"), # TODO: How to add datetime?
        gr.Text(label="agency"),
        gr.Text(label="agent"),
        gr.Dropdown(label="area_type", choices=["Marla", "Kanal"]),
        gr.Number(label="area_size"),
        gr.Dropdown(label="area_category", choices=["0-5 Marla", "5-10 Marla", "10-15 Marla", "15-20 Marla"])
    ],
    outputs=gr.Text(label="prediction result"),
    examples="data/examples"
)

# Launch the web UI locally on port 5155
demo.launch(server_port=5155)
