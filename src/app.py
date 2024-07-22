import datetime
import json

import gradio as gr
import mlflow
import pandas as pd
import requests
# from gradio_calendar import Calendar
from hydra import compose, initialize

from data import transform_data
from model import retrieve_model_with_version

with initialize(config_path="../configs", version_base=None):
    cfg = compose(config_name="main")


# You need to define a parameter for each column in your raw dataset

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
    features = {"property_id": property_id,
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
                "Area Category": area_category}

    # print(features)

    # Build a dataframe of one row
    raw_df = pd.DataFrame(features, index=[0])

    model_name = cfg.model.best_model_name
    model_version = cfg.model.best_model_version

    model: mlflow.pyfunc.PyFuncModel = retrieve_model_with_version(model_name, model_version)
    # This will read the saved transformers "v4" from ZenML artifact store
    # And only transform the input data (no fit here).
    X = transform_data(
        df=raw_df,
        model=model,
    )

    # Convert it into JSON
    example = X.iloc[0, :]

    example = json.dumps(
        {"inputs": example.to_dict()}
    )

    payload = example

    # Send POST request with the payload to the deployed Model API
    # Here you can pass the port number at runtime using Hydra
    response = requests.post(
        url=f"http://localhost:{cfg.deploy.model_port}/invocations",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    # Change this to some meaningful output for your model
    # For classification, it returns the predicted label
    # For regression, it returns the predicted value
    return response.json()


# Only one interface is enough
demo = gr.Interface(
                    fn=predict,

                    # Here, the arguments in `predict` function
                    # will populated from the values of these input components
                    inputs=[
                        # Select proper components for data types of the columns in your raw dataset
                        gr.Number(label="property_id"),
                        gr.Number(label="location_id"),
                        gr.Textbox(label="page_url"),
                        gr.Dropdown(label="property_type",
                                    choices=["House", "Flat", "Upper Portion", "Lower Portion", "Room", "Farm House", "Penthouse"]),
                        gr.Text(label="location"),
                        gr.Text(label="city"),
                        gr.Text(label="province_name"),
                        gr.Number(label="latitude"),
                        gr.Number(label="longitude"),
                        gr.Number(label="baths"), # slider
                        gr.Text(label="area"),  # Marla or Kanal + size
                        gr.Dropdown(label="purpose", choices=["For Sale", "For Rent"]),
                        gr.Number(label="bedrooms"), #slider
                        gr.Textbox(label="date_added"), # TODO: How to add datetime?
                        gr.Text(label="agency"),
                        gr.Text(label="agent"),
                        gr.Dropdown(label="area_type", choices=["Marla", "Kanal"]),
                        gr.Number(label="area_size"),
                        gr.Dropdown(label="area_category", # TODO: correct versions
                                    choices=["0-5 Marla", "5-10 Marla", "10-15 Marla", "15-20 Marla"])

                    ],

                    # The outputs here will get the returned value from `predict` function
                    outputs=gr.Text(label="prediction result"),

                    # This will provide the user with examples to test the API
                    examples="data/examples"
                    # data/examples is a folder contains a file `log.csv`
                    # which contains data samples as examples to enter by user
                    # when needed.
                    )

# Launch the web UI locally on port 5155
demo.launch(server_port=5155)

# Launch the web UI in Gradio cloud on port 5155
# demo.launch(share=True, server_port = 5155)
