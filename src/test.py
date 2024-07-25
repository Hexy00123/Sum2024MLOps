import pandas as pd
from requests import post
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import json
from data import preprocess_data
import numpy as np


if __name__ == '__main__':
    fields = ['property_id',
              'location_id',
              'page_url',
              'property_type',
              'location',
              'city',
              'province_name',
              'latitude',
              'longitude',
              'baths',
              'area',
              'purpose',
              'bedrooms',
              'date_added',
              'agency',
              'agent',
              'area_type',
              'area_size',
              'area_category']

    N = 5
    data = pd.read_csv('data/samples/sample.csv').iloc[:N, :]
    sample = preprocess_data(data, X_only=True)
    prices = np.array(list(data['price']))
    
    payload = json.dumps({"data": sample.to_dict(orient='records')})
    response = post(
        url="http://localhost:5001/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    # Process the response
    print("Response from Flask API:", response.text, response.json())

    predictions = np.array([i for x in response.json()['predictions'] for i in x])

    print(predictions)
    print(prices)

    # plt.plot([0,800_000_000], [0,800_000_000], color='red')
    plt.scatter(prices, predictions - prices)
    plt.xlabel('prices')
    plt.ylabel('predictions - prices')
    plt.savefig('temp.png')