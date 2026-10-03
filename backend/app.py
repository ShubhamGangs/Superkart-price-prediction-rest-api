# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_sales_forecaster_api = Flask("SuperKart Store Sales Forecaster")

# Load the trained machine learning model
model = joblib.load("superkart_model.joblib")

# Define a route for the home page (GET request)
@superkart_sales_forecaster_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Store Sales Forecasting API!"

# Define an endpoint for single prediction (POST request)
@superkart_sales_forecaster_api.post('/v1/predict')
def predict_sales():
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing product and store details and returns
    the predicted sales as a JSON response.
    """
    # Get the JSON data from the request body
    input_data_json = request.get_json()

    # Extract relevant features from the JSON data
    sample = {
        'Product_Weight': input_data_json['Product_Weight'],
        'Product_Allocated_Area': input_data_json['Product_Allocated_Area'],
        'Product_MRP': input_data_json['Product_MRP'],
        'Store_Age_Years': input_data_json['Store_Age_Years'],
        'Product_Sugar_Content': input_data_json['Product_Sugar_Content'],
        'Product_Type': input_data_json['Product_Type'],
        'Store_Size': input_data_json['Store_Size'],
        'Store_Location_City_Type': input_data_json['Store_Location_City_Type'],
        'Store_Type': input_data_json['Store_Type'],
        'Store_Age_Category': input_data_json['Store_Age_Category']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_df = pd.DataFrame([sample])

    # Make prediction directly
    predicted_sales = model.predict(input_df)[0]

    # Convert to Python float and round
    predicted_sales = round(float(predicted_sales), 2)

    # Return predicted sales
    return jsonify({'Predicted Sales (in dollars)': predicted_sales})


# Define an endpoint for batch prediction (POST request)
@superkart_sales_forecaster_api.post('/v1/predictbatch')
def predict_sales_batch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing details for multiple products
    and returns the predicted sales as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_df = pd.read_csv(file)

    # Make predictions for all rows in the DataFrame
    predicted_sales_raw = model.predict(input_df).tolist()

    # Format and round predictions
    predicted_sales_values = [round(float(val), 2) for val in predicted_sales_raw]

    # Create a dictionary of predictions with product IDs or row index as keys
    if 'Product_Id' in input_df.columns:
        ids = input_df['Product_Id'].tolist()
    else:
        ids = list(range(len(predicted_sales_values)))
        
    output_dict = dict(zip(ids, predicted_sales_values))

    # Return the predictions dictionary as a JSON response
    return jsonify(output_dict)

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_sales_forecaster_api.run(debug=True)
