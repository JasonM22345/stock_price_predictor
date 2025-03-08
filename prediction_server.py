from flask import Flask, request, jsonify
import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

app = Flask(__name__)

# Define the neural network model
class StockPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def verify_api_key(api_key):
    # Check if the provided API key is valid
    return api_key == "Change_API_key"

def predict_stock_prices(api_key, input_file_path, end_date):
    # Verify the API key
    if not verify_api_key(api_key):
        return jsonify({"status": "Error", "message": "Invalid API key"}), 401

    try:
        # Load the input CSV file
        input_data = pd.read_csv(input_file_path)

        # Load the trained model
        model = StockPredictor(input_size=len(input_data.columns) - 2, output_size=2)
        model.load_state_dict(torch.load("model/trained_model.pth"))
        model.eval()

        # Normalize input data
        scaler = MinMaxScaler()
        input_features = input_data.iloc[:, :-2]
        input_features_scaled = pd.DataFrame(scaler.fit_transform(input_features), columns=input_features.columns)

        # Convert input data to PyTorch tensor
        input_tensor = torch.tensor(input_features_scaled.values).float()

        # Make predictions
        with torch.no_grad():
            predictions = model(input_tensor)

        # Denormalize predictions
        predicted_prices = pd.DataFrame(scaler.inverse_transform(predictions), columns=["Predicted_Close", "Predicted_Close"])

        # Add the predicted prices to the input data
        input_data = pd.concat([input_data, predicted_prices], axis=1)

        # Filter data up to the specified end date
        input_data = input_data[input_data["date"] <= end_date]

        # Return the new data
        return jsonify({"status": "Success", "data": input_data.to_dict(orient="records")})

    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

@app.route('/predict_stock_prices', methods=['POST'])
def predict_stock_prices_api():
    api_key = request.form.get('api_key')
    input_file = request.files.get('input_file')
    end_date = request.form.get('end_date')

    # Save the input CSV file temporarily
    temp_file_path = "temp_input_file.csv"
    input_file.save(temp_file_path)

    result, status_code = predict_stock_prices(api_key, temp_file_path, end_date)

    # Remove the temporary input file
    os.remove(temp_file_path)

    return result, status_code

if __name__ == '__main__':
    app.run(debug=True)

