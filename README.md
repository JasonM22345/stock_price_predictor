# Stock Price Predictor

## Overview
The **Stock Price Predictor** project is a deep learning-based system designed to predict stock prices using historical data. It consists of two main components:
1. **Training System:** A neural network model is trained using historical stock data.
2. **Prediction Server:** A Flask-based API that loads the trained model and provides stock price predictions.

The training process utilizes **PyTorch** for deep learning and **scikit-learn** for data preprocessing. The model is trained in a **Docker container** that supports GPU acceleration for efficient training. The prediction system runs as a Flask API, allowing external users to request stock price predictions.

## Key Features
- **Deep Learning Model:** Uses a neural network with multiple layers and dropout regularization for stock price prediction.
- **Historical Data Processing:** Cleans and scales data from multiple stock files.
- **Training on GPU:** Runs inside a **Docker container** optimized for TensorFlow and PyTorch.
- **API-Based Prediction Service:** Provides stock price predictions via a REST API.

## How It Works (Algorithm Overview)
### Training Phase:
1. **Load and Preprocess Data:**
   - Historical stock price data is read from multiple CSV files.
   - The `close` column is extracted as the target variable.
   - Data is normalized using **MinMaxScaler**.
   - The dataset is split into training and testing sets.

2. **Train Neural Network Model:**
   - A deep learning model with three fully connected layers is created.
   - The model is trained using **Mean Squared Error (MSE) loss** and the **Adam optimizer**.
   - Training runs for 100 epochs with a batch size of 128.
   - The trained model is saved to a file (`trained_model.pth`).

### Prediction Phase:
1. **Load Trained Model:**
   - The model is loaded from the saved file.
   - The same preprocessing steps (scaling) are applied to the input data.

2. **Make Predictions:**
   - The input data is passed through the trained model.
   - Predictions are transformed back to their original scale.
   - The predicted stock prices are returned via the Flask API.

## Detailed Breakdown
### Training System (`training_v1.0.0.py`)
- Reads stock data from the `stock_data/` folder.
- Normalizes features using `MinMaxScaler`.
- Defines a **StockPredictor** neural network with ReLU activations and dropout.
- Splits data into training (80%) and testing (20%).
- Uses PyTorch `DataLoader` for batch processing.
- Saves the trained model to `model/trained_model.pth`.

#### Model Architecture
- **Input Layer**: The number of input features corresponds to the number of stock-related variables used for training.
- **Hidden Layers**:
  - **First Layer**: Fully connected layer with 512 neurons, followed by a ReLU activation function.
  - **Second Layer**: Fully connected layer with 256 neurons, followed by a ReLU activation function.
  - **Dropout Layer**: Applies a dropout rate of **0.3** to reduce overfitting.
- **Output Layer**: Produces two outputs (future predicted closing prices).

#### Training Process
- **Loss Function**: Uses **Mean Squared Error (MSE)** as the loss function to minimize the difference between predicted and actual prices.
- **Optimizer**: Utilizes the **Adam optimizer** with a learning rate of **0.0001**, which efficiently adapts the learning rate for better convergence.
- **Batch Processing**: The dataset is split into mini-batches of size **128**, allowing efficient training.
- **Epochs**: The model is trained for **100 epochs**, iterating through the dataset multiple times to refine weights.
- **Gradient Descent & Backpropagation**:
  - Computes gradients using **autograd** in PyTorch.
  - Updates weights using Adam’s adaptive moment estimation.
  - Ensures better convergence and minimizes loss.

### Prediction Server (`prediction_server.py`)
- Implements a **Flask API**.
- Accepts an input CSV file and API key.
- Loads the trained model.
- Normalizes input data before making predictions.
- Returns predicted stock prices in JSON format.

### Docker-Based Training (`training_container.py`)
- Uses a **Docker container** with TensorFlow and PyTorch.
- Copies the training script and stock data into the container.
- Installs necessary dependencies (`torch`, `pandas`, `scikit-learn`, etc.).
- Supports **GPU acceleration** using CUDA.
- Runs with:
  ```bash
  docker run -it -p 5000:5000 --gpus all --memory=80g --memory-swap=80g --shm-size=8g stock_prediction_image
  ```

## Installation and Execution
### Requirements
- Python 3.x
- PyTorch
- Pandas
- scikit-learn
- Flask (for API server)

### Running the Training Script
```bash
python3 training_v1.0.0.py
```
This will load stock data, train the model, and save it to `model/trained_model.pth`.

### Running the Prediction Server
```bash
python3 prediction_server.py
```
This starts a Flask API server that listens for requests.

### Making Predictions
Send a POST request with a CSV file:
```bash
curl -X POST -F "api_key=Change_API_key" -F "input_file=@test_stock_data.csv" -F "end_date=2025-01-01" http://localhost:5000/predict_stock_prices
```

## Notes
- The model trains on **Microsoft (MSFT), Oracle (ORCL), Adobe (ADBE), Autodesk (ADSK), and other stocks**.
- The `FilesToUpload/` folder in Docker contains all scripts and data for training.
- The API key must be updated for authentication.

## License
This project is open-source and available for modification and improvement.

