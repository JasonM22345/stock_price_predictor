import os
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from datetime import datetime

# Function to print status messages with timestamps
def print_status(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# Define a more complex neural network model
class StockPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)  # Increased complexity
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)  # Increased complexity
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Adjusted dropout rate
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

try:
    # Get a list of all CSV files in the "stock_data" folder
    data_folder = "stock_data"
    file_paths = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".csv")]

    # Concatenate data from all files
    all_data = pd.concat([pd.read_csv(file_path) for file_path in file_paths], ignore_index=True)

    # Check if 'close' column is present
    if 'close' not in all_data.columns:
        raise ValueError("Error: 'close' column not found in the concatenated data.")

    # Drop the first column ('date') and extract features and target
    all_data = all_data.drop(all_data.columns[0], axis=1, errors='ignore')

    # Check if there are at least two columns remaining after dropping the first column
    if len(all_data.columns) < 2:
        raise ValueError("Error: Not enough columns remaining after dropping the first column.")

    # Extract features and target
    features = all_data.iloc[:, :-2]  # Exclude the last two columns
    target = all_data.iloc[:, [-2, -1]]  # Last two columns ('close' and 'Close')

    # Print the number of lines of data loaded and discarded
    print_status(f"Loaded {len(all_data)} lines of data.")
    print_status(f"Discarded {sum(all_data.duplicated())} lines of duplicate data.")

    # Normalize data
    scaler = MinMaxScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    # Split data into training and testing sets
    train_features, test_features, train_target, test_target = train_test_split(
        features_scaled, target, test_size=0.2, random_state=42
    )

    # Convert data to PyTorch tensors
    train_data = TensorDataset(
        torch.tensor(train_features.values).float(),
        torch.tensor(train_target.values).float()
    )

    # Initialize the model, loss function, and optimizer
    input_size = len(features.columns)
    output_size = len(target.columns)
    model = StockPredictor(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model with more epochs
    num_epochs = 100  # Increased number of epochs
    batch_size = 128

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for batch_features, batch_target in train_loader:
            optimizer.zero_grad()  # Zero the gradients to prevent accumulation
            outputs = model(batch_features)
            loss = criterion(outputs, batch_target)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

        print_status(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Save the trained model
    model_folder = "model"
    os.makedirs(model_folder, exist_ok=True)
    model_filename = os.path.join(model_folder, "trained_model.pth")
    torch.save(model.state_dict(), model_filename)
    print_status(f"Model saved: {model_filename}")

except Exception as e:
    print_status(f"Error: {str(e)}")
