import os
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from datetime import datetime
import sys
import select
import time

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        # Initialize your dataset
        pass

    def __len__(self):
        # Return the length of the dataset
        pass

    def __getitem__(self, index):
        # Implement how to get an item from the dataset
        pass

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

# Function to print status messages with timestamps
def print_status(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# Function to get user input with timeout
def get_user_input(prompt, timeout=60):
    print(prompt)
    available_gpus = [f"{i + 1}. {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())]
    gpu_list = "\n".join(available_gpus)
    print(f"Available GPUs:\n{gpu_list}")
    print(f"Please select GPUs by entering their index (e.g., '1 2' for multiple GPUs) or 'a' for all.")
    print(f"You have {timeout} seconds to make a selection.")

    rlist, _, _ = select.select([sys.stdin], [], [], timeout)

    if rlist:
        return sys.stdin.readline().strip()
    else:
        print_status(f"No input received within {timeout} seconds. Using all GPUs.")
        return "a"

try:
    # Get a list of all CSV files in the "stock_data" folder
    data_folder = "stock_data"
    file_paths = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".csv")]

    # Initialize empty lists to store consistent features and targets
    consistent_features = []
    consistent_targets = []

    # Check if GPU is available
    if torch.cuda.is_available():
        # Prompt user to select GPUs with a timeout
        gpu_selection = get_user_input("Select GPUs to use (enter GPU index or 'a' for all):")

        # Check if the user selected GPUs
        if gpu_selection and gpu_selection.lower() != "a":
            selected_gpu_indices = [int(index) - 1 for index in gpu_selection.split()]
            selected_gpus = [torch.cuda.get_device_name(index) for index in selected_gpu_indices]

            # Update CUDA devices
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpu_indices))
            device = torch.device("cuda")
            print_status(f"Using GPUs: {selected_gpus}")
        else:
            device = torch.device("cuda")
            print_status("Using all available GPUs.")

    else:
        device = torch.device("cpu")
        print_status("No GPU available, using CPU.")

    for file_path in file_paths:
        # Read each CSV file
        data = pd.read_csv(file_path)

        # Check if 'close' column is present
        if 'close' not in data.columns:
            raise ValueError(f"Error: 'close' column not found in {file_path}.")

        # Drop the first column ('date') and extract features and target
        data = data.drop(data.columns[0], axis=1, errors='ignore')

        # Drop lines with missing values or problematic data
        data = data.dropna()  # Drop rows with missing values
        data = data.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric values to NaN
        data = data.dropna()  # Drop rows with NaN values

        # Check if there are at least two columns remaining after dropping the first column
        if len(data.columns) < 2:
            raise ValueError(f"Error: Not enough columns remaining after dropping the first column in {file_path}.")

        # Calculate percentage differences for high, low, close based on the open value
        data['high_diff'] = (data['high'] - data['open']) / data['open']
        data['low_diff'] = (data['low'] - data['open']) / data['open']
        data['close_diff'] = (data['close'] - data['open']) / data['open']

        # Calculate percentage difference for volume based on the open value
        data['volume_diff'] = (data['volume'] - data['open']) / data['open']

        # Extract features and target
        features = data[['open']]  # Use double brackets to keep it as a DataFrame
        target = data[['high_diff', 'low_diff', 'close_diff', 'volume_diff']]

        # Append consistent features and targets to the lists
        consistent_features.append(features)
        consistent_targets.append(target)

        # Print message after loading each dataset
        print_status(f"Loaded {len(data)} lines of data from {file_path}. "
                     f"This code uses the 'Open' column as the base for percentage changes and considers "
                     f"the percentage changes in high, low, close columns, as well as the volume. "
                     f"Adjustments have been made to ensure consistent numbers of samples for features and targets.")

    # Concatenate consistent features and targets
    all_features = pd.concat(consistent_features, ignore_index=True)
    all_targets = pd.concat(consistent_targets, ignore_index=True)

    # Drop any remaining rows with missing values
    all_data = pd.concat([all_features, all_targets], axis=1)
    all_data = all_data.dropna()

    # Print the number of lines of consistent data loaded
    print_status(f"Loaded {len(all_data)} lines of consistent data.")

    # Normalize data
    scaler = MinMaxScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(all_features), columns=all_features.columns)

    # Split data into training and testing sets
    train_features, test_features, train_targets, test_targets = train_test_split(
        features_scaled, all_targets, test_size=0.2, random_state=42
    )

    # Convert data to PyTorch tensors
    train_data = TensorDataset(
        torch.tensor(train_features.values).float(),
        torch.tensor(train_targets.values).float()
    )

    # Initialize the model, loss function, and optimizer
    input_size = len(all_features.columns)
    output_size = len(all_targets.columns)

    # Use DataParallel to utilize all available GPUs
    model = StockPredictor(input_size, output_size).to(device)

    if torch.cuda.device_count() > 1:
        print_status(f"Using {torch.cuda.device_count()} GPUs: {selected_gpus}")
        model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model with more epochs
    num_epochs = 100  # Increased number of epochs
    batch_size = 128

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for batch_features, batch_target in train_loader:
            optimizer.zero_grad()  # Zero the gradients to prevent accumulation
            outputs = model(batch_features.to(device))
            loss = criterion(outputs, batch_target.to(device))
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

        print_status(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Save the trained model
    model_folder = "model"
    os.makedirs(model_folder, exist_ok=True)
    model_filename = os.path.join(model_folder, "trained_model.pth")
    torch.save(model.state_dict(), model_filename)
    print_status(f"Model trained and saved to {model_filename}")

except Exception as e:
    print_status(f"Error: {e}")

