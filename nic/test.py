import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import os

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check GPU Availability
print(f"CUDA is available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Data
file_path = 'NF-CSE-CIC-IDS2018-v2.csv'
meta_data = pd.read_csv(file_path)

print(meta_data.shape)
print(meta_data.columns)
Attack_counts = meta_data['Attack'].value_counts()
print(Attack_counts)

print(meta_data.isnull().sum())
meta_data = meta_data.dropna()

data = meta_data
data.drop(columns=['Label', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR'], inplace=True)
data.rename(columns={"Attack": "label"}, inplace=True)

label_counts = data['label'].value_counts()
print(label_counts)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=['label']), data['label'], test_size=0.2, random_state=42
)

# Normalize Data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

normal_data = X_train_scaled[y_train == 'Benign']
anomaly_data = X_train_scaled[y_train != 'Benign']

x_train, x_val = train_test_split(normal_data, test_size=0.2)
x_train = torch.from_numpy(x_train).float().to(device)
x_val = torch.from_numpy(x_val).float().to(device)

test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
normal_tensor = torch.tensor(normal_data, dtype=torch.float32).to(device)

# Define the Autoencoder model
class Autoencoder_base(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder_base, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Model path for saving and loading
model_path = 'autoencoder_baseline.pth'

def train_eval(x_train, x_val, epochs, batch_size, input_dim):
    # Check if model already exists
    if os.path.exists(model_path):
        print("Loading saved model...")
        model = Autoencoder_base(input_dim, 16).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model, [], []

    best_loss = float('inf')
    patience = 5
    min_delta = 0.001
    patience_counter = 0

    model = Autoencoder_base(input_dim, 16).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    train_loader_b = DataLoader(TensorDataset(x_train, x_train), batch_size=batch_size, shuffle=True)
    val_loader_b = DataLoader(TensorDataset(x_val, x_val), batch_size=batch_size)
    best_model_state = None
    train_loss_baseline_all = []
    val_loss_baseline_all = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss_b = 0
        num_batches = 0
        for batch in train_loader_b:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            train_loss_b += loss.item()
            num_batches += 1
        train_loss_b /= num_batches
        train_loss_baseline_all.append(train_loss_b)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, _ in val_loader_b:
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_x).item()
        val_loss /= len(val_loader_b)
        val_loss_baseline_all.append(val_loss)

        # Early stopping check
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

    # Save the best model state if training was conducted
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), model_path)
    print("Model saved.")

    return model, train_loss_baseline_all, val_loss_baseline_all

# Training or loading the model
epochs = 50
batch_size = 32
input_dim = normal_data.shape[1]
model_baseline, train_losses_baseline, val_losses_baseline = train_eval(
    x_train=x_train, x_val=x_val, epochs=epochs, batch_size=batch_size, input_dim=input_dim
)

# Define threshold and detect anomalies
def calculate_threshold(model, data, percentile):
    model.eval()
    with torch.no_grad():
        normal_flag = model(data)
        mse = nn.MSELoss(reduction='none')(normal_flag, data)
        normal_flag_errors = mse.mean(dim=1).cpu().numpy()
        return np.percentile(normal_flag_errors, percentile)

def detect_anomalies(model, data, threshold):
    model.eval()
    with torch.no_grad():
        reconstructions = model(data)
        mse = nn.MSELoss(reduction='none')(reconstructions, data)
        reconstruction_errors = mse.mean(dim=1).cpu().numpy()
        return reconstruction_errors > threshold

# Binary encoding of labels for evaluation
y_test_binary = (y_test != 'Benign').astype(int)

# Set a fixed threshold at the 95th percentile for evaluation
starting_threshold_percentile = 94.5
optimal_threshold = calculate_threshold(model_baseline, normal_tensor, starting_threshold_percentile)
y_pred_optimized = detect_anomalies(model_baseline, test_tensor, optimal_threshold)

# Evaluate the results with the fixed threshold
accuracy_optimized = np.mean(y_pred_optimized == y_test_binary)
f1_class_1 = f1_score(y_test_binary, y_pred_optimized, pos_label=1)

# Print results
print(f'Fixed Threshold Percentile: {starting_threshold_percentile}')
print(f'Fixed Threshold Value: {optimal_threshold}')
print(f'Optimized Accuracy: {accuracy_optimized}')
print(f'Optimized F1 Score for class 1: {f1_class_1}')
print(classification_report(y_test_binary, y_pred_optimized))
print(confusion_matrix(y_test_binary, y_pred_optimized))

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses_baseline, label='Train Loss baseline')
plt.plot(val_losses_baseline, label='Validation Loss baseline')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
