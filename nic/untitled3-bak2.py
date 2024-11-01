# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from pyswarms.single.global_best import GlobalBestPSO
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torchsummary import summary

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

"""**Check GPU Availibility**"""

print(f"CUDA is available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

"""**Define the device**"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

"""**LOAD DATA**

"""



file_path = 'NF-CSE-CIC-IDS2018-v2.csv'
meta_data= pd.read_csv(file_path)

print(meta_data.shape)
print(meta_data.columns)
Attack_counts = meta_data['Attack'].value_counts()
print(Attack_counts)

print(meta_data.isnull().sum())
meta_data = meta_data.dropna()

sample_size = int(len(meta_data) * 0.1)  # using 1% dataset
data = meta_data.sample(n=sample_size, random_state=42)


data.drop(columns=['Label','IPV4_SRC_ADDR', 'IPV4_DST_ADDR'],inplace = True)
data.rename(columns={"Attack":"label"}, inplace=True)

label_counts = data['label'].value_counts()
print(label_counts)

"""**Split DATA**"""

# Split data
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['label']), data['label'], test_size=0.2, random_state=42)

# Normalize data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

normal_data = X_train_scaled[y_train == 'Benign']
anomaly_data = X_train_scaled[y_train != 'Benign']

# Add 0.01% anomaly data to normal_data==========================
#anomaly_fraction = 0.01  # 0.01%
#num_anomalies = int(len(normal_data) * anomaly_fraction)
#anomaly_sample = anomaly_data[np.random.choice(len(anomaly_data), num_anomalies, replace=False)]
# Combine normal data with the sampled anomalies
#normal_data = np.vstack([normal_data, anomaly_sample])
#============================

#only feature data and benign, no label data
x_train, x_val = train_test_split(normal_data, test_size=0.2)
x_train = torch.from_numpy(x_train).float().to(device)
x_val = torch.from_numpy(x_val).float().to(device)

# only feature data , no label data
test_tensor=torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

#only feature data and benign, no label data
normal_tensor=torch.tensor(normal_data, dtype=torch.float32).to(device)

"""**Define the Threshold and Detection Anormalies Functions**"""

# @title
def calculate_threshold(model, data):
    """Calculate threshold based on reconstruction error of normal data"""
    model.eval()
    with torch.no_grad():
        normal_flag = model(data)
        mse = nn.MSELoss(reduction='none')(normal_flag, data)
        normal_flag_errors = mse.mean(dim=1).cpu().numpy()
        return np.percentile(normal_flag_errors, 99)

def detect_anomalies(model, data, threshold):
    """Detect anomalies using the trained model"""
    model.eval()
    with torch.no_grad():
        reconstructions = model(data)
        mse = nn.MSELoss(reduction='none')(reconstructions, data)
        reconstruction_errors = mse.mean(dim=1).cpu().numpy()
        return reconstruction_errors > threshold

"""**Setting Banchmark**"""

# @title
# Define Autoencoder model
class Autoencoder_base(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder_base, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout with 20% probability
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_eval(x_train, x_val, epochs, batch_size, input_dim):
    best_loss = float('inf')
    # Early stopping parameters
    patience = 5  # Stop if no improvement for 5 consecutive epochs
    min_delta = 0.001  # Minimum improvement threshold
    patience_counter = 0

    model = Autoencoder_base(input_dim, 16).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    train_loader_b = DataLoader(TensorDataset(x_train, x_train), batch_size=batch_size, shuffle=True)
    val_loader_b = DataLoader(TensorDataset(x_val, x_val), batch_size=batch_size)
    best_model_state = None
    train_loss_baseline_all = []
    val_loss_baseline_all = []

    # Training loop with progress display
    for epoch in range(epochs):
        model.train()
        train_loss_b = 0
        num_batches = 0

        # Show progress for each batch in the current epoch
        with tqdm(total=len(train_loader_b), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for batch in train_loader_b:
                batch = batch[0].to(device)  # Move batch to GPU
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()

                train_loss_b += loss.item()
                num_batches += 1
                pbar.set_postfix({"Train Loss": train_loss_b / num_batches})
                pbar.update(1)

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
            best_model_state = model.state_dict().copy()  # 存储模型状态而不是保存到文件
        else:
            pass
            # patience_counter += 1
            # if patience_counter >= patience:
            #     print(f'Early stopping triggered at epoch {epoch+1}')
            #     break

    model.load_state_dict(best_model_state)
    return model, train_loss_baseline_all, val_loss_baseline_all


epochs = 50
batch_size = 32
input_dim = normal_data.shape[1]
model_baseline, train_losses_baseline, val_losses_baseline = train_eval(x_train=x_train, x_val=x_val,
                                             epochs=epochs, batch_size=batch_size, input_dim=input_dim)
threshold_baseline = calculate_threshold(model_baseline, normal_tensor)

y_pred_baseline = detect_anomalies(model_baseline, test_tensor, threshold_baseline)
y_test_binary = (y_test != 'Benign').astype(int)
Accuracy_baseline = np.mean(y_pred_baseline == y_test_binary)

# Print results
print(f'Accuracy: {Accuracy_baseline}')
print(f'Threshold: {threshold_baseline}')
print(classification_report(y_test_binary, y_pred_baseline))
print(confusion_matrix(y_test_binary, y_pred_baseline))

plt.figure(figsize=(10, 6))
plt.plot(train_losses_baseline, label='Train Loss baseline')
plt.plot(val_losses_baseline, label='Validation Loss baseline')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
