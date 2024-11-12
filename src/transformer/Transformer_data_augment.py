import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_epochs = 100
losses = []

# Load trajectory data from JSON file
with open('0020-trajectories_for_training_3d_truth.json', 'r') as f:
    data = json.load(f)


# Data augmentation functions for flattened data
def scale_trajectory(sequence, scale_range=(0.9, 1.1)):
    scale_factor = np.random.uniform(*scale_range)
    scaled_sequence = []
    for point in sequence:
        scaled_point = [
            point[0] * scale_factor,  # x
            point[1] * scale_factor,  # y
            point[2] * scale_factor,  # z
            point[3]  # frame_index (no scaling)
        ]
        scaled_sequence.append(scaled_point)
    return scaled_sequence


def add_noise(sequence, noise_std=0.005):
    noisy_sequence = []
    for point in sequence:
        noisy_point = [
            point[0] + np.random.normal(0, noise_std),  # x
            point[1] + np.random.normal(0, noise_std),  # y
            point[2] + np.random.normal(0, noise_std),  # z
            point[3]  # frame_index (no noise)
        ]
        noisy_sequence.append(noisy_point)
    return noisy_sequence


# Dataset class with data augmentation
class TrajectoryDataset(Dataset):
    def __init__(self, data, augment_multiplier=100):
        self.data = data
        self.sequences = []
        self.labels = []
        self.augment_multiplier = augment_multiplier
        self.prepare_data()

    def prepare_data(self):
        for traj_data in self.data:
            trajectory = [[point['x'], point['y'], point['z'], point['frame_index']] for point in
                          traj_data["trajectory"]]
            key_points = traj_data["key_points"]

            # Prepare labels as the 3 key points (x, y, z)
            label = []
            for key in ["throw_point", "highest_point", "hit_point"]:
                if key in key_points:
                    point = key_points[key]
                    label.append([point['x'], point['y'], point['z']])
                else:
                    label.append([0, 0, 0])  # default if key point is missing
            self.sequences.append(trajectory)
            self.labels.append(label)

    def __len__(self):
        return len(self.sequences) * self.augment_multiplier

    def __getitem__(self, idx):
        # Determine which original sequence to use
        original_idx = idx % len(self.sequences)
        sequence = self.sequences[original_idx]
        label = self.labels[original_idx]

        # Apply data augmentation
        augmentation_type = idx // len(self.sequences) % 2
        if augmentation_type == 0:
            augmented_sequence = scale_trajectory(sequence)
        else:
            augmented_sequence = add_noise(sequence)

        # Convert to tensors
        sequence_tensor = torch.tensor(augmented_sequence, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return sequence_tensor, label_tensor


# Define the Transformer Encoder-Decoder model for coordinate prediction
class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_heads=4, num_layers=2, output_dim=3):
        super(TrajectoryTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * 3)  # 3 key points, each with x, y, z coordinates
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # Aggregate encoder outputs
        x = self.decoder(x)
        return x.view(-1, 3, 3)  # reshape to (batch, 3 key points, 3 coordinates)


# Instantiate Dataset and DataLoader
dataset = TrajectoryDataset(data, augment_multiplier=100)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Model, Loss and Optimizer
model = TrajectoryTransformer().to(device)  # Move model to GPU
criterion = nn.MSELoss()  # regression loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop with progress display
for epoch in range(num_epochs):
    total_loss = 0
    dataloader_tqdm = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
    for sequences, labels in dataloader_tqdm:
        sequences, labels = sequences.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()
        output = model(sequences)  # Shape: (batch_size, 3 key points, 3 coordinates)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        dataloader_tqdm.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")

# Plotting the loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.show()

# Save the trained model
model_file_path = "trained_trajectory_transformer_with_keypoints.pth"
torch.save(model.state_dict(), model_file_path)
print(f"Model saved to {model_file_path}")


# Load the model
def load_model(model_file_path):
    model = TrajectoryTransformer().to(device)
    model.load_state_dict(torch.load(model_file_path))
    model.eval()
    return model


# Prediction function to find the closest frame indices
def predict_key_points(model, trajectory_data):
    predictions = []
    for traj_data in trajectory_data:
        sequence = [[point['x'], point['y'], point['z'], point['frame_index']] for point in traj_data["trajectory"]]
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)  # Move to GPU

        with torch.no_grad():
            predicted_coordinates = model(sequence_tensor).squeeze(0).cpu().numpy()  # (3, 3) for x, y, z of key points

        # Find the closest frame_index for each predicted key point
        trajectory_coordinates = np.array(sequence)[:, :3]  # (N, 3) for x, y, z
        frame_indices = []
        for coord in predicted_coordinates:
            # Calculate Euclidean distance and get the closest frame index
            distances = np.linalg.norm(trajectory_coordinates - coord, axis=1)
            closest_index = np.argmin(distances)
            frame_indices.append(sequence[closest_index][3])  # Get frame_index from original data

        predictions.append(frame_indices)
    return predictions


# Load data and model
with open('0020-trajectories_for_training_3d_truth.json', 'r') as f:
    data = json.load(f)

model_file_path = "trained_trajectory_transformer_with_keypoints.pth"
loaded_model = load_model(model_file_path)

# Make predictions
predictions = predict_key_points(loaded_model, data)
print("Predicted Key Frame Indices:", predictions)
