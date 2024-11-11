import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # For showing training progress

num_epochs = 100
losses = []

# Load trajectory data from JSON file
with open('0020-trajectories_for_training_3d_truth.json', 'r') as f:
    data = json.load(f)

# Data augmentation functions
def scale_trajectory(sequence, scale_range=(0.9, 1.1)):
    scale_factor = np.random.uniform(*scale_range)
    scaled_sequence = []
    for point in sequence:
        scaled_point = [
            point[0] * scale_factor,  # x
            point[1] * scale_factor,  # y
            point[2] * scale_factor,  # z
            point[3]                  # frame_index remains the same
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
            point[3]                                    # frame_index remains the same
        ]
        noisy_sequence.append(noisy_point)
    return noisy_sequence

# Dataset class that processes data with data augmentation
class TrajectoryDataset(Dataset):
    def __init__(self, data, augment_multiplier=100):
        self.data = data
        self.sequences = []
        self.labels = []
        self.augment_multiplier = augment_multiplier
        self.prepare_data()

    def prepare_data(self):
        for traj_data in self.data:
            trajectory = traj_data["trajectory"]
            key_points = traj_data["key_points"]

            # Convert trajectory points into (x, y, z, frame_index) sequences
            sequence = [[point['x'], point['y'], point['z'], point['frame_index']] for point in trajectory]
            label_sequence = [0] * len(sequence)  # Default label is 0 (non-key point)

            # Annotate key points
            for key, key_point in key_points.items():
                # Find the index of the key point based on its frame_index
                frame_index = key_point['frame_index']
                index_in_sequence = next(
                    (i for i, pt in enumerate(trajectory) if pt['frame_index'] == frame_index), None
                )
                if index_in_sequence is not None:
                    if key == 'throw_point':
                        label_sequence[index_in_sequence] = 1
                    elif key == 'highest_point':
                        label_sequence[index_in_sequence] = 2
                    elif key == 'hit_point':
                        label_sequence[index_in_sequence] = 3

            self.sequences.append(sequence)
            self.labels.append(label_sequence)

    def __len__(self):
        return len(self.sequences) * self.augment_multiplier

    def __getitem__(self, idx):
        # Determine which original sequence to use
        original_idx = idx % len(self.sequences)
        sequence = self.sequences[original_idx]
        label_sequence = self.labels[original_idx]

        # Apply data augmentation
        augmentation_type = idx // len(self.sequences) % 2
        if augmentation_type == 0:
            # Apply scaling
            augmented_sequence = scale_trajectory(sequence)
        else:
            # Apply noise
            augmented_sequence = add_noise(sequence)

        # Convert to tensors
        sequence_tensor = torch.tensor(augmented_sequence, dtype=torch.float32)
        label_tensor = torch.tensor(label_sequence, dtype=torch.long)

        return sequence_tensor, label_tensor

# Define the Transformer model (updated input_dim to 4)
class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_classes=4, num_heads=4, num_layers=2):
        super(TrajectoryTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.classifier(x)
        return x

# Instantiate Dataset and DataLoader
dataset = TrajectoryDataset(data, augment_multiplier=100)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Model, Loss and Optimizer
model = TrajectoryTransformer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop with progress display
for epoch in range(num_epochs):
    total_loss = 0
    # Use tqdm to show progress
    dataloader_tqdm = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    for sequences, labels in dataloader_tqdm:
        optimizer.zero_grad()
        output = model(sequences)  # Shape: (batch_size, seq_len, num_classes)
        output = output.view(-1, output.shape[-1])  # Flatten for CrossEntropy
        labels = labels.view(-1)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # Update tqdm description with current loss
        dataloader_tqdm.set_postfix(loss=loss.item())

    # Average loss for the epoch
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
model_file_path = "trained_trajectory_transformer.pth"
torch.save(model.state_dict(), model_file_path)
print(f"Model saved to {model_file_path}")

# Load the model
def load_model(model_file_path):
    model = TrajectoryTransformer()
    model.load_state_dict(torch.load(model_file_path))
    model.eval()
    return model

# Prediction function
def predict_key_points(model, trajectory_data):
    predictions = []
    for traj_data in trajectory_data:
        sequence = [[point['x'], point['y'], point['z'], point['frame_index']] for point in traj_data["trajectory"]]
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(sequence_tensor)
            predicted_labels = torch.argmax(output, dim=-1).squeeze().tolist()

        trajectory_predictions = []
        for i, label in enumerate(predicted_labels):
            if label != 0:  # Only interested in key points
                frame_index = traj_data["trajectory"][i]["frame_index"]
                trajectory_predictions.append((frame_index, label))

        predictions.append(trajectory_predictions)

    return predictions

# Load data and model
with open('0020-trajectories_for_training_3d_truth.json', 'r') as f:
    data = json.load(f)

model_file_path = "trained_trajectory_transformer.pth"
loaded_model = load_model(model_file_path)

# Make predictions
predictions = predict_key_points(loaded_model, data)
print("Predictions with frame indices and labels:", predictions)
