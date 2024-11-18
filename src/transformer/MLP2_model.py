import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
import numpy as np
import random

# Load trajectory data from JSON file
with open('0020-trajectories_for_training_3d_truth.json', 'r') as f:
    data = json.load(f)

# Function to add noise to a trajectory
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

# Function to generate augmented data
def generate_augmented_data(data, num_samples=1000, noise_std=0.005):
    augmented_data = []
    for _ in range(num_samples):
        # Randomly select a trajectory from the original data
        original_sample = random.choice(data)
        original_trajectory = original_sample['trajectory']

        # Add noise to the trajectory
        noisy_trajectory = add_noise(
            [[point['x'], point['y'], point['z'], point['frame_index']] for point in original_trajectory],
            noise_std=noise_std
        )

        # Prepare the augmented sample
        noisy_sample = {
            'trajectory': [{'x': pt[0], 'y': pt[1], 'z': pt[2], 'frame_index': pt[3]} for pt in noisy_trajectory],
            'key_points': original_sample['key_points']  # Copy key points directly
        }
        augmented_data.append(noisy_sample)
    return augmented_data

# Generate 10,000 augmented samples
augmented_data = generate_augmented_data(data, num_samples=1000)

# Combine original and augmented data
augmented_dataset = data + augmented_data
print(f"Original dataset size: {len(data)}, Augmented dataset size: {len(augmented_dataset)}")

# Dataset class that processes data with frame_index included
class TrajectoryDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.features = []
        self.labels = []
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

            # Combine two previous, current, and two next points as features
            for i in range(2, len(sequence) - 2):
                combined_features = (
                    sequence[i - 2][:3] +  # Two points before
                    sequence[i - 1][:3] +  # One point before
                    sequence[i][:3] +      # Current point
                    sequence[i + 1][:3] +  # One point after
                    sequence[i + 2][:3]    # Two points after
                )
                self.features.append(combined_features)
                self.labels.append(label_sequence[i])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# Define an MLP model
class TrajectoryMLP(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=64, num_classes=4):
        super(TrajectoryMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Instantiate Dataset
dataset = TrajectoryDataset(augmented_dataset)

# Split into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, Loss, and Optimizer
model = TrajectoryMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop with Best Model Saving
num_epochs = 500
best_val_loss = float('inf')
best_model_path = "best_trajectory_mlp.pth"
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for features, labels in train_loader:
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for features, labels in val_loader:
            output = model(features)
            loss = criterion(output, labels)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with validation loss: {best_val_loss}")

# Plotting the loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

print(f"Best model saved at: {best_model_path}")


# Load the best model
def load_model(model_file_path):
    model = TrajectoryMLP()
    model.load_state_dict(torch.load(model_file_path))
    model.eval()
    return model


# Helper function to extract true labels
def get_true_labels(data):
    true_labels = []
    for traj_data in data:
        key_points = traj_data["key_points"]
        sequence_length = len(traj_data["trajectory"])
        label_sequence = [0] * sequence_length  # Default label is 0 (non-key point)

        # Annotate key points
        for key, key_point in key_points.items():
            frame_index = key_point['frame_index']
            index_in_sequence = next(
                (i for i, pt in enumerate(traj_data["trajectory"]) if pt['frame_index'] == frame_index), None
            )
            if index_in_sequence is not None:
                if key == 'throw_point':
                    label_sequence[index_in_sequence] = 1
                elif key == 'highest_point':
                    label_sequence[index_in_sequence] = 2
                elif key == 'hit_point':
                    label_sequence[index_in_sequence] = 3

        true_labels.extend(label_sequence)
    return true_labels


# Helper function to extract predicted labels
def get_predicted_labels(model, data):
    predicted_labels = []
    for traj_data in data:
        sequence = [[point['x'], point['y'], point['z'], point['frame_index']] for point in traj_data["trajectory"]]

        # Prepare combined features for prediction
        for i in range(2, len(sequence) - 2):
            combined_features = (
                sequence[i - 2][:3] +  # Two points before
                sequence[i - 1][:3] +  # One point before
                sequence[i][:3] +      # Current point
                sequence[i + 1][:3] +  # One point after
                sequence[i + 2][:3]    # Two points after
            )
            combined_tensor = torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = model(combined_tensor)
                label = torch.argmax(output, dim=-1).item()
                predicted_labels.append(label)
    return predicted_labels


# Load data and best model
loaded_model = load_model(best_model_path)

# Get true and predicted labels
true_labels = get_true_labels(augmented_dataset)
predicted_labels = get_predicted_labels(loaded_model, augmented_dataset)

# Ensure labels are aligned in size
min_len = min(len(true_labels), len(predicted_labels))
true_labels = np.array(true_labels[:min_len])
predicted_labels = np.array(predicted_labels[:min_len])

# Calculate F1 score
print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=['Non-key', 'Throw', 'Highest', 'Hit']))

f1 = f1_score(true_labels, predicted_labels, average='weighted')
print(f"Weighted F1 Score: {f1}")
