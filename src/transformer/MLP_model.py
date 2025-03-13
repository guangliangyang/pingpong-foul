import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Load trajectory data from JSON file
with open('0020-trajectories_for_training_3d_truth.json', 'r') as f:
    data = json.load(f)


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

            # Combine previous, current, and next points as features
            for i in range(1, len(sequence) - 1):
                prev_point = sequence[i - 1]
                curr_point = sequence[i]
                next_point = sequence[i + 1]
                combined_features = prev_point[:3] + curr_point[:3] + next_point[:3]  # Only (x, y, z) of each point
                self.features.append(combined_features)
                self.labels.append(label_sequence[i])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# Define an MLP model
class TrajectoryMLP(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, num_classes=4):
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


# Instantiate Dataset and DataLoader
dataset = TrajectoryDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, Loss and Optimizer
model = TrajectoryMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 1000
losses = []
for epoch in range(num_epochs):
    total_loss = 0
    for features, labels in dataloader:
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

# Plotting the loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.show()

# Save the trained model
model_file_path = "trained_trajectory_mlp.pth"
torch.save(model.state_dict(), model_file_path)
print(f"Model saved to {model_file_path}")


# Load the model
def load_model(model_file_path):
    model = TrajectoryMLP()
    model.load_state_dict(torch.load(model_file_path))
    model.eval()
    return model


# Prediction function
def predict_key_points(model, trajectory_data):
    predictions = []
    for traj_data in trajectory_data:
        sequence = [[point['x'], point['y'], point['z'], point['frame_index']] for point in traj_data["trajectory"]]

        # Prepare combined features for prediction
        trajectory_predictions = []
        for i in range(1, len(sequence) - 1):
            prev_point = sequence[i - 1]
            curr_point = sequence[i]
            next_point = sequence[i + 1]
            combined_features = prev_point[:3] + curr_point[:3] + next_point[:3]
            combined_tensor = torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = model(combined_tensor)
                label = torch.argmax(output, dim=-1).item()

            if label != 0:  # Only interested in key points
                frame_index = curr_point[3]
                trajectory_predictions.append((frame_index, label))

        predictions.append(trajectory_predictions)

    return predictions


# Load data and model
with open('0020-trajectories_for_training_3d_truth.json', 'r') as f:
    data = json.load(f)

loaded_model = load_model(model_file_path)

# Make predictions
predictions = predict_key_points(loaded_model, data)
print("Predictions with frame indices and labels:", predictions)
