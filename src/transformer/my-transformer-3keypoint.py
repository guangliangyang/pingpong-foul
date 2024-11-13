import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data augmentation functions
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


# Function to compute min-max values
def compute_min_max(json_file):
    min_vals = np.array([float('inf'), float('inf'), float('inf')])
    max_vals = np.array([float('-inf'), float('-inf'), float('-inf')])

    with open(json_file, 'r') as f:
        data = json.load(f)
        for item in data:
            # Extract trajectory
            for point in item["trajectory"]:
                coords = np.array([point["x"], point["y"], point["z"]])
                min_vals = np.minimum(min_vals, coords)
                max_vals = np.maximum(max_vals, coords)

            # Extract key points
            for key_point in item["key_points"].values():
                coords = np.array([key_point["x"], key_point["y"], key_point["z"]])
                min_vals = np.minimum(min_vals, coords)
                max_vals = np.maximum(max_vals, coords)

    return min_vals, max_vals


# Compute min and max values for normalization
min_vals, max_vals = compute_min_max('0020-trajectories_for_training_3d_truth.json')


# Dataset class to load JSON data with 1000x data augmentation and normalization
class TrajectoryDataset(Dataset):
    def __init__(self, json_file, min_vals, max_vals, max_len=50, augment=True, augment_factor=10000):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.max_len = max_len
        self.augment = augment
        self.augment_factor = augment_factor

    def __len__(self):
        return len(self.data) * self.augment_factor

    def normalize(self, coords):
        # Normalize to [0, 1] range
        return (coords - self.min_vals) / (self.max_vals - self.min_vals)

    def __getitem__(self, idx):
        original_idx = idx % len(self.data)
        trajectory_data = self.data[original_idx]["trajectory"]
        key_points_data = self.data[original_idx]["key_points"]

        # Prepare x as the trajectory sequence: [x, y, z, frame_index]
        x = [[point["x"], point["y"], point["z"], point["frame_index"]] for point in trajectory_data]

        # Apply data augmentation if enabled
        if self.augment:
            x = scale_trajectory(x)
            x = add_noise(x)

        # Normalize x
        x = [self.normalize(np.array(point[:3])).tolist() + [point[3]] for point in x]

        # Pad or truncate x to max_len
        x = x[:self.max_len] + [[0, 0, 0, 0]] * (self.max_len - len(x))

        # Prepare and normalize y
        y = [
            self.normalize(np.array([
                key_points_data["throw_point"]["x"],
                key_points_data["throw_point"]["y"],
                key_points_data["throw_point"]["z"]
            ])).tolist() + [key_points_data["throw_point"]["frame_index"]],
            self.normalize(np.array([
                key_points_data["highest_point"]["x"],
                key_points_data["highest_point"]["y"],
                key_points_data["highest_point"]["z"]
            ])).tolist() + [key_points_data["highest_point"]["frame_index"]],
            self.normalize(np.array([
                key_points_data["hit_point"]["x"],
                key_points_data["hit_point"]["y"],
                key_points_data["hit_point"]["z"]
            ])).tolist() + [key_points_data["hit_point"]["frame_index"]]
        ]

        # Convert to tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # print("x=",x)
        # print("y=",y)
        return x, y


# Load the dataset with 1000x data augmentation and normalization
dataset = TrajectoryDataset('0020-trajectories_for_training_3d_truth.json', min_vals, max_vals, augment=True,
                            augment_factor=10000)
loader = DataLoader(dataset, batch_size=64, shuffle=True)


# Position Embedding class
class PositionEmbedding(nn.Module):
    def __init__(self, emb_dim, max_len):
        super(PositionEmbedding, self).__init__()
        self.position_embedding = nn.Embedding(max_len, emb_dim)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        return self.position_embedding(positions)


# Transformer model
class TransformerModel(nn.Module):
    def __init__(self, emb_dim, nhead, num_layers, max_len):
        super(TransformerModel, self).__init__()

        # Project 3D coordinates + frame index to embedding dimension
        self.coordinate_projection = nn.Linear(4, emb_dim)
        self.position_embedding = PositionEmbedding(emb_dim, max_len)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(emb_dim, 4)  # Project back to 4D space

    def create_masks(self, src, tgt):
        src_pad_mask = (src == 0).all(dim=-1).to(device)
        tgt_pad_mask = (tgt == 0).all(dim=-1).to(device)

        tgt_len = tgt.size(1)
        tgt_causal_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device) == 1, diagonal=1)
        return src_pad_mask, tgt_pad_mask, tgt_causal_mask

    def forward(self, src, tgt):
        # Project the 3D trajectory and frame index
        src_emb = self.coordinate_projection(src) + self.position_embedding(src)
        tgt_emb = self.coordinate_projection(tgt) + self.position_embedding(tgt)

        # Create masks
        src_pad_mask, tgt_pad_mask, tgt_causal_mask = self.create_masks(src, tgt)

        # Encoding
        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)

        # Decoding
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_causal_mask, tgt_key_padding_mask=tgt_pad_mask,
                              memory_key_padding_mask=src_pad_mask)

        output = self.fc_out(output)
        return output


# Initialize the model
model = TransformerModel(emb_dim=32, nhead=4, num_layers=3, max_len=50).to(device)

# Loss and optimizer
criterion = nn.MSELoss()  # MSELoss for regression on 3D coordinates
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Shift target sequence for input to decoder
        tgt_input = y[:, :-1]
        tgt_output = y[:, 1:]

        # Forward pass
        output = model(x, tgt_input)

        # Compute loss
        loss = criterion(output, tgt_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")


# Prediction function
def predict_sequence(model, x, min_vals, max_vals, max_len=3):
    model.eval()
    x = x.to(device)
    tgt_input = torch.zeros((1, max_len, 4), device=device)  # Initialize target input with zeros

    for i in range(max_len):
        with torch.no_grad():
            output = model(x, tgt_input)
        next_point = output[:, i, :].unsqueeze(1)  # Take next predicted point
        tgt_input[:, i, :] = next_point.squeeze(1)

    # De-normalize only the first 3 dimensions (coordinates), leave frame index as is
    denormalized_output = tgt_input.squeeze(0).cpu().numpy()
    denormalized_output[:, :3] = denormalized_output[:, :3] * (max_vals - min_vals) + min_vals

    return denormalized_output  # Return as numpy array


# Testing the prediction
for x, y in loader:
    x_sample = x[0].unsqueeze(0)  # Take one sequence for prediction
    predicted_sequence = predict_sequence(model, x_sample, min_vals, max_vals)
    print("Predicted Key Points:", predicted_sequence)
    print("Actual Key Points:", y[0].cpu().numpy())
    break
