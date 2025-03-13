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
    return [[point[0] * scale_factor, point[1] * scale_factor, point[2] * scale_factor, point[3]] for point in sequence]


def add_noise(sequence, noise_std=0.0001):
    return [[point[0] + np.random.normal(0, noise_std),
             point[1] + np.random.normal(0, noise_std),
             point[2] + np.random.normal(0, noise_std),
             point[3]] for point in sequence]


# Adjust frame indices to start from 0
def adjust_frame_indices(data):
    """
    Adjust each trajectory's frame_index to start from 0.

    Args:
        data (list): List of trajectory dictionaries containing "trajectory" and "key_points".

    Returns:
        list: Adjusted data with normalized frame indices.
    """
    adjusted_data = []
    for item in data:
        # Find the starting frame index
        start_index = item["trajectory"][0]["frame_index"]

        # Adjust trajectory frame indices
        for point in item["trajectory"]:
            point["frame_index"] -= start_index

        # Adjust key points' frame indices
        for key_point in item["key_points"].values():
            key_point["frame_index"] -= start_index

        adjusted_data.append(item)
    return adjusted_data


# Load and adjust the JSON file
input_file = '0020-trajectories_for_training_3d_truth.json'
output_file = '0020-trajectories_for_training_3d_truth_adjusted.json'

with open(input_file, 'r') as f:
    data = json.load(f)

# Adjust the frame indices
adjusted_data = adjust_frame_indices(data)

# Save the adjusted data back to a new JSON file
with open(output_file, 'w') as f:
    json.dump(adjusted_data, f, indent=4)

print(f"Adjusted data saved to {output_file}")


# Compute min and max values for normalization
def compute_min_max(json_file):
    min_vals = np.array([float('inf'), float('inf'), float('inf'), float('inf')])
    max_vals = np.array([float('-inf'), float('-inf'), float('-inf'), float('-inf')])

    with open(json_file, 'r') as f:
        data = json.load(f)
        for item in data:
            for point in item["trajectory"]:
                coords = np.array([point["x"], point["y"], point["z"], point["frame_index"]])
                min_vals = np.minimum(min_vals, coords)
                max_vals = np.maximum(max_vals, coords)
            for key_point in item["key_points"].values():
                coords = np.array([key_point["x"], key_point["y"], key_point["z"], key_point["frame_index"]])
                min_vals = np.minimum(min_vals, coords)
                max_vals = np.maximum(max_vals, coords)

    return min_vals, max_vals


# Normalize frame_index
def normalize_frame_index(frame_index, min_frame, max_frame):
    return (frame_index - min_frame) / (max_frame - min_frame + 1e-8)


# Generate sinusoidal positional encoding
def generate_positional_encoding(frame_indices, emb_dim):
    """
    Generate sinusoidal positional encoding for the given frame indices.

    Args:
        frame_indices (torch.Tensor): Frame indices of shape (batch_size, seq_len, 1)
        emb_dim (int): Dimensionality of the embeddings

    Returns:
        torch.Tensor: Positional encoding of shape (batch_size, seq_len, emb_dim)
    """
    position = frame_indices  # Shape: (batch_size, seq_len, 1)
    div_term = torch.exp(torch.arange(0, emb_dim, 2, device=frame_indices.device) * -(np.log(10000.0) / emb_dim))

    # Reshape div_term to match the last dimension of position
    div_term = div_term.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, emb_dim // 2)

    pos_enc = torch.zeros((frame_indices.size(0), frame_indices.size(1), emb_dim), device=frame_indices.device)
    pos_enc[..., 0::2] = torch.sin(position * div_term)  # Sin on even indices
    pos_enc[..., 1::2] = torch.cos(position * div_term)  # Cos on odd indices
    return pos_enc


# Compute min and max values for normalization
min_vals, max_vals = compute_min_max(output_file)


# Dataset class
class TrajectoryDataset(Dataset):
    def __init__(self, json_file, min_vals, max_vals, max_len=500, augment=True, augment_factor=10000):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.max_len = max_len
        self.augment = augment
        self.augment_factor = augment_factor

    def normalize(self, coords):
        return (coords - self.min_vals) / (self.max_vals - self.min_vals + 1e-8)

    def __len__(self):
        return len(self.data) * self.augment_factor

    def __getitem__(self, idx):
        original_idx = idx % len(self.data)
        trajectory_data = self.data[original_idx]["trajectory"]
        key_points_data = self.data[original_idx]["key_points"]

        # Prepare x as the trajectory sequence: [x, y, z, frame_index]
        x = [[
            point["x"], point["y"], point["z"],
            normalize_frame_index(point["frame_index"], self.min_vals[3], self.max_vals[3])
        ] for point in trajectory_data]

        # Apply data augmentation if enabled
        if self.augment:
            x = scale_trajectory(x)
            x = add_noise(x)

        # Normalize x
        x = [self.normalize(np.array(point)).tolist() for point in x]

        # Pad or truncate x to max_len
        x = x[:self.max_len] + [[0, 0, 0, 0]] * (self.max_len - len(x))

        # Prepare and normalize y
        y = [
            self.normalize(np.array([
                key_points_data["throw_point"]["x"],
                key_points_data["throw_point"]["y"],
                key_points_data["throw_point"]["z"],
                normalize_frame_index(key_points_data["throw_point"]["frame_index"], self.min_vals[3], self.max_vals[3])
            ])).tolist(),
            self.normalize(np.array([
                key_points_data["highest_point"]["x"],
                key_points_data["highest_point"]["y"],
                key_points_data["highest_point"]["z"],
                normalize_frame_index(key_points_data["highest_point"]["frame_index"], self.min_vals[3],
                                      self.max_vals[3])
            ])).tolist(),
            self.normalize(np.array([
                key_points_data["hit_point"]["x"],
                key_points_data["hit_point"]["y"],
                key_points_data["hit_point"]["z"],
                normalize_frame_index(key_points_data["hit_point"]["frame_index"], self.min_vals[3], self.max_vals[3])
            ])).tolist()
        ]

        # Pad y to max_len
        y = y + [[0, 0, 0, 0]] * (self.max_len - len(y))

        # Convert to tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y


# Load the dataset
dataset = TrajectoryDataset('0020-trajectories_for_training_3d_truth.json', min_vals, max_vals)
loader = DataLoader(dataset, batch_size=64, shuffle=True)


# Transformer model
class TransformerModel(nn.Module):
    def __init__(self, emb_dim, nhead, num_layers, max_len):
        super(TransformerModel, self).__init__()
        self.coordinate_projection = nn.Linear(3, emb_dim)  # Project x, y, z to embedding dimension
        self.fc_out = nn.Linear(emb_dim, 4)  # Project back to 4D space
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def create_masks(self, src, tgt):
        src_pad_mask = (src == 0).all(dim=-1).to(device)
        tgt_pad_mask = (tgt == 0).all(dim=-1).to(device)
        tgt_causal_mask = torch.triu(torch.ones((tgt.size(1), tgt.size(1)), device=device) == 1, diagonal=1)
        return src_pad_mask, tgt_pad_mask, tgt_causal_mask

    def forward(self, src, tgt):
        coords = src[..., :3]  # Extract x, y, z
        frame_indices = src[..., 3:4]  # Extract frame_index
        pos_enc = generate_positional_encoding(frame_indices, emb_dim=self.coordinate_projection.out_features)
        src_emb = self.coordinate_projection(coords) + pos_enc
        tgt_emb = self.coordinate_projection(tgt[..., :3]) + generate_positional_encoding(
            tgt[..., 3:4], emb_dim=self.coordinate_projection.out_features
        )
        src_pad_mask, tgt_pad_mask, tgt_causal_mask = self.create_masks(src, tgt)
        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_causal_mask, tgt_key_padding_mask=tgt_pad_mask,
                              memory_key_padding_mask=src_pad_mask)
        return self.fc_out(output)


# Initialize the model
model = TransformerModel(emb_dim=32, nhead=4, num_layers=3, max_len=500).to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        tgt_input = y[:, :-1]
        tgt_output = y[:, 1:]
        output = model(x, tgt_input)
        loss = criterion(output, tgt_output)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")


# Prediction function
def predict_sequence(model, x, min_vals, max_vals, max_len=3):
    model.eval()
    x = x.to(device)
    tgt_input = torch.zeros((1, max_len, 4), device=device)
    for i in range(max_len):
        with torch.no_grad():
            output = model(x, tgt_input)
        tgt_input[:, i, :] = output[:, i, :].squeeze(1)
    denormalized_output = tgt_input.squeeze(0).cpu().numpy()
    denormalized_output[:, :3] = denormalized_output[:, :3] * (max_vals[:3] - min_vals[:3]) + min_vals[:3]
    return denormalized_output


# Testing the prediction
for x, y in loader:
    x_sample = x[0].unsqueeze(0)
    predicted_sequence = predict_sequence(model, x_sample, min_vals, max_vals)
    print("Predicted Key Points:", predicted_sequence)
    print("Actual Key Points:", y[0].cpu().numpy())
    break
