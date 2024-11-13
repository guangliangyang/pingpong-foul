# 定义字典
zidian_x = '<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,q,w,e,r,t,y,u,i,o,p,a,s,d,f,g,h,j,k,l,z,x,c,v,b,n,m'
zidian_x = {word: i for i, word in enumerate(zidian_x.split(','))}

zidian_xr = [k for k, v in zidian_x.items()]

zidian_y = {k.upper(): v for k, v in zidian_x.items()}

zidian_yr = [k for k, v in zidian_y.items()]

import random

import numpy as np
import torch

def get_data():
    # Define the word set and their probabilities
    words = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'q', 'w', 'e', 'r',
        't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k',
        'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm'
    ]
    p = np.array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
    ])
    p = p / p.sum()

    # Randomly select n words
    n = random.randint(30, 48)
    x = np.random.choice(words, size=n, replace=True, p=p).tolist()

    # Reverse x
    x = x[::-1]

    # Segment x based on type (numeric or character) and extract the first item in each segment
    segments = []
    current_segment = [x[0]]

    for i in range(1, len(x)):
        if (x[i].isdigit() and current_segment[0].isdigit()) or (x[i].isalpha() and current_segment[0].isalpha()):
            current_segment.append(x[i])
        else:
            segments.append(current_segment[0].upper())  # Take the first item of the current segment
            current_segment = [x[i]]

    segments.append(current_segment[0].upper())  # Add the first item of the last segment

    # Target y is the list of first items in each segment
    y = segments

    # Add start and end tokens
    x = ['<SOS>'] + x + ['<EOS>']
    y = ['<SOS>'] + y + ['<EOS>']

    # Pad x and y to fixed lengths
    x = x + ['<PAD>'] * 50
    y = y + ['<PAD>'] * 51
    x = x[:50]
    y = y[:51]

    # Encode x and y as indices
    x = [zidian_x[i] for i in x]
    y = [zidian_y[i] for i in y]

    # Convert to tensors
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)

    return x, y




# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

    def __len__(self):
        return 1000000

    def __getitem__(self, i):
        return get_data()


# 数据加载器
loader = torch.utils.data.DataLoader(dataset=Dataset(),
                                     batch_size=64,
                                     drop_last=True,
                                     shuffle=True,
                                     collate_fn=None)
import torch
import torch.nn as nn
import torch.optim as optim
import os
import warnings

# 屏蔽特定的警告
warnings.filterwarnings("ignore", category=UserWarning, message="The PyTorch API of nested tensors is in prototype stage")
warnings.filterwarnings("ignore", category=UserWarning, message="Torch was not compiled with flash attention")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model parameters
input_dim = len(zidian_x)
output_dim = len(zidian_y)
emb_dim = 32
nhead = 4
num_layers = 3
max_len = 51


# Position Embedding
class PositionEmbedding(nn.Module):
    def __init__(self, emb_dim, max_len):
        super(PositionEmbedding, self).__init__()
        self.position_embedding = nn.Embedding(max_len, emb_dim)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        return self.position_embedding(positions)


# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, nhead, num_layers, max_len):
        super(TransformerModel, self).__init__()

        self.embedding_x = nn.Embedding(input_dim, emb_dim)
        self.embedding_y = nn.Embedding(output_dim, emb_dim)
        self.position_embedding_x = PositionEmbedding(emb_dim, max_len)
        self.position_embedding_y = PositionEmbedding(emb_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(emb_dim, output_dim)

    def create_masks(self, src, tgt):
        src_pad_mask = (src == zidian_x['<PAD>']).to(device)
        tgt_pad_mask = (tgt == zidian_y['<PAD>']).to(device)

        tgt_len = tgt.size(1)
        tgt_causal_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device) == 1, diagonal=1)
        return src_pad_mask, tgt_pad_mask, tgt_causal_mask

    def forward(self, src, tgt):
        src_emb = self.embedding_x(src) + self.position_embedding_x(src)
        tgt_emb = self.embedding_y(tgt) + self.position_embedding_y(tgt)

        src_pad_mask, tgt_pad_mask, tgt_causal_mask = self.create_masks(src, tgt)

        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_causal_mask, tgt_key_padding_mask=tgt_pad_mask,
                              memory_key_padding_mask=src_pad_mask)

        output = self.fc_out(output)
        return output


# Path for saving/loading the model
model_path = "transformer_model.pth"

# Initialize the model
model = TransformerModel(input_dim, output_dim, emb_dim, nhead, num_layers, max_len).to(device)

# Check if the model file exists
if os.path.exists(model_path):
    # Load the model if it exists
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded from", model_path)
else:
    # Train the model if the file does not exist
    criterion = nn.CrossEntropyLoss(ignore_index=zidian_y['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training the model...")
    num_epochs = 5  # Train for a single epoch as requested
    for epoch in range(num_epochs):
        model.train()
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            tgt_input = y[:, :-1]
            tgt_output = y[:, 1:]

            output = model(x, tgt_input)
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(loader)}], Loss: {loss.item():.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print("Model trained and saved to", model_path)


# Prediction function
def predict_sequence(model, x, max_len=50):
    x = x.to(device)
    tgt_input = torch.LongTensor([[zidian_y['<SOS>']]]).to(device)

    for _ in range(max_len - 1):
        with torch.no_grad():
            output = model(x, tgt_input)

        next_token = output[:, -1, :].argmax(1).unsqueeze(1)
        tgt_input = torch.cat([tgt_input, next_token], dim=1)

        if next_token.item() == zidian_y['<EOS>']:
            break

    output_sequence = [zidian_yr[idx] for idx in tgt_input.squeeze(0).tolist()]
    return output_sequence


# Assuming all previous code has been defined (model, predict_sequence, etc.)

# Load a single batch of data
for i, (x, y) in enumerate(loader):
    break  # Load only one batch

# Print the encoded input, target, and predicted output for the first two sequences
for i in range(6):
    print(f"Sample {i + 1}:")

    # Convert input sequence indices to characters
    input_seq = ''.join([zidian_xr[idx] for idx in x[i].tolist()])
    print("Input x:", input_seq)

    # Convert target sequence indices to characters
    target_seq = ''.join([zidian_yr[idx] for idx in y[i].tolist()])

    # Predict sequence using the loaded or trained model
    predicted_seq = ''.join(predict_sequence(model, x[i].unsqueeze(0)))
    print("Target  y:", target_seq)
    print("Predicted:", predicted_seq)

    print("-" * 50)  # Separator for readability

