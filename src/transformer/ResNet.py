import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
import cv2

# 使用 GPU 的设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 视频路径和标注文件路径
video_path = 'C:\\workspace\\datasets\\foul-video\\c1.mp4'
csv_path = '0020-trajectories_for_training_video_truth.csv'

# 读取标注数据
labels_df = pd.read_csv(csv_path)

# 创建帧图像目录
frame_dir = 'extracted_frames'
os.makedirs(frame_dir, exist_ok=True)

# 提取视频帧并存储到 frame_dir
def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1
    cap.release()

extract_frames(video_path, frame_dir)

# 自定义数据集类
class KeyFrameDataset(Dataset):
    def __init__(self, frame_dir, labels_df, transform=None):
        self.frame_dir = frame_dir
        self.labels_df = labels_df
        self.transform = transform
        self.frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
        self.labels = self.generate_labels()

    def generate_labels(self):
        # 初始化所有帧为负样本（非关键帧）
        labels = [0] * len(self.frames)
        # 根据CSV标注关键帧
        for _, row in self.labels_df.iterrows():
            labels[int(row['throw_point_index'])] = 1
            labels[int(row['hit_point_index'])] = 1
        return labels

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.frame_dir, self.frames[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 定义图像预处理和数据增强
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 加载数据集
dataset = KeyFrameDataset(frame_dir, labels_df, transform=transform)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定义ResNet模型
class KeyFrameResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(KeyFrameResNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 初始化模型、损失函数、优化器，并将模型移动到 GPU
model = KeyFrameResNet(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据移到 GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)  # 将数据移到 GPU
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_keyframe_resnet.pth')
            print("Saved best model with validation loss:", best_val_loss)

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25)

# 加载最佳模型进行测试
def load_best_model(model_path):
    model = KeyFrameResNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

best_model = load_best_model('best_keyframe_resnet.pth')

# 测试函数：检测长视频中的关键帧
def predict_key_frames(model, frame_dir):
    frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    key_frames = []
    for idx, frame_path in enumerate(frame_paths):
        image = Image.open(frame_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
        with torch.no_grad():
            output = model(image)
            _, pred = torch.max(output, 1)
            if pred.item() == 1:
                key_frames.append(idx)
    return key_frames

# 使用最佳模型预测关键帧
key_frames = predict_key_frames(best_model, frame_dir)
print("Predicted Key Frames:", key_frames)
