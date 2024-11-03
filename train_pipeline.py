import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import cv2
from model import LIPINCModel
from utils import get_color_structure_frames
import time

# Define custom dataset for video data
class LipSyncDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=5):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        # length_error, _, combined_frames, residue_frames, _, _ = get_color_structure_frames(self.num_frames, video_path)
        
        try:
            length_error, _, combined_frames, residue_frames, _, _ = get_color_structure_frames(self.num_frames, video_path)
            
            if length_error:
                raise ValueError("Video too short")
            if combined_frames is None or residue_frames is None or combined_frames.size == 0 or residue_frames.size == 0:
                raise ValueError("No valid frames found")
        except ValueError as e:
            print(f"Skipping video {video_path} due to error: {e}")
            return None
        
        # Reshape and convert frames to tensors
        # combined_frames = np.reshape(combined_frames, (1,) + combined_frames.shape)
        # residue_frames = np.reshape(residue_frames, (1,) + residue_frames.shape)

        combined_frames = torch.tensor(combined_frames, dtype=torch.float32).permute(3, 0, 1, 2)
        residue_frames = torch.tensor(residue_frames, dtype=torch.float32).permute(3, 0, 1, 2)

        label = torch.tensor(label, dtype=torch.long)
        
        return combined_frames, residue_frames, label

# Custom collate function to handle None values
def collate_fn(batch):
    batch = [data for data in batch if data is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# Training function
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for data in dataloader:
            if data is None:
                continue
            combined_frames, residue_frames, labels = data
            combined_frames = combined_frames.to(device)
            residue_frames = residue_frames.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(combined_frames, residue_frames)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate statistics
            running_loss += loss.item() * combined_frames.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_predictions += labels.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct_predictions.double() / total_predictions
        epoch_end = time.time()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Time: {epoch_end - epoch_start:.2f}s")

    print('Training complete')

# Main function to set up dataset, dataloader, model, loss, and optimizer
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    video_dir = "/home/ubuntu/arnab/lipinc-pytorch/sample_train_data"
    video_paths = [os.path.join(video_dir, fname) for fname in os.listdir(video_dir) if fname.endswith(".mp4")]
    labels = [0 if "real" in fname else 1 for fname in video_paths]  # Assuming filenames contain "real" or "fake"

    # Create dataset and dataloader
    dataset = LipSyncDataset(video_paths, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Initialize model, loss function, and optimizer
    model = LIPINCModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, eps=0.1)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, device, num_epochs=2)

    # Save the trained model
    torch.save(model.state_dict(), "lipinc_model.pth")

if __name__ == "__main__":
    main()
