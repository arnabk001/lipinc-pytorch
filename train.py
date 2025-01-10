import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from skimage.metrics import structural_similarity
from model import LIPINCModel

# Custom Dataset class to load video frames
def load_data(input_paths, labels):
    data = []
    for path, label in zip(input_paths, labels):
        # Load combined_frames and residue_frames as input features
        combined_frames = np.load(f"{path}_combined.npy")
        residue_frames = np.load(f"{path}_residue.npy")
        data.append((combined_frames, residue_frames, label))
    return data

class LipSyncDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        combined_frames, residue_frames, label = self.data[idx]
        combined_frames = torch.tensor(combined_frames, dtype=torch.float32).permute(3, 0, 1, 2)  # (C, D, H, W)
        residue_frames = torch.tensor(residue_frames, dtype=torch.float32).permute(3, 0, 1, 2)  # (C, D, H, W)
        label = torch.tensor(label, dtype=torch.long)
        return combined_frames, residue_frames, label

# Consistency loss definition
def similarity(a, b):
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    score, _ = structural_similarity(a_np, b_np, multichannel=True, full=True)
    return score

def total_loss(frame_input, model, y_true, y_pred, criterion):
    # Cross-entropy loss
    cce = criterion(y_pred, y_true)
    
    # Consistency loss
    z = model.cnn_frame(frame_input)
    tot = 0
    batch_size = z.size(0)
    for i in range(batch_size):
        a = z[i]
        for j in range(batch_size):
            b = z[j]
            sim = similarity(a, b)
            tot += sim
    
    avg_sim = tot / (batch_size * batch_size)
    bce = nn.BCELoss()(torch.tensor([avg_sim]), y_true.float())
    
    return cce + 5 * bce

# Training function
def train(model, dataloader, optimizer, device, num_epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (combined_frames, residue_frames, labels) in enumerate(dataloader):
            combined_frames, residue_frames, labels = combined_frames.to(device), residue_frames.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(combined_frames, residue_frames)
            loss = total_loss(combined_frames, model, labels, outputs, criterion)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:    # Print every 10 batches
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.3f}")
                running_loss = 0.0

    print("Training finished")

if __name__ == "__main__":
    # Define device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    input_paths = ["video1", "video2", "video3"]  # Replace with actual paths
    labels = [0, 1, 0]  # Replace with actual labels
    data = load_data(input_paths, labels)
    dataset = LipSyncDataset(data)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize model
    model = LIPINCModel().to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, eps=0.1)

    # Train the model
    train(model, dataloader, optimizer, device, num_epochs=10)
    
    # Save the model
    torch.save({'model_state_dict': model.state_dict()}, "lipinc_model.pth")
