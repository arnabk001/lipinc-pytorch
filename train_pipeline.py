import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
import time
import os
import cv2
import multiprocessing
from model import LIPINCModel, CustomLoss
from utils import get_color_structure_frames
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# Define custom dataset for loading npy files
class LipSyncNpyDataset(Dataset):
    def __init__(self, csv_file):
        # Load the CSV file
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        combined_path = row['combined_save_path']
        residue_path = row['residue_save_path']
        label = row['label']

        # Load the npy files
        combined_frames = np.load(combined_path)
        residue_frames = np.load(residue_path)

        # Convert to tensors
        combined_frames = torch.tensor(combined_frames, dtype=torch.float32).permute(3, 0, 1, 2)
        residue_frames = torch.tensor(residue_frames, dtype=torch.float32).permute(3, 0, 1, 2)
        label = torch.tensor(label, dtype=torch.long)

        return combined_frames, residue_frames, label

# Custom collate function to handle batch loading
def collate_fn(batch):
    combined_frames_batch = torch.stack([item[0] for item in batch])
    residue_frames_batch = torch.stack([item[1] for item in batch])
    labels_batch = torch.stack([item[2] for item in batch])
    return combined_frames_batch, residue_frames_batch, labels_batch

# Training function
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
    
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for combined_frames, residue_frames, labels in dataloader:
                combined_frames = combined_frames.to(device)
                residue_frames = residue_frames.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(combined_frames, residue_frames)
                loss = criterion(combined_frames, labels, outputs)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Calculate statistics
                running_loss += loss.item() * combined_frames.size(0)
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels.data)
                total_predictions += labels.size(0)

                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct_predictions.double() / total_predictions
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())
        epoch_end = time.time()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Time: {epoch_end - epoch_start:.2f}s")

    print('Training complete')
    return train_losses, train_accuracies


# Validation function
def validate_model(model, dataloader, criterion, device, output_folder):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for combined_frames, residue_frames, labels in tqdm(dataloader, desc="Validating"):
            combined_frames = combined_frames.to(device)
            residue_frames = residue_frames.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(combined_frames, residue_frames)
            loss = loss = criterion(combined_frames, labels, outputs)

            # Calculate statistics
            running_loss += loss.item() * combined_frames.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_predictions += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions.double() / total_predictions

    # Calculate additional metrics
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    print(f"Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    timestamp = datetime.now().strftime("%m%d_%H%M")
    plt.savefig(os.path.join(output_folder, f"confusion_matrix_{timestamp}.png"))
    plt.close()

    return loss, accuracy

# Main function to set up dataset, dataloader, model, loss, and optimizer
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    csv_file = "/home/ubuntu/arnab/lipinc-pytorch/FakeAVCeleb_processed/FakeAVCeleb_processed_data.csv"
    dataset = LipSyncNpyDataset(csv_file)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataset and dataloader
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, 
                            num_workers=multiprocessing.cpu_count(), 
                            collate_fn=collate_fn,
                            pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, 
                            num_workers=multiprocessing.cpu_count(), 
                            collate_fn=collate_fn,
                            pin_memory=True)

    # Initialize model, loss function, and optimizer
    model = LIPINCModel().to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = CustomLoss(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001, eps=0.1)

    # Create output folder for saving results
    output_folder = "./outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Train the model
    train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, device, 
                                                 num_epochs=30)
    
    # Validate the model
    val_loss, val_accuracy = validate_model(model, val_loader, criterion, device, output_folder)

    # Save the trained model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(output_folder, f"lipinc_model_{timestamp}.pth")
    torch.save(model.state_dict(), model_save_path)

    # Plot training and validation metrics
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(train_losses) + 1), [val_loss] * len(train_losses), label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"loss_plot_{timestamp}.png"))
    plt.close()

    plt.figure()
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(train_accuracies) + 1), [val_accuracy] * len(train_accuracies), label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"accuracy_plot_{timestamp}.png"))
    plt.close()

if __name__ == "__main__":
    main()
