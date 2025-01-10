import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# Define custom dataset for loading npy files
class LipSyncNpyDataset(Dataset):
    def __init__(self, csv_files):
        # Load and concatenate CSV files
        dataframes = []
        for csv_file in csv_files:
            # Read CSV file into a DataFrame
            df = pd.read_csv(csv_file)

            # Check for necessary columns
            required_columns = ['combined_save_path', 'residue_save_path', 'label']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV file {csv_file} is missing one or more required columns: {required_columns}")
            
            # Append the valid DataFrame to the list
            dataframes.append(df)

        # Concatenate all DataFrames and store in self.df
        if len(dataframes) == 0:
            raise ValueError("No valid CSV files provided")
        
        self.df = pd.concat(dataframes, ignore_index=True)
                # Calculate and print dataset statistics
        self.total_samples = len(self.df)
        self.real_samples = len(self.df[self.df['label'] == 0])
        self.fake_samples = len(self.df[self.df['label'] == 1])
        print(f"Total samples: {self.total_samples}, Real samples (0): {self.real_samples}, Fake samples (1): {self.fake_samples}")

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
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, output_folder="outputs"):
    model.train()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
    
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for combined_frames, residue_frames, labels in train_loader:
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

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions.double() / total_predictions
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())
        epoch_end = time.time()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Time: {epoch_end - epoch_start:.2f}s")

        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item())

        ## Step the scheduler
        # scheduler.step()

    print('Training complete')
    return train_losses, train_accuracies, val_losses, val_accuracies



# Validation function
def validate_model(model, dataloader, criterion, device):
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
            loss = criterion(combined_frames, labels, outputs)

            # Calculate statistics
            running_loss += loss.item() * combined_frames.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_predictions += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions.double() / total_predictions

    print(f"Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    return loss, accuracy

def test_model(model, dataloader, device, output_folder):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for combined_frames, residue_frames, labels in tqdm(dataloader, desc="Testing"):
            combined_frames = combined_frames.to(device)
            residue_frames = residue_frames.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(combined_frames, residue_frames)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_predictions += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = correct_predictions.double() / total_predictions
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

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

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_folder, f"roc_curve_{timestamp}.png"))
    plt.close()

# Main function to set up dataset, dataloader, model, loss, and optimizer
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    csv_files = ["/home/ubuntu/arnab/lipinc-pytorch/FakeAVCeleb_processed/FakeAVCeleb_processed_data.csv",
                 "/home/ubuntu/arnab/lipinc-pytorch/FakeAVCeleb_processed/FakeAVCeleb_real_processed_data.csv"]
    dataset = LipSyncNpyDataset(csv_files)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Further split training set into training and validation sets (90% for training, 10% for validation)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create dataset and dataloader
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, 
                            num_workers=multiprocessing.cpu_count(), 
                            collate_fn=collate_fn,
                            pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True, 
                            num_workers=multiprocessing.cpu_count(), 
                            collate_fn=collate_fn,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, 
                             num_workers=multiprocessing.cpu_count(), 
                             collate_fn=collate_fn, 
                             pin_memory=True)

    # Initialize model, loss function, and optimizer
    model = LIPINCModel().to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = CustomLoss(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001, eps=0.1)

    # Cosine Annealing with warm restarts
    # Add the scheduler for Cosine Annealing with Warm Restarts
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-5)

    # Create output folder for saving results
    output_folder = "./outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Train the model
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(model, 
                                                                             train_loader, 
                                                                             val_loader, 
                                                                             criterion,
                                                                             optimizer, 
                                                                            #  scheduler, 
                                                                             device, 
                                                                             num_epochs=30, 
                                                                             output_folder=output_folder)
    
    # Save the trained model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(output_folder, f"lipinc_model_{timestamp}.pth")
    torch.save(model.state_dict(), model_save_path)

    # Plot training and validation metrics
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"loss_plot_{timestamp}.png"))
    plt.close()

    plt.figure()
    plt.plot(range(1, len(train_accuracies) + 1), [acc.cpu().item() if torch.is_tensor(acc) else acc for acc in train_accuracies], label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), [acc.cpu().item() if torch.is_tensor(acc) else acc for acc in val_accuracies], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"accuracy_plot_{timestamp}.png"))
    plt.close()

     # Test the model
    test_model(model, test_loader, device, output_folder)

if __name__ == "__main__":
    main()
