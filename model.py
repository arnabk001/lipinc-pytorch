import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from skimage.metrics import structural_similarity

class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, queries, keys, values, mask=None):
        d_k = queries.size(-1)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d_k)
        # print(f"DotProductAttention - scores shape: {scores.shape}")
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(scores, dim=-1)
        # print(f"DotProductAttention - weights shape: {weights.shape}")
        output = torch.matmul(weights, values)
        # print(f"DotProductAttention - output shape: {output.shape}")
        return output

class CNNBranch(nn.Module):
    def __init__(self, input_channels, num_frames):
        super(CNNBranch, self).__init__()
        self.conv3d = nn.Conv3d(input_channels, 8, kernel_size=3, padding=1, stride=1, bias=False)
        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.bn = nn.BatchNorm3d(8)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(8 * (num_frames // 2) * 32 * 72, 8 * 8 * 3)
        self.reshape = (-1, 3, 8, 8)
        
    def forward(self, x):
        x = F.relu(self.conv3d(x))
        # print(f"CNNBranch - after conv3d shape: {x.shape}")
        x = self.pool3d(x)
        # print(f"CNNBranch - after pool3d shape: {x.shape}")
        x = self.bn(x)
        # print(f"CNNBranch - after batchnorm shape: {x.shape}")
        x = self.dropout(x)
        # print(f"CNNBranch - after dropout shape: {x.shape}")
        x = self.flatten(x)
        # print(f"CNNBranch - after flatten shape: {x.shape}")
        x = F.relu(self.dense(x))
        # print(f"CNNBranch - after dense shape: {x.shape}")
        x = x.view(self.reshape)
        # print(f"CNNBranch - after reshape shape: {x.shape}")
        return x

class LIPINCModel(nn.Module):
    def __init__(self):
        super(LIPINCModel, self).__init__()
        self.cnn_frame = CNNBranch(input_channels=3, num_frames=8)
        self.cnn_residue = CNNBranch(input_channels=3, num_frames=7)
        
        self.attention1 = DotProductAttention()
        self.attention2 = DotProductAttention()
        self.attention3 = DotProductAttention()
        
        self.conv2d_1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(64)
        self.conv2d_2 = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.bn2d_2 = nn.BatchNorm2d(64)
        
        self.pool2d_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
        self.conv2d_3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2d_3 = nn.BatchNorm2d(128)
        self.conv2d_4 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.bn2d_4 = nn.BatchNorm2d(128)
        
        self.pool2d_2 = nn.MaxPool2d(kernel_size=2, padding=1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 3, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2)

    def forward(self, frame_input, residue_input):
        cnn_frame = self.cnn_frame(frame_input)
        cnn_residue = self.cnn_residue(residue_input)
        
        # Color branch
        keys = cnn_frame
        values = cnn_frame
        queries = cnn_residue
        conv_color = self.attention1(queries, keys, values)

        # Structure branch
        keys = cnn_residue
        values = cnn_residue
        queries = cnn_frame
        conv_res = self.attention2(queries, keys, values)
        
        # Fusion
        keys = conv_color
        values = conv_color
        queries = conv_res
        conv_fusion = self.attention3(queries, keys, values)
        
        # Concat
        conv = torch.cat([cnn_residue, conv_fusion], dim=1)
        # print(f"LIPINCModel - after concat shape: {conv.shape}")
        
        # MLP
        conv = F.relu(self.conv2d_1(conv))
        # print(f"LIPINCModel - after conv2d_1 shape: {conv.shape}")
        conv = self.bn2d_1(conv)
        # print(f"LIPINCModel - after bn2d_1 shape: {conv.shape}")
        conv = F.relu(self.conv2d_2(conv))
        # print(f"LIPINCModel - after conv2d_2 shape: {conv.shape}")
        conv = self.bn2d_2(conv)
        # print(f"LIPINCModel - after bn2d_2 shape: {conv.shape}")
        
        conv = self.pool2d_1(conv)
        # print(f"LIPINCModel - after pool2d_1 shape: {conv.shape}")
        
        conv = F.relu(self.conv2d_3(conv))
        # print(f"LIPINCModel - after conv2d_3 shape: {conv.shape}")
        conv = self.bn2d_3(conv)
        # print(f"LIPINCModel - after bn2d_3 shape: {conv.shape}")
        conv = F.relu(self.conv2d_4(conv))
        # print(f"LIPINCModel - after conv2d_4 shape: {conv.shape}")
        conv = self.bn2d_4(conv)
        # print(f"LIPINCModel - after bn2d_4 shape: {conv.shape}")
        
        conv = self.pool2d_2(conv)
        # print(f"LIPINCModel - after pool2d_2 shape: {conv.shape}")
        
        conv = self.flatten(conv)
        # print(f"LIPINCModel - after flatten shape: {conv.shape}")

        conv = F.relu(self.fc1(conv))
        # print(f"LIPINCModel - after fc1 shape: {conv.shape}")
        conv = F.relu(self.fc2(conv))
        # print(f"LIPINCModel - after fc2 shape: {conv.shape}")
        out = F.softmax(self.fc3(conv), dim=1)
        # print(f"LIPINCModel - final output shape: {out.shape}")
        
        return out


# Define the consistency loss calculation
def similarity(a, b):
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    score, _ = structural_similarity(a_np, b_np, multichannel=True, full=True)
    return score

def total_loss(frame_input, model, y_true, y_pred):
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
    bce = F.binary_cross_entropy(torch.tensor([avg_sim]), y_true.float())
    
    return cce + 5 * bce

if __name__ == "__main__":
    # Define the model
    model = LIPINCModel()

    # Define a loss function (cross-entropy loss is used here as an example)
    criterion = nn.CrossEntropyLoss()

    # Define an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=0.1)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    # Dummy inputs for testing
    frame_input = torch.randn(1, 3, 8, 64, 144)
    residue_input = torch.randn(1, 3, 7, 64, 144)
    
    # Forward pass
    output = model(frame_input, residue_input)
    print(f"Output shape: {output.shape}")