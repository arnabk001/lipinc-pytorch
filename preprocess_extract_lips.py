import os
import cv2
import torch
import numpy as np
import pandas as pd
import csv
from utils import get_color_structure_frames

# Function to preprocess a video and save the features to a .npy file
def preprocess_and_save(video_path, output_folder, label, output_csv, csv_writer):
    try:
        # Get color and structure frames
        length_error, _, combined_frames, residue_frames, _, _ = get_color_structure_frames(n_frames=5, path=video_path)
        
        if length_error:
            raise ValueError("Video too short")
        if combined_frames is None or residue_frames is None or combined_frames.size == 0 or residue_frames.size == 0:
            raise ValueError("No valid frames found")
        
        # Prepare output folder
        label_folder = "fake" if label == 1 else "real"
        save_path = os.path.join(output_folder, label_folder)
        os.makedirs(save_path, exist_ok=True)
        
        # Save combined and residue frames as a .npy file
        combined_filename = os.path.basename(video_path).replace('.mp4', '_combined.npy')
        residue_filename = os.path.basename(video_path).replace('.mp4', '_residue.npy')
        
        combined_save_path = os.path.join(save_path, combined_filename)
        residue_save_path = os.path.join(save_path, residue_filename)
        
        np.save(combined_save_path, combined_frames)
        np.save(residue_save_path, residue_frames)
        
        # Write to output CSV
        csv_writer.writerow([video_path, combined_save_path, residue_save_path, label])
        
        print(f"Saved processed data for {video_path} at {combined_save_path} and {residue_save_path}")
    except ValueError as e:
        print(f"Skipping video {video_path} due to error: {e}")

# Main function to read the CSV and preprocess the dataset
def main():
    # Set paths
    csv_file = "/home/ubuntu/arnab/lipinc-pytorch/FakeAVCeleb_v1.2/meta_data.csv"
    folder_location = "/home/ubuntu/arnab/lipinc-pytorch/"
    output_folder = "FakeAVCeleb_processed"
    output_csv = "/home/ubuntu/arnab/lipinc-pytorch/FakeAVCeleb_processed/FakeAVCeleb_processed_data.csv"
    
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Open the output CSV file for writing
    with open(output_csv, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['original_file_path', 'combined_save_path', 'residue_save_path', 'label'])
        
        # Iterate through each row of the CSV
        for index, row in df.iterrows():
            method = row['method']
            path = row['path']
            filename = row['filename']
            
            # Construct full video path
            video_path = os.path.join(path, filename)
            
            # Determine label based on method column
            if 'wav2lip' in method.lower():
                label = 1 
            elif 'real' in method.lower():
                label = 0
            
            # Preprocess and save features
            preprocess_and_save(video_path, output_folder, label, output_csv, csv_writer)

if __name__ == "__main__":
    main()
