import os
import cv2
import dlib
import csv
from os import listdir
from os.path import isdir, join

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Filter videos with a single detectable face throughout the video')
    parser.add_argument('--input_folder', type=str, help='Path to the folder containing subfolders with video files')
    parser.add_argument('--output_csv', type=str, help='Path to output CSV file', default='output.csv')
    parser.add_argument('--skip_frame', type=int, help='Number of frames to skip while processing the video', default=10)
    args = parser.parse_args()
    return args

# Function to check if a single face is present throughout the video
def check_single_face_in_video(video_path, detector, skip_frame):
    video_stream = cv2.VideoCapture(video_path)
    single_face_present = True
    frame_count = 0

    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            break

        # Skip frames to speed up processing
        if frame_count % skip_frame != 0:
            frame_count += 1
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame)

        # Check if exactly one face is detected
        if len(faces) != 1:
            single_face_present = False
            break

        frame_count += 1

    video_stream.release()
    return single_face_present

# Main function to process all videos in subfolders
def main():
    args = parse_args()
    input_folder = args.input_folder
    output_csv = args.output_csv
    skip_frame = args.skip_frame

    detector = dlib.get_frontal_face_detector()

    total_videos = 0
    single_face_videos = 0

    # Create CSV file to store results
    with open(output_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Video File'])

        # Iterate over all subfolders
        for subfolder in listdir(input_folder):
            subfolder_path = join(input_folder, subfolder)
            if isdir(subfolder_path):
                # Iterate over all video files in the subfolder
                for video_file in listdir(subfolder_path):
                    video_path = join(subfolder_path, video_file)
                    if video_file.endswith(('.mp4', '.avi', '.mov')):
                        total_videos += 1
                        # Check if the video has a single detectable face throughout
                        if check_single_face_in_video(video_path, detector, skip_frame):
                            single_face_videos += 1
                            csv_writer.writerow([video_path])
                            print(f"Single face detected throughout: {video_path}")
                        else:
                            print(f"Multiple/no faces detected in: {video_path}")

    # Print summary
    print(f"Total videos analyzed: {total_videos}")
    print(f"Total videos with a single face throughout: {single_face_videos}")

if __name__ == "__main__":
    main()
