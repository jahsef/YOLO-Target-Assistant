import cv2
import os

def extract_frames(video_path, output_folder, interval):
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    print(f"Total frames in video: {total_frames}")

    # Start frame extraction
    saved_frame_count = 0  # To count the number of frames saved
    frame_count = 0
    
    counter = 1
    while frame_count < total_frames:
        # Skip frames to reach the correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        success, frame = cap.read()

        if success:
            # Handle duplicate filenames for images
            base_name = f'frame_{saved_frame_count}'
            ext = '.jpg'
            frame_filename = os.path.join(output_folder, f"{base_name}{ext}")
            
            # Check for duplicate filenames
            
            while os.path.exists(frame_filename):
                frame_filename = os.path.join(output_folder, f"{base_name}({counter}){ext}")
                counter += 1

            # Save the frame
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        # Move to the next frame that should be saved
        frame_count += interval

    cap.release()
    print("Frame extraction completed.")

# Example usage
cwd = os.getcwd()
list_videos = os.listdir(os.path.join(cwd, r'train\video_processing\videos_to_convert'))

for video in list_videos:
    print(f"Processing video: {video}")
    
    # Construct paths
    video_path = os.path.join(cwd, r'train\video_processing\videos_to_convert', video)
    output_folder = os.path.join(cwd, r'train\video_processing\converted_videos')

    # Extract frames from the video
    extract_frames(
        video_path=video_path,
        output_folder=output_folder,
        interval=36  # For example, save every 5th frame
    )