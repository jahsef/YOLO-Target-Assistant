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

    while frame_count < total_frames:
        # Skip frames to reach the correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        success, frame = cap.read()

        if success:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        # Move to the next frame that should be saved
        frame_count += interval

    cap.release()
    print("Frame extraction completed.")

# Example usage
cwd = os.getcwd()
list_videos = os.listdir(os.path.join(cwd,'videos'))
for video in list_videos:
    poop = video[:video.index('.')]

    os.makedirs(os.path.join(cwd,'output',poop), exist_ok=True)
    extract_frames(
        video_path=os.path.join(cwd, 'videos', video),
        output_folder=os.path.join(cwd, "output",poop),
        interval=240  # For example, save every 300th frame
    )
