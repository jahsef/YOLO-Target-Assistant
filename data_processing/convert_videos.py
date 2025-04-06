import cv2
import os
import imageio.v3 as iio
import time
from multiprocessing import Pool
import multiprocessing

class FPSTracker:
    def __init__(self, update_interval=5.0):
        self.frame_count = 0
        self.last_update = time.perf_counter()
        self.update_interval = update_interval

    def update(self):
        self.frame_count += 1
        current_time = time.perf_counter()
        
        if current_time - self.last_update >= self.update_interval:
            fps = self.frame_count / (current_time - self.last_update)
            print(f'FPS: {fps:.2f}')
            self.frame_count = 0
            self.last_update = current_time
            
def crop_frame(image,crop_dim = (640,640)):
    original_dim = (image.shape[1],image.shape[0])#wxh
    offset_width = (original_dim[0] - crop_dim[0])//2
    offset_height= (original_dim[1] - crop_dim[1])//2

    crop_region = [offset_height, original_dim[1]-offset_height, offset_width, original_dim[0]-offset_width]
    image = image[crop_region[0]:crop_region[1], crop_region[2]:crop_region[3]]#h x w
    return image

def process_frame(args):
    video_path, frame_count, output_folder, base_name, crop_dim = args
    # Each process opens its own VideoCapture instance.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Move to the required frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    success, frame = cap.read()
    cap.release()
    if not success:
        print(f"Warning: Could not read frame {frame_count} from {video_path}")
        return

    # Crop the frame
    frame = crop_frame(frame, crop_dim=crop_dim)
    # Use a unique filename using the frame number (avoids collisions)
    frame_filename = os.path.join(output_folder, f"{base_name}_{frame_count}.png")
    cv2.imwrite(frame_filename, frame)
    return frame_filename

def extract_frames(video_path, output_folder, interval, num_workers=4, crop_dim=(640,640)):
    start = time.perf_counter()
    os.makedirs(output_folder, exist_ok=True)

    # Open video to get total frame count
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    total_extracted = total_frames // interval
    print(f"Total frames: {total_frames}, total extracted: {total_extracted}, expected time@5fps: {(total_extracted / 5):.2f} s")
    
    base_name = os.path.basename(os.path.splitext(video_path)[0])
    tasks = []
    # Create a task for every frame we want to extract
    for frame_count in range(0, total_frames, interval):
        tasks.append((video_path, frame_count, output_folder, base_name, crop_dim))
    
    # Use a pool to run frame extraction in parallel
    with Pool(processes=num_workers) as pool:
        # This will block until all processes are finished
        results = pool.map(process_frame, tasks)
    
    print(f"Frame extraction completed: {video_path}")
    time_taken = time.perf_counter() - start
    fps = (total_extracted / time_taken)
    print(f'actual time:{(time_taken):.2f}, fps: {fps:.2f}')
    




def cum(video):
    print(f"Processing video: {video}")
    video_path = os.path.join(cwd, r'data_processing\videos_to_convert', video)
    output_folder = os.path.join(cwd, r'data_processing\converted_videos')
    extract_frames(
        video_path=video_path,
        output_folder=output_folder,
        interval=128 
    )
if __name__ == '__main__':
    cwd = os.getcwd()#train\video_processing\
    # print(f"Processing video: {video}")

    list_videos = os.listdir(os.path.join(cwd, r'data_processing\videos_to_convert'))
    output_folder = os.path.join(cwd, r'data_processing\converted_videos')
    for video_path in list_videos:
        # print(f'starting process for {video}')
        # Process(target = cum, args = (video,)).start()
        # cum(video)
        video_path = os.path.join(cwd, r'data_processing\videos_to_convert', video_path)

        extract_frames(
            video_path=video_path,
            output_folder=output_folder,
            interval=84,
            num_workers=6 
        )