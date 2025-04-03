import cv2
import os
import imageio.v3 as iio
import time

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

def extract_frames(video_path, output_folder, interval):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)#should probably use imageio for this but have to install more stuff
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    total_extracted = total_frames // interval
    print(f"Total frames: {total_frames}, total extracted: {total_extracted}, expected time@1.5fps: {(total_extracted / 1.5):.2f} s")

    frame_count = 0
    
    counter = 0
    fps_tracker = FPSTracker()
    while frame_count < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        success, frame = cap.read()
        if success:
            base_name = os.path.basename(os.path.splitext(video_path)[0])#os lib poop
            ext = '.png'
            frame_filename = os.path.join(output_folder, f"{base_name}{ext}") 
            while os.path.exists(frame_filename):
                frame_filename = os.path.join(output_folder, f"{base_name}({counter}){ext}")
                counter += 1
            frame = crop_frame(frame,crop_dim=(640,640))
            cv2.imwrite(frame_filename, frame)
            # iio.imwrite(frame_filename, frame, extension=".png")#imageio requires a bgr conversion since opencv bgr
        frame_count += interval
        fps_tracker.update()
        
    cap.release()
    print(f"Frame extraction completed: {video_path}")

cwd = os.getcwd()#train\video_processing\
list_videos = os.listdir(os.path.join(cwd, r'data_processing\videos_to_convert'))

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
    from multiprocessing import Process
    for video in list_videos:
        # print(f'starting process for {video}')
        Process(target = cum, args = (video,)).start()