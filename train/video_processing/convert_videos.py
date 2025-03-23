import cv2
import os
from threading import Thread

def extract_frames(video_path, output_folder, interval):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    print(f"Total frames in video: {total_frames}")

    saved_frame_count = 0  # To count the number of frames saved
    frame_count = 0
    
    counter = 1
    while frame_count < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        success, frame = cap.read()
        if success:
            base_name = os.path.basename(video_path) #f'frame_{saved_frame_count}'
            ext = '.jpg'
            frame_filename = os.path.join(output_folder, f"{base_name}{ext}") 
            while os.path.exists(frame_filename):
                frame_filename = os.path.join(output_folder, f"{base_name}({counter}){ext}")
                counter += 1
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        frame_count += interval
    cap.release()
    print("Frame extraction completed.")
    
# start_i = i
# step = num_processes*interval
# def fart(i,interval,cap,video_path,output_folder,total_frames):
#     print(i*interval, total_frames, interval)
#     saved_frame_count = 0  # To count the number of frames saved
#     frame_count = 0
#     counter = 1
#     for j in range(i*interval,total_frames,interval):
        
#         cap.set(cv2.CAP_PROP_POS_FRAMES, j)
#         success, frame = cap.read()
#         if success:
#             base_name = os.path.basename(video_path) #f'frame_{saved_frame_count}'
#             ext = '.jpg'
#             frame_filename = os.path.join(output_folder, f"{base_name}{ext}") 
#             while os.path.exists(frame_filename):
#                 frame_filename = os.path.join(output_folder, f"{base_name}({counter}){ext}")
#                 counter += 1
#             cv2.imwrite(frame_filename, frame)
#             saved_frame_count += 1
#         frame_count += interval
# def extract_frames(video_path, output_folder, interval):
#     num_workers = 8
#     os.makedirs(output_folder, exist_ok=True)
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Unable to open video file {video_path}")
#         return
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
#     print(f"Total frames in video: {total_frames}")

#     thread_list = []
#     for i in range(num_workers):
#         thread_list.append(Thread(target = fart, args = (i,interval,cap,video_path,output_folder,total_frames,)))
#     for thread in thread_list:
#         thread.start()
#         #[i*interval::num_workers*interval]
#     # print(type(thread_list))
#     while True:
#         if len(thread_list) == 0:
#             break
#         for thread in thread_list:
#             if not thread.is_alive():
#                 thread_list.remove(thread)

#     cap.release()
#     print("Frame extraction completed.")



cwd = os.getcwd()#train\video_processing\
list_videos = os.listdir(os.path.join(cwd, r'train\video_processing\videos_to_convert'))

def cum(video):
    print(f"Processing video: {video}")
    video_path = os.path.join(cwd, r'train\video_processing\videos_to_convert', video)
    output_folder = os.path.join(cwd, r'train\video_processing\converted_videos')

    extract_frames(
        video_path=video_path,
        output_folder=output_folder,
        interval=90 
    )

if __name__ == '__main__':
    from multiprocessing import Process
    # for video in list_videos:
    #     cum(video)
    
    for video in list_videos:
        print(f'starting process for {video}')
        Process(target = cum, args = (video,)).start()