from multiprocessing import Event, Process
import time
import numpy as np

def screen_cap(screenshot_ready, inference_ready):
        inference_ready.wait(0.3)
        
        print('"writing image"')
        time.sleep(np.random.normal(loc=0.6, scale=0.1))
        screenshot_ready.set()

def inference(screenshot_ready, inference_ready):
    while True:
        # print('waiting until screenshot_ready')
        screenshot_ready.wait()
        inference_ready.clear()
        
        print('inferencing')
        screenshot_ready.clear()
        time.sleep(np.random.normal(loc=0.8, scale=0.2))
        inference_ready.set()

if __name__ == '__main__':
    screenshot_ready = Event()
    inference_ready = Event()
    inference_ready.set()
    
    sc_thread = Process(target=screen_cap, args=(screenshot_ready, inference_ready), daemon=True)
    inf_thread = Process(target=inference, args=(screenshot_ready, inference_ready), daemon=True)

    sc_thread.start()
    inf_thread.start()

    sc_thread.join()
    inf_thread.join()
