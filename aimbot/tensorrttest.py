import cv2
import time
import os
import numpy as np
import cupy as cp
from utils.tensorrt_engine import TensorRT_Engine
import sys
from pathlib import Path
github_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(os.path.join(github_dir,'BettererCam')))
# can replace with bettercam just no cupy support
import betterercam
# print(betterercam.__file__)

class fart:
    def preprocess(self, frame: cp.ndarray):
        # Ensure input is in HWC format and uint8
        if frame.ndim != 3 or frame.dtype != cp.uint8:
            raise ValueError("Input frame must be in HWC format with dtype=uint8")

        # Convert HWC to BCHW
        bchw = cp.ascontiguousarray(frame.transpose(2, 0, 1)[cp.newaxis, ...])

        # Normalize to [0.0, 1.0] range
        float_frame = bchw.astype(cp.float32, copy=False)
        float_frame /= 255.0

        # Ensure contiguous memory layout
        return cp.ascontiguousarray(float_frame)

        return cp.ascontiguousarray(float_frame)
    def main(self):
        cwd = os.getcwd()
        base_dir = "runs/train/EFPS_4000img_11s_retrain_1440p_batch6_epoch200/weights"
        engine_name = "896x1440_stripped.engine"
        model_path = os.path.join(cwd, base_dir, engine_name)
        imgsz = (896,1440)
        self.model = TensorRT_Engine(model_path,imgsz,conf_threshold= 0)



        self.camera = betterercam.create(region = (69,420,128,690), output_color='BGR',max_buffer_len=2, nvidia_gpu = True)

        img_path = os.path.join(cwd, 'train/datasets/EFPS_4000img/images/train/frame_1012(1013).jpg')
        img = cv2.imread(img_path)
        img = cv2.resize(img, imgsz[::-1]) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        og_img = img.copy()
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0).astype(np.float32)#add batch dim
        img /= 255.0 
        np_img = np.ascontiguousarray(img)
        
        cp_img = cp.asarray(np_img) 
        # print(f'cp shape: {cp_img.shape}')
        # output = model.inference_cp(cp_img)
        # print("Output shape:", output.shape)
        # batch_idx = 0
        # detection_idx = 0
        # print(output)
# Run manual preprocessing
        # manual_preprocessed = cp.asarray(np_img)

        # # Run preprocess function
        # function_preprocessed = self.preprocess(cp.asarray(np_img))

        # # Compare shapes and values
        # print("Manual shape:", manual_preprocessed.shape)
        # print("Function shape:", function_preprocessed.shape)
        # print("Shape match:", manual_preprocessed.shape == function_preprocessed.shape)

        # print("Manual min/max:", manual_preprocessed.min(), manual_preprocessed.max())
        # print("Function min/max:", function_preprocessed.min(), function_preprocessed.max())
        # print("Value match:", cp.allclose(manual_preprocessed, function_preprocessed))
        
        # warmpsudps
        for _ in range(3):
            processed_image = self.preprocess(cp.asarray(og_img))
            print('test file')
            print(processed_image)
            # processed_image = cp.asarray(np_img)
            # print(cp_img.data.ptr)
            output = self.model.inference_cp(processed_image)
            # print("Output shape:", output.shape)
            print(f'output:\n{output}')
            output = None
            processed_image = None
        
        # for i in range(64):
        #     start = time.perf_counter()
        #     for _ in range(1000):
        #         cp_img = cp.asarray(np_img) 
        #         results = model.inference_cp(cp_img)
        #         print(results)
        #         cp_img = None
        #         results = None
        #     print(f"Inference/s: {1000 / (time.perf_counter() - start):.2f}")
            

if __name__ == '__main__':
    fart().main()