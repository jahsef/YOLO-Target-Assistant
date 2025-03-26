import os
import tensorrt as trt
# import pycuda.autoinit#auto init cuda mem context
#init context in here bad
import numpy as np
import cupy as cp
import pycuda.cuda as cuda



class TensorRT_Engine:
    def __init__(self,engine_file_path, conf_threshold,verbose = False):
        self.TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine(engine_file_path)
        self.context = self.engine.create_execution_context()
        self.input_tensor_name = self.engine.get_tensor_name(0)#assuming first tensor is input
        self.output_tensor_name = self.engine.get_tensor_name(1)
        # print(input_tensor_size)
        
        # time.sleep(10)
        self.stream = cp.cuda.Stream()
        self.input_ptr, self.output_ptr = 0,0
        self.conf_threshold = conf_threshold
        self.input_shape = self.engine.get_tensor_shape(self.input_tensor_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_tensor_name)
        self.imgsz = self.input_shape[2:]
        print(f'engine imgsz: {self.imgsz}')
        self.input_dtype = self.engine.get_tensor_dtype(self.input_tensor_name)
        self.output_dtype = self.engine.get_tensor_dtype(self.output_tensor_name)
        #could do float
        # cp_input_dtype = cp.dtype(trt.nptype(self.input_dtype))
        # cp_output_dtype = cp.dtype(trt.nptype(self.output_dtype))
        # print(cp_input_dtype)
        # print(cp_output_dtype)
        self._alloc_output_tensor()

        

        
        
        
    def _load_engine(self,engine_file_path):
        assert os.path.exists(engine_file_path)
        print(f"Reading engine from {engine_file_path}")
        with open(engine_file_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def inference_cp(self, input_data: cp.ndarray) -> cp.ndarray:
        if input_data.data.ptr == 0:
            raise ValueError("Input tensor has an invalid address")
        
        if input_data.dtype != cp.float32:
            input_data = input_data.astype(cp.float32)
            
        try:
            self.context.set_tensor_address(self.input_tensor_name, input_data.data.ptr)
            # self.context.set_tensor_address(self.output_tensor_name, self.output_ptr)
        except Exception as e:
            raise RuntimeError(f"Failed to set tensor address: {e}")

        # self.context.set_tensor_address(self.input_tensor_name, input_data.data.ptr)
        self.context.execute_async_v3(self.stream.ptr)

        self.stream.synchronize()
        # Return the output buffer (no need to create a new array)
        # return self.output_buffer
        return self._parse_cp_results()
    #(1,300,6)
    #(1,6)
    def _parse_cp_results(self):
        #x1,y1,x2,y2,conf,cls_id
        removed_batch_dim = self.output_buffer.reshape(self.output_shape[1:])
        filtered_results = removed_batch_dim[removed_batch_dim[:, 4] > self.conf_threshold]
        # print(filtered_results.shape)
        return filtered_results
    
    # def inference_np(self,input_data: np.ndarray) -> np.ndarray:
    #     #this might not even work with cupy streams lol
    #     #should add direct cupy integration
    #     #output as cupy arr?
    #     #should probably handle cupy -> cpu outside of this class
    #     cuda.memcpy_htod_async(self.input_ptr, input_data, self.stream)
    #     self.context.execute_async_v3(self.stream.ptr)
    #     output = np.empty(self.output_shape, dtype=np.float32)
    #     cuda.memcpy_dtoh_async(output, self.output_ptr ,self.stream)
    #     self.stream.synchronize()
    #     return output
    
    def _alloc_output_tensor(self):
        self.output_buffer = cp.empty(self.output_shape, dtype=trt.nptype(self.output_dtype))
        self.output_ptr = self.output_buffer.data.ptr
        self.context.set_tensor_address(self.output_tensor_name, self.output_ptr)
                
if __name__ == '__main__':
    import torch
    import cv2
    import time
    torch.cuda.empty_cache()
    cwd = os.getcwd()
    base_dir = "runs/train/EFPS_4000img_11n_1440p_batch11_epoch100/weights"
    engine_name = "320x320_stripped.engine"
    model_path = os.path.join(cwd, base_dir, engine_name)
    imgsz = (320,320)
    model = TensorRT_Engine(model_path,conf_threshold= 0, verbose = True)

    img_path = os.path.join(cwd, 'train/datasets/EFPS_4000img/images/train/frame_1012(1013).jpg')
    img = cv2.imread(img_path)
    img = cv2.resize(img, imgsz[::-1]) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0).astype(np.float32)#add batch dim
    img /= 255.0 
    np_img = np.ascontiguousarray(img)
    
    cp_img = cp.asarray(np_img) 
    print(f'cp shape: {cp_img.shape}')
    output = model.inference_cp(cp_img)
    print("Output shape:", output.shape)
    batch_idx = 0
    detection_idx = 0
    print(output)

    # warmpsudps
    for _ in range(64):
        cp_img = cp.asarray(np_img) 
        # print(cp_img.data.ptr)
        output = model.inference_cp(cp_img)
    #     print("Output shape:", output.shape)
    #     print(output)
        # output = None
        # cp_img = None

    fart = time.perf_counter()
    for i in range(16):
        start = time.perf_counter()
        for _ in range(1000):
            results = model.inference_cp(cp_img)
        print(f"Inference/s: {1000 / (time.perf_counter() - start):.2f}")
    print(f'avg inference / s: {16*1000 / (time.perf_counter() - fart)}')

