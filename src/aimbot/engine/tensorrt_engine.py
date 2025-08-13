import os
import tensorrt as trt
# import pycuda.autoinit#auto init cuda mem context
#init context in here bad, use default context
import numpy as np
import cupy as cp
# import pycuda.cuda as cuda



class TensorRT_Engine:
    def __init__(self,engine_file_path:str, conf_threshold:float,verbose:bool = False):
        
        self.TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine(engine_file_path)
        self.context = self.engine.create_execution_context()

        #now explicitly defines input instead of assuming first idx is input tensor
        self.input_tensor_name = None
        self.output_tensor_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_tensor_name = name
            else:
                self.output_tensor_name = name
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
        
        
        

        
    def _load_engine(self, engine_file_path: str) -> trt.ICudaEngine:
        print(f'Reading engine file from: {engine_file_path}')
        with open(engine_file_path, "rb") as f:
            runtime = trt.Runtime(self.TRT_LOGGER)
            return runtime.deserialize_cuda_engine(f.read())

    def inference_cp(self, src: cp.ndarray) -> cp.ndarray:
        """
        src needs to be a contiguous array in bchw
        
        also needs batch size 1, dynamic batch size is slow and this is for real time inference
        
        returns parsed results without batch dim and without low conf things
        """

        if not (src.flags.c_contiguous and src.shape == self.input_shape):
            src = cp.ascontiguousarray(src.astype(self.input_dtype))

        self.context.set_tensor_address(self.input_tensor_name, src.data.ptr)
        self.context.execute_async_v3(self.stream.ptr)

        self.stream.synchronize()

        return self._parse_cp_results()

    def _parse_cp_results(self):
        #x1,y1,x2,y2,conf,cls_id
        removed_batch_dim = self.output_buffer.reshape(self.output_shape[1:])
        #removing batch dim retains contiguity if input is contiguous (already is after preprocessing)
        filtered_results = removed_batch_dim[removed_batch_dim[:, 4] > self.conf_threshold]
        # print(filtered_results.shape)
        return filtered_results
    
    # def _parse_cp_results(self):
    #     # Skip reshape if possible
    #     # output = self.output_buffer.reshape(-1, 6)  # Only if needed
        
    #     # GPU-accelerated thresholding
    #     valid_ids = cp.where(self.output_buffer[:, 4] > self.conf_threshold)[0]
    #     return self.output_buffer[valid_ids]
    
    def _alloc_output_tensor(self):
        self.output_buffer = cp.empty(self.output_shape, dtype=trt.nptype(self.output_dtype))
        self.output_ptr = self.output_buffer.data.ptr
        self.context.set_tensor_address(self.output_tensor_name, self.output_ptr)
        
    def __del__(self):
        # Explicit cleanup for CUDA resources
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'engine'):
            del self.engine
        if hasattr(self, 'stream'):
            self.stream.synchronize()
            del self.stream      
                   
if __name__ == '__main__':
    import torch
    import cv2
    import time
    torch.cuda.empty_cache()
    cwd = os.getcwd()
    base_dir = "models/EFPS_4000img_11s/weights"
    engine_name = "320x320_stripped.engine"
    model_path = os.path.join(cwd, base_dir, engine_name)
    imgsz = (320,320)
    model = TensorRT_Engine(model_path,conf_threshold= 0, verbose = True)

    img_path = os.path.join(cwd, 'datasets/EFPS_4000img_640x640/images/train/frame_13(15).jpg')
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
    for _ in range(128):
        cp_img = cp.asarray(np_img) 
        # print(cp_img.data.ptr)
        output = model.inference_cp(cp_img)
        # print("Output shape:", output.shape)
        print(output)
        # output = None
        # cp_img = None

    iterations = 16
    inferences = 2560
    
    fart = time.perf_counter()
    for i in range(iterations):
        start = time.perf_counter()
        for _ in range(inferences):
            results = model.inference_cp(cp_img)
        print(f"Inference/s: {inferences / (time.perf_counter() - start):.2f}")
    print(f'avg inference / s: {iterations*inferences / (time.perf_counter() - fart)}')

