import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit#auto init cuda mem context
import numpy as np
import cupy as cp




class TensorRT_Engine:
    def __init__(self,engine_file_path, imgsz, verbose = False):
        self.TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine(engine_file_path)
        self.context = self.engine.create_execution_context()
        self.input_tensor_name = self.engine.get_tensor_name(0)#assuming first tensor is input
        self.output_tensor_name = self.engine.get_tensor_name(1)
        self.imgsz = imgsz
        self.context.set_input_shape(self.input_tensor_name, (1, 3, *imgsz))
        self.stream = cp.cuda.Stream()
        self.input_address, self.output_address = 0,0
        self.input_shape = None
        self.output_shape = None
        #a lot of this stuff might be useless
        self.input_dtype = self.engine.get_tensor_dtype(self.input_tensor_name)
        self.output_dtype = self.engine.get_tensor_dtype(self.input_tensor_name)
        self.input_itemsize = self.input_dtype.itemsize
        self.output_itemsize = self.output_dtype.itemsize
        
        self._malloc_iotensors()
        # self.output_data_size =trt.volume(self.output_shape) * self.output_itemsize
        # self.output_memory = cp.cuda.UnownedMemory(ptr = int(self.output_address),size = self.output_data_size,owner = self)
        # self.output_ptr = cp.cuda.MemoryPointer(self.output_memory, offset = 0)
        

        
        
        
    def _load_engine(self,engine_file_path):
        assert os.path.exists(engine_file_path)
        print(f"Reading engine from {engine_file_path}")
        with open(engine_file_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def inference_cp(self, input_data: cp.ndarray) -> cp.ndarray:
        # Set pointers (no changes needed if using pre-allocated buffers)
        self.context.set_tensor_address(self.input_tensor_name, input_data.data.ptr)
        self.context.set_tensor_address(self.output_tensor_name, self.output_address)

        # Execute inference
        self.context.execute_async_v3(self.stream.ptr)
        
        # Synchronize the stream AND the device
        self.stream.synchronize()
        # Return the output buffer (no need to create a new array)
        
        return self.output_buffer
    
    
    
    def inference_np(self,input_data: np.ndarray) -> np.ndarray:
        #should add direct cupy integration
        #output as cupy arr?
        #should probably handle cupy -> cpu outside of this class
        cuda.memcpy_htod_async(self.input_address, input_data, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.output_address ,self.stream)
        self.stream.synchronize()
        return output
    
    def _malloc_iotensors(self):
        #use cupy instead of pycuda for malloc
        input_shape = self.engine.get_tensor_shape(self.input_tensor_name)
        output_shape = self.engine.get_tensor_shape(self.output_tensor_name)
        input_dtype = self.engine.get_tensor_dtype(self.input_tensor_name)
        output_dtype = self.engine.get_tensor_dtype(self.output_tensor_name)
        #nptype for dynamic data size
        
        self.input_buffer = cp.empty(input_shape, dtype=trt.nptype(input_dtype))
        self.input_address = self.input_buffer.data.ptr 
        self.output_buffer = cp.empty(output_shape, dtype=trt.nptype(output_dtype))
        self.output_address = self.output_buffer.data.ptr

        self.context.set_tensor_address(self.input_tensor_name, self.input_address)
        self.context.set_tensor_address(self.output_tensor_name, self.output_address)
                
    def get_cupy_arr_ptrs(self):
        input_data_size = trt.volume(self.input_shape) * self.input_itemsize
        output_data_size =trt.volume(self.output_shape) * self.output_itemsize
        #class pycuda.driver.DeviceAllocation
        #pycuda memptr object can be cast to int
        input_memory = cp.cuda.UnownedMemory(ptr = int(self.input_address),size = input_data_size,owner = self)
        input_ptr = cp.cuda.MemoryPointer(input_memory, offset = 0)
        output_memory = cp.cuda.UnownedMemory(ptr = int(self.output_address),size = output_data_size,owner = self)
        output_ptr = cp.cuda.MemoryPointer(output_memory, offset = 0)
        
        return (input_ptr,output_ptr)
    
if __name__ == '__main__':
    import cv2
    import time
    cwd = os.getcwd()
    base_dir = "runs/train/EFPS_4000img_11s_retrain_1440p_batch6_epoch200/weights"
    engine_name = "896x1440_stripped.engine"
    model_path = os.path.join(cwd, base_dir, engine_name)
    imgsz = (896,1440)
    model = TensorRT_Engine(model_path,imgsz)



  

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
    # output = model.inference_np(np_img)
    output = model.inference_cp(cp_img)
    print("Output shape:", output.shape)
    batch_idx = 0
    detection_idx = 0
    print(output[batch_idx][detection_idx])

    # warmpsudps
    for _ in range(1):
        cp_img = cp.asarray(np_img) 
        model.inference_cp(cp_img)

    # for i in range(64):
    #     start = time.perf_counter()
    #     for _ in range(1000):
    #         model.inference_cp(cp_img)
    #     print(f"Inference/s: {1000 / (time.perf_counter() - start):.2f}")


