import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time




class TensorRT_Engine:
    def __init__(self,engine_file_path, imgsz, verbose = False):
        self.TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
        self.engine = self.load_engine(engine_file_path)
        self.context = self.engine.create_execution_context()
        input_tensor_name = self.engine.get_tensor_name(0)#assuming first tensor is input
        self.imgsz = imgsz
        self.context.set_input_shape(input_tensor_name, (1, 3, *imgsz))
        self.stream = cuda.Stream()
        self.input_addresses, self.output_addresses = [], []
        self.output_shape = None
        self._malloc_iotensors()
        
    def load_engine(self,engine_file_path):
        assert os.path.exists(engine_file_path)
        print(f"Reading engine from {engine_file_path}")
        with open(engine_file_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
        
    def inference(self,input_data):
        #should add direct cupy integration
        #output as cupy arr?
        #should probably handle cupy -> cpu outside of this class
        cuda.memcpy_htod_async(self.input_addresses[0], input_data, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.output_addresses[0], self.stream)
        self.stream.synchronize()
        return output
    
    def _malloc_iotensors(self):
        #some of this stuff works for more than 1 i/o tensor
        #some dont tho lol
        for tensor_name in self.engine:
            # print(tensor_name)
            tensor_shape = self.context.get_tensor_shape(tensor_name)
            dtype = self.engine.get_tensor_dtype(tensor_name)
            size = trt.volume(tensor_shape) * dtype.itemsize
            allocation = cuda.mem_alloc(size)
            self.context.set_tensor_address(tensor_name, allocation)
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.input_addresses.append(allocation)
            else:
                self.output_addresses.append(allocation)
                self.output_shape = tensor_shape
                
if __name__ == '__main__':
    cwd = os.getcwd()
    base_dir = "runs/train/EFPS_4000img_11s_retrain_1440p_batch6_epoch200/weights"
    engine_name = "896x1440_stripped.engine"
    model_path = os.path.join(cwd, base_dir, engine_name)
    imgsz = (896,1440)
    model = TensorRT_Engine(model_path,imgsz)
    # time.sleep(10)

    img_path = os.path.join(cwd, 'train/datasets/EFPS_4000img/images/train/frame_1012(1013).jpg')
    img = cv2.imread(img_path)
    img = cv2.resize(img, imgsz[::-1]) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0).astype(np.float32)#add batch dim
    img /= 255.0 
    img = np.ascontiguousarray(img)

    output = model.inference(img)
    print("Output shape:", output.shape)
    batch_idx = 0
    detection_idx = 0
    print(output[batch_idx][detection_idx])

    # warmpsudps
    for _ in range(64):
        model.inference(img)

    for i in range(64):
        start = time.perf_counter()
        for _ in range(1000):
            model.inference(img)
        print(f"Inference/s: {1000 / (time.perf_counter() - start):.2f}")


