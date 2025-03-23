
import os


model_path = os.path.join(os.getcwd(),"runs/train/EFPS_4000img_11s_1440p_batch6_epoch200/weights/best_stripped.engine")

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger()
# Load the TensorRT engine
def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

# Load the engine
engine = load_engine(model_path)
context = engine.create_execution_context()
# Set dynamic input shape (if needed)
input_tensor_name = engine.get_tensor_name(0)  # Assumes first binding is input
context.set_input_shape(input_tensor_name, (1, 3, 896, 1440))  # Example shape
context.set_tensor_address(name, ptr)
# Lists to hold device (GPU) buffers
inputs, outputs, bindings = [], [], []
stream = cuda.Stream()

# Loop over engine bindings to allocate memory
for tensor in engine:
    tensor_shape = context.get_tensor_shape(tensor)  # Use context, not engine
    
    dtype = engine.get_tensor_dtype(tensor)
    
    size = trt.volume(tensor_shape) * dtype.itemsize  # Correct size calculation

    allocation = cuda.mem_alloc(size)
    bindings.append(int(allocation))
    
    if engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT:
        inputs.append(allocation)
    else:
        outputs.append(allocation)
    print(f'tensor shape: {tensor_shape}')
    print(f'dtype: {dtype}')
    print(f'size: {size}')
        
# Perform inference
def infer(input_data):
    cuda.memcpy_htod_async(inputs[0], input_data, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    output = cuda.memcpy_dtoh_async(outputs[0], stream)
    stream.synchronize()
    return output
