    
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Load the TensorRT engine
def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

# Allocate memory for inputs/outputs
def allocate_buffers(engine, context):
    inputs = {}
    outputs = {}
    for binding in engine:
        # Get tensor shape
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            shape = context.get_tensor_shape(binding)  # Get input shape from context
        else:
            shape = engine.get_tensor_shape(binding)  # Get output shape from engine
        
        # Calculate size
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        
        # Allocate device memory
        device_mem = cuda.mem_alloc(size * dtype.itemsize)
        
        # Assign to inputs or outputs
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs[binding] = device_mem
        else:
            outputs[binding] = device_mem

    return inputs, outputs

    
if __name__ == "__main__":
    engine_file = "runs//train//train_run//weights//best_stripped.engine"  # Path to your .engine file
    engine = load_engine(engine_file)
    context = engine.create_execution_context()

    # Set input shape dynamically
    input_shape = (1, 3, 1440, 1440)  # Example shape (adjust as needed)
    context.set_input_shape("input_tensor_name", input_shape)

    # Allocate buffers
    inputs, outputs = allocate_buffers(engine, context)

    # Bind tensors to memory addresses
    for name, ptr in inputs.items():
        context.set_tensor_address(name, ptr)
    for name, ptr in outputs.items():
        context.set_tensor_address(name, ptr)

    # Prepare input data
    input_data = np.random.random(input_shape).astype(np.float16)  # Example input
    cuda.memcpy_htod(inputs["input_tensor_name"], input_data.ravel())  # Copy input data to GPU

    # Perform inference
    context.execute_async_v2()  # Run inference asynchronously

    # Retrieve output data
    output_shape = context.get_tensor_shape("output_tensor_name")
    output_size = trt.volume(output_shape)
    output_dtype = trt.nptype(engine.get_tensor_dtype("output_tensor_name"))
    host_output = np.empty(output_size, dtype=output_dtype)
    cuda.memcpy_dtoh(host_output, outputs["output_tensor_name"])  # Copy output data to CPU

    print("Output data:", host_output)