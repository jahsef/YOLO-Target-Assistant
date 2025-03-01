import os
def strip_metadata(input_file, output_file):
    with open(input_file, "rb") as f:
        # Read the length of the metadata (first 4 bytes)
        meta_length = int.from_bytes(f.read(4), byteorder="little", signed=True)
        
        # Skip the metadata
        f.seek(meta_length, 1)  # Move forward by `meta_length` bytes
        
        # Read the rest of the file (serialized engine)
        engine_data = f.read()
    
    # Write the serialized engine to a new file
    with open(output_file, "wb") as f:
        f.write(engine_data)

# Example usage
cwd = os.getcwd()
engine_path = os.path.join(cwd,"runs//train//train_run//weights//best.engine")
strip_metadata(engine_path, os.path.join(cwd,"runs//train//train_run//weights//best_stripped.engine"))