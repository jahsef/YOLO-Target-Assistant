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
base_dir = "runs/train/EFPS_4000img_11n_1440p_batch11_epoch100/weights"
engine_name = "320x320.engine"
stripped_engine_name = engine_name[:engine_name.index('.')] + '_stripped.engine'
engine_path = os.path.join(cwd,base_dir,engine_name)
strip_metadata(engine_path, os.path.join(cwd,base_dir,stripped_engine_name))