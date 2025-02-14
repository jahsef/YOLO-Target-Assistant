import os


cwd = os.getcwd()
base_dir = os.path.join(cwd,'train')
pre_split_dir = os.path.join(base_dir,'pre_split_dataset')
split_dir = os.path.join(base_dir,'split_dataset')


import shutil

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=True)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
clear_directory(split_dir)