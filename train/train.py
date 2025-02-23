from ultralytics import YOLO
import os
import pickle
def main():
        
    # Define the paths to the dataset and YAML file
    cwd = os.getcwd()
    data_yaml_path = os.path.join(cwd,'train//split_dataset//data.yaml')  # Update with your data.yaml path
    weights_path = os.path.join(cwd,'yolo12s.pt')  # You can use a pretrained model like yolov8n.pt
    epochs = 100  # Adjust the number of epochs as needed
    batch = 5  # Adjust based on your GPU memory
    imgsz = 1440  # Image size for training

    # Load the YOLOv8 model (pretrained or custom)
    model = YOLO("yolo11s.pt")  # Use the YOLOv8 pre-trained weights or your own

    # Train the model
    model.train(
        data=data_yaml_path,  # Path to your dataset configuration file
        epochs=epochs,  # Number of training epochs
        batch=batch,  # Batch size for training
        imgsz=imgsz,  # Image size for training
        project='runs/train',  # Directory to save the training results
        name='train_run',  # Directory name for the run
        exist_ok=True  # Overwrite the existing directory if necessary
    )

    # new_metrics = model.metrics
    # keys_me_want = ['keys','maps','results_dict','speed']
    # only_metrics_i_want = {}
    # for key in keys_me_want:
    #     only_metrics_i_want[key] = new_metrics[key]
    
    # curr_metrics = {}
    # curr_metrics_path = os.path.join(cwd,"train//curr_metrics.pkl")
    # if os.path.exists(curr_metrics_path):
    #     with open(file = curr_metrics_path, mode = 'rb') as f:
    #         curr_metrics = pickle.loads(f)
    # else:
    #     print('curr metrics file dont exist lol')
    # print(f'Current Metrics:\n {curr_metrics}\n')
    # print(f'New Metrics:\n {only_metrics_i_want}\n')
    #fitness
    #maps
    #results_dict 
    #speed
    
    # while True:
    #     print('Save new model? (Y/N)')
    #     user_input = input().lower()
    #     if user_input == "y":
    #         model.save()
    #         with open(curr_metrics_path,'rb') as f:#saving new metrics to file
    #             pickle.dump(only_metrics_i_want)
    #         break
    #     elif user_input == "n":
    #         print('Model not saved!')
    #         break
    #     else:
    #         print('kys you didnt type the thing in right')
    model.save()
    

print("Training complete!")

if __name__ == "__main__":
    main()