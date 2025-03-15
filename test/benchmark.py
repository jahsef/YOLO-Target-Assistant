import ultralytics.utils.benchmarks
import os

cwd = os.getcwd()
model_path = os.path.join(cwd,"runs/train/\EFPS_3000image_realtrain_1440x1440_100epoch_batch6_11s/weights/best.pt")

if __name__ == '__main__':
    ultralytics.utils.benchmarks.benchmark(
        model = model_path,
        data = os.path.join(os.getcwd(),"train/datasets/EFPS_4000img/data.yaml"),
        imgsz = 1440,
        device = 'cuda',
        format = ''
        )

