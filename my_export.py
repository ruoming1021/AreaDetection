from ultralytics import YOLO

model = YOLO('/home/vip/Documents/Cource_Demo/runs/detect/train/weights/best.pt')

model.export(format='engine', half=True, device='0')