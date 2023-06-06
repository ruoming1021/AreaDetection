from ultralytics import YOLO

model = YOLO('/home/vip/Documents/PyQt_Demo/Cource/weights/yolov8m.pt')
model.train(data='/home/vip/Documents/PyQt_Demo/Cource/course.yaml',epochs=50)
# model.export(format='engine', half=True, device='0')