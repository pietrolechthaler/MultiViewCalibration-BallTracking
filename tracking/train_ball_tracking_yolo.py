from ultralytics import YOLO

model = YOLO('yolov8m.pt')

model.train(data='data.yaml', epochs=50, batch=16, imgsz=640)

result = model.predict('../video/prova.mp4', save=True)




