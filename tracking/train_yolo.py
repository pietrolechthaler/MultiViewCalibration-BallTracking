from ultralytics import YOLO

model = YOLO('yolo11n.pt')


model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    save=True,
    augment=True,  # Augmentation automatica
    hsv_h=0.015,  # Hue augmentation
    hsv_s=0.7,    # Saturation
    hsv_v=0.4,    # Brightness
    degrees=10,   # Rotazione
    flipud=0.5,   # Flip verticale
)


