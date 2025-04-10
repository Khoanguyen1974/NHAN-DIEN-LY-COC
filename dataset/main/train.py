from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # or yolov5s.pt, yolov8s.pt, etc.

# Train
results = model.train(
    data=r"D:\bandau\dataset\dataset\data.yml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu"  # "0" = GPU, "cpu" = CPU
)