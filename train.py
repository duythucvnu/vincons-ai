import torch
from ultralytics import YOLO

def main():
    print("=== STARTING YOLO26s TRAINING ===")

    device_info = "CPU"
    if torch.cuda.is_available():
        device_info = f"GPU: {torch.cuda.get_device_name(0)} (CUDA v{torch.version.cuda})"

    print(f"SYS_DEVICE_DETECTED: {device_info}")

    model = YOLO("yolo26s.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        freeze=10,
        batch=16 if torch.cuda.is_available() else 2,
        lr0=0.001,
        weight_decay=0.005,
        patience=20,
        project="runs/detect",
        name="vin_construction_yolo26s",
        exist_ok=True,
    )

if __name__ == "__main__":
    main()