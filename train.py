import torch
from ultralytics import YOLO


def main():
    print("=== STARTING YOLO26s TRAINING ON NVIDIA V100 ===")

    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    model = YOLO("yolo26s.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=5,
        imgsz=640,
        device=0,
        freeze=10,
        batch=16,
        lr0=0.001,
        weight_decay=0.005,
        patience=20,
        project="runs/detect",
        name="vin_construction_yolo26s",
        exist_ok=True,
    )

    print("=== TRAINING COMPLETED SUCCESSFULLY! ===")


if __name__ == "__main__":
    main()