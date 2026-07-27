# -*- coding: utf-8 -*-
import os
import sys
import zipfile
import torch
import boto3
from botocore.client import Config
from ultralytics import YOLO

MINIO_CONF = {
    "endpoint": "157.66.100.182:9001",
    "access_key": "5ivWMdFo5QGKCD2FcHOf",
    "secret_key": "l370OvFrPikCrYioFBZNbWv89r48q7DPee0HS2UQ",
    "bucket_name": "ai-data"
}

DATASET_ZIP_NAME = "dataset.zip"
LOCAL_ZIP_PATH = "dataset.zip"
EXTRACT_DIR = "dataset"

OUTPUT_MODEL_NAME = "models/best.pt"
LOCAL_WEIGHTS_PATH = "runs/detect/vin_construction_yolo26s/weights/best.pt"


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_CONF["endpoint"],
        aws_access_key_id=MINIO_CONF["access_key"],
        aws_secret_access_key=MINIO_CONF["secret_key"],
        config=Config(signature_version="s3v4"),
    )


def download_dataset():
    print("BƯỚC 1: Đang kết nối MinIO và tự động tải tập dữ liệu...")
    s3 = get_s3_client()

    s3.download_file(
        Bucket=MINIO_CONF["bucket_name"],
        Key=DATASET_ZIP_NAME,
        Filename=LOCAL_ZIP_PATH,
    )

    print("Tải tập dữ liệu thành công. Đang tiến hành giải nén...")

    with zipfile.ZipFile(LOCAL_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    print(f"Giải nén hoàn tất vào thư mục cục bộ: '{EXTRACT_DIR}/'")

    if os.path.exists(LOCAL_ZIP_PATH):
        os.remove(LOCAL_ZIP_PATH)


def upload_trained_weights():
    print("\nBƯỚC 3: Đang tự động đẩy file trọng số tốt nhất lên MinIO...")

    if not os.path.exists(LOCAL_WEIGHTS_PATH):
        print(f"Lỗi: Không tìm thấy file trọng số tại đường dẫn: {LOCAL_WEIGHTS_PATH}")
        return

    s3 = get_s3_client()

    s3.upload_file(
        Filename=LOCAL_WEIGHTS_PATH,
        Bucket=MINIO_CONF["bucket_name"],
        Key=OUTPUT_MODEL_NAME,
    )

    print(
        f"Đẩy file trọng số lên MinIO thành công tại đường dẫn: "
        f"'{MINIO_CONF['bucket_name']}/{OUTPUT_MODEL_NAME}'"
    )


def main():
    print("=== KHỞI ĐỘNG TIẾN TRÌNH HUẤN LUYỆN YOLO26s ===", flush=True)

    device_info = "CPU"
    if torch.cuda.is_available():
        device_info = f"GPU: {torch.cuda.get_device_name(0)} (CUDA v{torch.version.cuda})"

    print(f"SYS_DEVICE_DETECTED: {device_info}", flush=True)
    sys.stdout.flush()

    try:
        download_dataset()
    except Exception as e:
        print(f"LỖI TỰ ĐỘNG TẢI DỮ LIỆU TỪ MINIO: {str(e)}", flush=True)
        sys.exit(1)

    print("\nBƯỚC 2: Bắt đầu huấn luyện mô hình YOLO26s...", flush=True)

    model = YOLO("yolo26s.pt")

    model.train(
        data=os.path.join(EXTRACT_DIR, "data.yaml"),
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

    try:
        upload_trained_weights()
    except Exception as e:
        print(f"LỖI TỰ ĐỘNG ĐẨY FILE WEIGHTS LÊN MINIO: {str(e)}", flush=True)

    print("\n=== QUY TRÌNH HUẤN LUYỆN KHÉP KÍN HOÀN TẤT THÀNH CÔNG ===", flush=True)


if __name__ == "__main__":
    main()