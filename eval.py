import os
import sys
import zipfile
import torch
import boto3
from botocore.client import Config
from ultralytics import YOLO

MINIO_CONF = {
    "endpoint": "157.66.100.182:9000",
    "access_key": "5ivWMdFo5QGKCD2FcHOf",
    "secret_key": "l370OvFrPikCrYioFBZNbWv89r48q7DPee0HS2UQ",
    "bucket_name": "ai-data"
}
DATASET_ZIP_NAME = "dataset.zip"
LOCAL_ZIP_PATH = "dataset.zip"
EXTRACT_DIR = "./data"
DATA_YAML = "./data/data.yaml"

PAST_MODEL = "past.pt"
NOW_MODEL = "now.pt"


def get_s3_client():
    endpoint = MINIO_CONF["endpoint"]
    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        endpoint = f"http://{endpoint}"

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=MINIO_CONF["access_key"],
        aws_secret_access_key=MINIO_CONF["secret_key"],
        config=Config(signature_version="s3v4")
    )


def download_assets():
    s3 = get_s3_client()
    bucket = MINIO_CONF["bucket_name"]
    
    print("BƯỚC 1: Đang tải 2 file mô hình từ MinIO về đối sánh...", flush=True)
    s3.download_file(Bucket=bucket, Key="models/past.pt", Filename=PAST_MODEL)
    s3.download_file(Bucket=bucket, Key="models/now.pt", Filename=NOW_MODEL)
    print("Tải mô hình thành công.", flush=True)
    
    if not os.path.exists(DATA_YAML):
        print("Cảnh báo: Không tìm thấy dataset cục bộ. Đang tải tự động từ MinIO...", flush=True)
        s3.download_file(Bucket=bucket, Key=DATASET_ZIP_NAME, Filename=LOCAL_ZIP_PATH)

        with zipfile.ZipFile(LOCAL_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

        os.remove(LOCAL_ZIP_PATH)
        print("Tải và giải nén dữ liệu kiểm định thành công.", flush=True)

def evaluate_model(model_path):
    print(f"Đang kiểm định mô hình: {model_path}...", flush=True)
    model = YOLO(model_path)

    metrics = model.val(
        data=DATA_YAML,
        split="val",
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False
    )

    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map

    latency = sum([metrics.speed.get(k, 0.0) for k in ["preprocess", "inference", "postprocess"]])

    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "latency": latency
    }


def main():
    try:
        download_assets()
    except Exception as e:
        print(f"LỖI TẢI TÀI NGUYÊN TỪ MINIO: {str(e)}", flush=True)
        sys.exit(1)

    m_past = evaluate_model(PAST_MODEL)
    m_now = evaluate_model(NOW_MODEL)

    d_precision = m_now["precision"] - m_past["precision"]
    d_recall = m_now["recall"] - m_past["recall"]
    d_map50 = m_now["map50"] - m_past["map50"]
    d_latency = m_now["latency"] - m_past["latency"]

    def fmt(delta, is_percent=True):
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta*100:.2f}%" if is_percent else f"{sign}{delta:.2f} ms"

    print("\n" + "=" * 60)
    print("BÁO CÁO THAY ĐỔI HIỆU NĂNG MÔ HÌNH (V100 Server)")
    print(f"Precision: {m_past['precision']*100:.2f}% -> {m_now['precision']*100:.2f}% ({fmt(d_precision)})")
    print(f"Recall: {m_past['recall']*100:.2f}% -> {m_now['recall']*100:.2f}% ({fmt(d_recall)})")
    print(f"mAP50: {m_past['map50']*100:.2f}% -> {m_now['map50']*100:.2f}% ({fmt(d_map50)})")
    print(f"Latency: {m_past['latency']:.1f}ms -> {m_now['latency']:.1f}ms ({fmt(-d_latency, False)})")
    print("=" * 60 + "\n", flush=True)

    for path in [PAST_MODEL, NOW_MODEL]:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    main()