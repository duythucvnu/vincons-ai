# vincons-ai

Concise deployment guide to run and integrate the training pipeline.

## Directory

```text
vincons-ai/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── train.py                # YOLO26s training script
├── sam.py                  # SAM 3 FastAPI Service (Auto-labeling API)
├── sam3.pt                 # [Local Only] SAM 3 model weight file
├── runs/                   # [Local Only] Output weights
└── dataset/                # [Local Only] Place raw images & labels here
```

## Setup

```bash
git clone https://github.com/duythucvnu/vincons-ai.git
cd vincons-ai
pip install -r requirements.txt
```

---

## YOLO26s Training Pipeline

### 1. Prepare the dataset
```text
dataset/                
├── train/
├── valid/
├── test/
└── data.yaml
```

Create `dataset/data.yaml`:

```yaml
names:
  0: crane
  1: worker
  2: supply platform
```

### 2. Run Training

Execute:

```bash
docker compose up --build
```

After training completes, the model weights will be saved to:

```text
runs/detect/vin_construction_yolo26s/weights/
```

---

## SAM3 API Service

Run the API in the background on **port 8000**:

```bash
nohup python3 sam.py > api.log 2>&1 &
```
### Additional Information

Interactive Swagger Documentation: http://<SERVER_IP>:8000/docs

Endpoint URL: POST http://<SERVER_IP>:8000/predict-sam3

Headers: Content-Type: multipart/form-data

Form-Data Fields:

image: File (Binary image upload)

prompts: String (Accepts both JSON arrays like ["person", "crane"] or comma-separated strings like person, crane, supply platform)

Link model SAM3: https://drive.google.com/drive/folders/1nv4xJkZzplog01zqylYsjMxHuQydSFjp?usp=sharing