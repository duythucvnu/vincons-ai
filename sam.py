# -*- coding: utf-8 -*-
import io
import json
import torch
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics.models.sam import SAM3SemanticPredictor

MODEL_PATH = "sam3.pt"
predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("=== ĐANG KHỞI TẠO MÔ HÌNH SAM 3 ===")
    overrides = dict(
        conf=0.45,
        task="detect",
        mode="predict",
        model=MODEL_PATH,
        quantize=16 if torch.cuda.is_available() else 0,
        save=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    print("=== KHỞI TẠO SAM3 THÀNH CÔNG ===")
    yield
    print("=== TẮT SERVER API ===")


app = FastAPI(
    title="SAM 3 API Service",
    description="Production-grade API for Zero-Shot Object Detection & Auto-Labeling using SAM 3",
    version="1.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict-sam3")
async def predict_sam3(
    image: UploadFile = File(...),
    prompts: str = Form(...),
):
    global predictor

    if predictor is None:
        raise HTTPException(status_code=503, detail="Mô hình SAM 3 chưa được tải thành công.")

    prompts_list = []

    try:
        parsed = json.loads(prompts)
        if isinstance(parsed, list):
            prompts_list = parsed
        elif isinstance(parsed, str):
            prompts_list = [parsed]
    except json.JSONDecodeError:
        prompts_list = [p.strip() for p in prompts.split(",") if p.strip()]

    if not prompts_list:
        raise HTTPException(
            status_code=400,
            detail="Danh sách câu nhắc (prompts) không được trống.",
        )

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w_img, h_img = pil_image.size

        predictor.set_image(pil_image)

        if hasattr(predictor, "model") and predictor.model is not None:
            if not hasattr(predictor.model, "mask_threshold"):
                predictor.model.mask_threshold = 0.0

        results = predictor(text=prompts_list)

        predictions = []

        for result in results:
            if hasattr(result, "boxes") and result.boxes is not None:
                boxes_norm = result.boxes.xywhn.cpu().numpy()
                boxes_pixel = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                for box_n, box_p, cls_idx, conf in zip(
                    boxes_norm,
                    boxes_pixel,
                    classes,
                    confidences,
                ):
                    class_id = int(cls_idx)

                    predictions.append(
                        {
                            "class_id": class_id,
                            "label": prompts_list[class_id],
                            "confidence": round(float(conf), 4),
                            "bbox": [int(coord) for coord in box_p],
                            "bbox_normalized": [
                                round(float(coord), 6) for coord in box_n
                            ],
                        }
                    )

        return {
            "status": "success",
            "filename": image.filename,
            "width": w_img,
            "height": h_img,
            "predictions": predictions,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi trong quá trình xử lý AI: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print(" MÁY CHỦ API SAM 3 ĐANG KHỞI CHẠY...")
    print("• Địa chỉ máy chủ chính: http://127.0.0.1:8000")
    print("• Trang tài liệu tương tác: http://127.0.0.1:8000/docs")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)