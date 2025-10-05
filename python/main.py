# main.py (最终生产版)

import os
import cv2
import numpy as np
import base64
import time
import logging
import traceback
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

from processor import ImageProcessor

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
processor = None

# --- Lifespan 上下文管理器 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor
    logging.info("应用启动：开始初始化...")

    app.state.REQUEST_COUNT = Counter("fastapi_requests_total", "所有HTTP请求的总数", ["method", "endpoint", "http_status"])
    app.state.REQUEST_LATENCY = Histogram("fastapi_requests_latency_seconds", "HTTP请求的延迟（秒）", ["endpoint"])
    app.state.PROCESSING_LATENCY = Histogram("image_processing_latency_seconds", "完整图像处理流程的耗时（秒）", ["stage"])
    app.state.REQUESTS_IN_PROGRESS = Gauge("fastapi_requests_in_progress", "当前正在处理的HTTP请求数")

    logging.info("初始化 ImageProcessor 并启动工作进程池...")
    try:
        # 在这里定义所有需要预加载的模型的路径
        # 这些路径将传递给工作进程，在启动时加载
        processor = ImageProcessor(
            human_detector_path="yolov8n.pt",
            matting_model_path="pretrained/SGHM-ResNet50.pth",
            secondary_model_path="train/exp_yolov11m/weights/best.pt"
        )
    except Exception as e:
        logging.error(f"初始化图像处理器失败: {e}", exc_info=True)
        processor = None

    logging.info("初始化完成，服务已就绪。")
    yield

    if processor:
        processor.shutdown()
    logging.info("应用正在关闭。")

app = FastAPI(title="高级图像处理服务", lifespan=lifespan)

# --- 中间件 ---
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    request_count = getattr(request.app.state, 'REQUEST_COUNT', None)
    requests_in_progress = getattr(request.app.state, 'REQUESTS_IN_PROGRESS', None)
    request_latency = getattr(request.app.state, 'REQUEST_LATENCY', None)
    if not all([request_count, requests_in_progress, request_latency]): return await call_next(request)
    requests_in_progress.inc()
    start_time = time.time()
    try:
        response = await call_next(request)
        latency = time.time() - start_time
        request_latency.labels(endpoint=request.url.path).observe(latency)
        request_count.labels(method=request.method, endpoint=request.url.path, http_status=response.status_code).inc()
    except Exception as e:
        request_count.labels(method=request.method, endpoint=request.url.path, http_status=500).inc()
        raise e
    finally:
        requests_in_progress.dec()
    return response

# --- API 端点 ---
@app.post("/process_images_batch/")
async def process_images_batch(
        image_files: List[UploadFile] = File(...),
        # 注意：chicken_model_path 不再需要，因为模型已在启动时固定
        blur_method: str = Form("gaussian"),
        human_conf: float = Form(0.25),
        chicken_conf: float = Form(0.25),
):
    total_start_time = time.time()
    logging.info(f"开始处理批量图像请求，共 {len(image_files)} 张图片。")
    if processor is None: raise HTTPException(status_code=503, detail="服务暂时不可用：图像处理器未能加载。")

    start_time = time.time()
    image_nps = []
    for image_file in image_files:
        contents = await image_file.read()
        frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None: image_nps.append(frame)
        else: logging.warning(f"无法解码文件: {image_file.filename}，已跳过。")
    if not image_nps: raise HTTPException(status_code=400, detail="没有提供任何有效的图像文件。")
    logging.info(f"  [API] 阶段1: 批量读取与解码 {len(image_nps)} 张图片完成, 耗时: {time.time() - start_time:.4f}s")

    start_time = time.time()
    try:
        # 调用多GPU处理方法，不再传递 secondary_model_path
        processed_results = processor.process_batch_multi_gpu(
            image_nps=image_nps,
            blur_method=blur_method,
            human_conf=human_conf,
            chicken_conf=chicken_conf
        )
        logging.info(f"  [API] 阶段2: 核心处理器完成所有任务, 耗时: {time.time() - start_time:.4f}s")
    except Exception as e:
        logging.error(f"处理批量请求时发生严重错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

    logging.info(f"批量图像请求处理完毕, 总耗时: {time.time() - total_start_time:.4f}s")
    return JSONResponse(content={"results": processed_results})

app.mount("/metrics", make_asgi_app())
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001)
