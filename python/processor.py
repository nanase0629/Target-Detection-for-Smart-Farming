# processor.py (最终生产版 - 启动时加载所有模型，使用文件日志)

import os
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from ultralytics import YOLO
from collections import OrderedDict
import inference
from model.model import HumanMatting
import base64
import time
import multiprocessing as mp
import uuid
import traceback

# =================== 顶层函数，用于持久化的工作进程 ===================

def _setup_subprocess_logging(gpu_id: int, project_root: str):
    """为每个子进程设置独立的日志文件"""
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"GPU-{gpu_id}")
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(log_dir, f"gpu_{gpu_id}_process.log"), mode='w')
    formatter = logging.Formatter('%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def _gpu_worker_process(
        gpu_id: int,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        model_paths: dict,
        project_root: str
):
    """一个长期运行的工作进程函数，启动时加载所有模型。"""
    logger = _setup_subprocess_logging(gpu_id, project_root)
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)

    logger.info(f"启动，开始加载所有模型...")

    try:
        # --- 阶段 1: 在子进程启动时，加载所有三个模型 ---
        stage_start_time = time.time()
        human_detector = YOLO(model_paths['human'])
        secondary_model = YOLO(model_paths['secondary'])

        matting_model = HumanMatting(backbone='resnet50')
        state_dict = torch.load(model_paths['matting'], map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        matting_model.load_state_dict(new_state_dict)
        matting_model = matting_model.to(device).eval()

        logger.info(f"阶段 1: 所有模型加载完成, 耗时: {time.time() - stage_start_time:.4f}s")
        logger.info(f"进入任务等待循环。")

        while True:
            job_id, task_data = input_queue.get()
            if task_data is None:
                logger.info("收到退出信号，进程终止。")
                break

            image_nps, blur_method, human_conf, chicken_conf = task_data
            logger.info(f"收到任务 {job_id}，开始处理 {len(image_nps)} 张图片。")

            task_start_time = time.time()

            # --- 阶段 2: 批量人体检测 ---
            stage_start_time = time.time()
            human_results_batch = human_detector.predict(source=image_nps, conf=human_conf, classes=0, device=device, verbose=False)
            logger.info(f"  阶段 2: 人体检测完成, 耗时: {time.time() - stage_start_time:.4f}s")

            # --- 阶段 3: 串行抠图打码 ---
            # stage_start_time = time.time()
            # processed_frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in image_nps]
            # all_human_boxes = [res.boxes.xyxy.cpu().numpy().tolist() for res in human_results_batch]
            # for i, frame_rgb in enumerate(processed_frames_rgb):
            #     for box in all_human_boxes[i]:
            #         x1,y1,x2,y2=map(int,box);x1,y1,x2,y2=max(0,x1),max(0,y1),min(frame_rgb.shape[1],x2),min(frame_rgb.shape[0],y2)
            #         if x2<=x1 or y2<=y1: continue
            #         person_region=frame_rgb[y1:y2,x1:x2]
            #         try:
            #             _,pred_mask=inference.single_inference(matting_model,Image.fromarray(person_region),device=device)
            #             if pred_mask.ndim==3: pred_mask=pred_mask.squeeze(axis=2)
            #             mask_3d=np.repeat((pred_mask>0.5)[:,:,np.newaxis],3,axis=2)
            #             if blur_method=='gaussian': blurred=cv2.GaussianBlur(person_region.copy(),(99,99),30)
            #             else: h,w=person_region.shape[:2];small=cv2.resize(person_region.copy(),(w//15,h//15),interpolation=cv2.INTER_LINEAR);blurred=cv2.resize(small,(w,h),interpolation=cv2.INTER_NEAREST)
            #             processed_region=np.where(mask_3d,blurred,person_region)
            #             frame_rgb[y1:y2,x1:x2]=processed_region
            #         except Exception as e: logger.warning(f"处理人物区域时出错: {e}")
            # blurred_frames_bgr=[cv2.cvtColor(frame_rgb,cv2.COLOR_RGB2BGR) for frame_rgb in processed_frames_rgb]
            # logger.info(f"  阶段 3: 抠图打码完成, 耗时: {time.time()-stage_start_time:.4f}s")

            # 为了保持数据流，直接将原始帧（BGR格式）传递给下一阶段
            all_human_boxes = [res.boxes.xyxy.cpu().numpy().tolist() for res in human_results_batch]
            blurred_frames_bgr = image_nps
            logger.info("  阶段 3: 抠图打码阶段已跳过。")

            # --- 阶段 4: 批量二次检测 ---
            stage_start_time = time.time()
            secondary_results_batch = secondary_model.predict(source=blurred_frames_bgr, conf=chicken_conf, device=device, verbose=False)
            logger.info(f"  阶段 4: 二次检测完成, 耗时: {time.time() - stage_start_time:.4f}s")

            # --- 阶段 5: 结果整合编码 ---
            stage_start_time = time.time()
            task_results = []
            for i, secondary_results in enumerate(secondary_results_batch):
                annotated_frame=secondary_results.plot();chicken_boxes=secondary_results.boxes.xyxy.cpu().numpy().tolist() if secondary_results.boxes else[]
                _,encoded_image_bytes=cv2.imencode(".jpg",annotated_frame)
                final_image_base64=base64.b64encode(encoded_image_bytes).decode('utf-8')
                task_results.append({"final_image_base64":final_image_base64,"detected_humans":all_human_boxes[i],"detected_chickens":chicken_boxes})
            logger.info(f"  阶段 5: 结果整合编码完成, 耗时: {time.time()-stage_start_time:.4f}s")
            logger.info(f"任务 {job_id} 完成, 总耗时: {time.time() - task_start_time:.4f}s")

            output_queue.put((job_id, task_results))

    except Exception as e:
        logger.error(f"发生致命错误，进程退出: {e}", exc_info=True)
        # 注意：此时 job_id 可能未定义，所以我们用一个固定的错误标记
        output_queue.put(("FATAL_ERROR", f"Worker {os.getpid()} on GPU {gpu_id} crashed: {e}"))

class ImageProcessor:
    def __init__(self,
                 human_detector_path='yolov8n.pt',
                 matting_model_path='./pretrained/SGHM-ResNet50.pth',
                 secondary_model_path='./train/exp_yolov11m/weights/best.pt'):

        logging.info("正在初始化图像处理器 (主进程)...")
        self.PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
        self.device_count = torch.cuda.device_count()
        if self.device_count == 0:
            logging.error("未检测到GPU，多GPU处理模式无法启动。")
            self.workers = []
            return

        ctx = mp.get_context("spawn")
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.workers = []

        def to_abs(path): return os.path.join(self.PROJECT_ROOT, path) if not os.path.isabs(path) else path

        model_paths = {
            'human': to_abs(human_detector_path),
            'matting': to_abs(matting_model_path),
            'secondary': to_abs(secondary_model_path)
        }

        for gpu_id in range(self.device_count):
            worker = ctx.Process(
                target=_gpu_worker_process,
                args=(gpu_id, self.input_queue, self.output_queue, model_paths, self.PROJECT_ROOT)
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            logging.info(f"工作进程 for GPU {gpu_id} 已启动 (PID: {worker.pid})。")

    def shutdown(self):
        logging.info("正在关闭工作进程...")
        for _ in self.workers: self.input_queue.put((None, None))
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive(): worker.terminate()
        logging.info("所有工作进程已关闭。")

    def process_batch_multi_gpu(self, image_nps: list, blur_method='gaussian', human_conf=0.25, chicken_conf=0.25):
        if self.device_count == 0: raise NotImplementedError("无可用GPU。")
        num_images, num_workers = len(image_nps), len(self.workers)
        if num_workers == 0: return []

        image_chunks = np.array_split(image_nps, num_workers)

        job_id = str(uuid.uuid4())
        jobs_dispatched = 0
        for chunk in image_chunks:
            if len(chunk) > 0:
                task_data = (list(chunk), blur_method, human_conf, chicken_conf)
                self.input_queue.put((job_id, task_data))
                jobs_dispatched += 1

        logging.info(f"已将 {num_images} 张图片分发为 {jobs_dispatched} 个任务 (Job ID: {job_id})。等待处理结果...")

        final_results = []
        for _ in range(jobs_dispatched):
            try:
                retrieved_job_id, result_chunk = self.output_queue.get(timeout=300)
                if retrieved_job_id == job_id and isinstance(result_chunk, list):
                    final_results.extend(result_chunk)
                else: logging.error(f"一个工作进程返回了错误或不匹配的结果: {result_chunk}")
            except Exception as e:
                logging.error(f"等待工作进程结果时发生错误或超时: {e}")
                break

        logging.info(f"已收到所有任务块的结果，共聚合了 {len(final_results)} 个结果。")
        return final_results