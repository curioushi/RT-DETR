#!/usr/bin/env python3
"""
RT-DETR 目标检测推理服务器

用法:
    python rtdetrv2_inference_server.py --host 0.0.0.0 --port 22336 --device cuda

API端点:
    POST /inference - 目标检测推理
    GET /health - 健康检查
    GET /docs - API文档
"""

import argparse
import base64
import io
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
from huggingface_hub import hf_hub_download

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core import YAMLConfig


class DetectionRequest(BaseModel):
    """检测请求模型"""
    image: str  # Base64编码的图像数据
    image_format: str = "jpeg"  # 图像格式
    confidence_threshold: float = 0.6  # 置信度阈值


class DetectionResult(BaseModel):
    """单个检测结果模型"""
    label: int
    confidence: float
    quad: List[List[float]]  # 4个点的坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]


class DetectionResponse(BaseModel):
    """检测响应模型"""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


class RTDETRInferenceServer:
    """RT-DETR目标检测推理服务器类"""

    def __init__(self, device: str = "cpu", config_path: str = None, model_path: str = None):
        self.device = torch.device(device)
        self.config_path = config_path
        self.model_path = model_path
        self.model = None
        self.transform = None

    def load_model(self):
        """加载RT-DETR检测模型"""
        print("正在加载RT-DETR检测模型...")

        if not self.config_path:
            raise ValueError("必须提供config参数")

        # 确定模型路径
        if self.model_path:
            model_path = self.model_path
            print(f"使用本地模型: {model_path}")
        else:
            # 从Hugging Face下载
            model_path = hf_hub_download(
                repo_id="Curioushi61/BoxAutoLabel",
                filename="box_detection.pth",
                cache_dir="./pretrained_models",
            )
            print(f"从Hugging Face下载模型到: {model_path}")

        # 加载配置
        cfg = YAMLConfig(self.config_path)

        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # 加载模型状态
        cfg.model.load_state_dict(state)

        # 创建deploy模式模型
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()
                
            def forward(self, images, orig_target_sizes, targets=None):
                outputs = self.model(images, targets)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs

        self.model = Model().to(self.device)
        self.model.eval()

        # 设置预处理变换
        self.transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])

        print("模型加载完成")

    def decode_base64_image(self, image_data: str, image_format: str) -> Image.Image:
        """解码Base64图像数据"""
        try:
            # 移除可能的data URL前缀
            if image_data.startswith("data:image/"):
                image_data = image_data.split(",")[1]

            # 解码Base64数据
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            # 转换为RGB模式
            if image.mode != "RGB":
                image = image.convert("RGB")

            return image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"图像解码失败: {str(e)}")

    @staticmethod
    def fix_points_order(points: list[tuple[int, int]], label: int) -> list[tuple[int, int]]:
        """Fix points order based on label"""
        order = None
        if label in [0, 1, 2]:
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            sorted_idx_x = np.argsort(xs)
            if ys[sorted_idx_x[0]] < ys[sorted_idx_x[1]]:
                left_top_idx = sorted_idx_x[0]
                left_bottom_idx = sorted_idx_x[1]
            else:
                left_top_idx = sorted_idx_x[1]
                left_bottom_idx = sorted_idx_x[0]
            if ys[sorted_idx_x[2]] < ys[sorted_idx_x[3]]:
                right_top_idx = sorted_idx_x[2]
                right_bottom_idx = sorted_idx_x[3]
            else:
                right_top_idx = sorted_idx_x[3]
                right_bottom_idx = sorted_idx_x[2]
            order = [left_top_idx, left_bottom_idx, right_bottom_idx, right_top_idx]
        elif label == 3:
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            sorted_idx_x = np.argsort(xs)
            sorted_idx_y = np.argsort(ys)
            if xs[sorted_idx_y[0]] < xs[sorted_idx_y[3]]:
                p0_idx = sorted_idx_y[0]
                p2_idx = sorted_idx_y[3]
                if xs[sorted_idx_y[1]] < xs[sorted_idx_y[2]]:
                    p1_idx = sorted_idx_y[1]
                    p3_idx = sorted_idx_y[2]
                else:
                    p1_idx = sorted_idx_y[2]
                    p3_idx = sorted_idx_y[1]
                order = [p0_idx, p1_idx, p2_idx, p3_idx]
            else:
                p1_idx = sorted_idx_y[3]
                p3_idx = sorted_idx_y[0]
                if xs[sorted_idx_y[1]] < xs[sorted_idx_y[2]]:
                    p0_idx = sorted_idx_y[1]
                    p2_idx = sorted_idx_y[2]
                else:
                    p0_idx = sorted_idx_y[2]
                    p2_idx = sorted_idx_y[1]
                order = [p0_idx, p1_idx, p2_idx, p3_idx]
        else:
            raise ValueError(f"Invalid label: {label}")

        points = [points[i] for i in order]
        return points

    def run_inference(self, image: Image.Image, confidence_threshold: float = 0.6) -> Dict[str, Any]:
        """运行模型推理"""
        print("开始预处理图像...")

        # 获取原始图像尺寸
        w, h = image.size
        orig_size = torch.tensor([w, h])[None].to(self.device)

        # 预处理图像
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        print(f"预处理完成，图像形状: {image_tensor.shape}")

        # 运行推理
        print("开始推理...")
        start_time = time.time()

        with torch.no_grad():
            output = self.model(image_tensor, orig_size)

        inference_time = time.time() - start_time
        print(f"推理完成，耗时: {inference_time:.2f}秒")

        # 处理输出结果
        labels, boxes, scores, quads, masks, offsets = output
        labels = labels[0]
        boxes = boxes[0]
        scores = scores[0]
        quads = quads[0]
        masks = masks[0]
        offsets = offsets[0]

        masks = F.sigmoid(masks)
        masks_max_pool = F.max_pool2d(masks.unsqueeze(1), kernel_size=5, padding=2, stride=1).squeeze(1)
        masks_peak = masks * (masks == masks_max_pool).float()

        boxes = boxes.cpu().numpy()
        labels = labels.cpu().numpy()
        scores = scores.cpu().numpy()
        quads = quads.cpu().numpy()
        masks = masks.cpu().numpy()
        masks_peak = masks_peak.cpu().numpy()
        offsets = offsets.cpu().numpy()

        top3_indices = np.argsort(scores)[-3:]
        top3_labels = []
        top3_scores = []
        top3_quads = []
        for i in top3_indices:
            label = labels[i]
            top3_labels.append(int(label))
            top3_scores.append(float(scores[i]))
            # Find the four points with maximum values in pred_masks_peak_np
            mask_flat = masks_peak[i].flatten()
            top_4_indices = np.argsort(mask_flat)[-4:]
            ys, xs = np.unravel_index(top_4_indices, masks_peak[i].shape)
            x_offsets = offsets[i, 0, ys, xs]
            y_offsets = offsets[i, 1, ys, xs]
            quads = np.stack([xs + x_offsets, ys + y_offsets], axis=1) * 8
            quads[:, 0] *= w / 640
            quads[:, 1] *= h / 640
            quads = self.fix_points_order(quads.tolist(), label)
            top3_quads.append(quads)

        # 准备检测结果
        detections = []
        for label, score, quad in zip(top3_labels, top3_scores, top3_quads):
            detection = DetectionResult(
                label=label,
                confidence=score,
                quad=quad
            )
            detections.append(detection)

        results = {
            "detections": detections,
            "processing_time": inference_time,
            "total_detections": len(detections)
        }

        print(f"检测到 {len(detections)} 个目标")
        return results


# 全局服务器实例
server = None


def create_app(device: str = "cpu", config_path: str = None, model_path: str = None):
    """创建FastAPI应用"""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理"""
        # 启动时加载模型
        global server
        server = RTDETRInferenceServer(
            device=device, config_path=config_path, model_path=model_path
        )
        server.load_model()
        yield
        # 关闭时清理资源（如果需要）

    app = FastAPI(
        title="RT-DETR目标检测推理服务器",
        description="基于RT-DETR的目标检测HTTP服务",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health_check():
        """健康检查端点"""
        return {"status": "healthy", "message": "RT-DETR目标检测推理服务器运行正常"}

    @app.post("/inference", response_model=DetectionResponse)
    async def inference_endpoint(request: DetectionRequest):
        """目标检测推理端点"""
        try:
            # 解码图像
            image = server.decode_base64_image(request.image, request.image_format)

            # 运行推理
            results = server.run_inference(image, request.confidence_threshold)

            return DetectionResponse(
                status="success",
                message=f"推理完成，检测到 {results['total_detections']} 个目标，耗时 {results['processing_time']:.2f} 秒",
                data=results,
            )

        except Exception as e:
            return DetectionResponse(status="error", message=f"推理失败: {str(e)}")

    return app


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RT-DETR目标检测推理服务器")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=22336, help="服务器端口")
    parser.add_argument("--device", type=str, default="cuda", help="使用设备 (cpu/cuda)")
    parser.add_argument("--config", type=str, default="configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml", help="配置文件路径")
    parser.add_argument("--model", type=str, help="模型checkpoint路径 (可选，不提供则从Hugging Face下载)")
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("启动RT-DETR目标检测推理服务器...")
    print(f"主机: {args.host}")
    print(f"端口: {args.port}")
    print(f"设备: {args.device}")
    print(f"配置文件: {args.config}")
    if args.model:
        print(f"模型文件: {args.model}")
    else:
        print("模型: 从Hugging Face自动下载")
    print(f"API文档: http://{args.host}:{args.port}/docs")

    app_instance = create_app(
        device=args.device, config_path=args.config, model_path=args.model
    )

    uvicorn.run(app_instance, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
