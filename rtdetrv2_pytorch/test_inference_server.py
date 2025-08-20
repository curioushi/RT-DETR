#!/usr/bin/env python3
"""
RT-DETR推理服务器测试脚本
"""

import base64
import requests
import json
from PIL import Image, ImageDraw
import io
import cv2
import numpy as np


def encode_image_to_base64(image_path: str) -> str:
    """将图像文件编码为Base64字符串"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def test_inference_server(image_path: str, server_url: str = "http://localhost:22336"):
    """测试推理服务器"""
    
    # 编码图像
    print(f"正在编码图像: {image_path}")
    image_base64 = encode_image_to_base64(image_path)
    
    # 准备请求数据
    request_data = {
        "image": image_base64,
        "image_format": "jpeg",
        "confidence_threshold": 0.6
    }
    
    # 发送请求
    print(f"正在发送请求到: {server_url}/inference")
    try:
        response = requests.post(
            f"{server_url}/inference",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("推理成功!")
            print(f"状态: {result['status']}")
            print(f"消息: {result['message']}")
            
            if result['data']:
                data = result['data']
                print(f"检测到 {data['total_detections']} 个目标")
                print(f"处理时间: {data['processing_time']:.2f} 秒")
                
                for i, detection in enumerate(data['detections']):
                    print(f"目标 {i+1}:")
                    print(f"  标签: {detection['label']}")
                    print(f"  置信度: {detection['confidence']:.3f}")
                    print(f"  四边形坐标: {detection['quad']}")
                
                # 保存推理结果图像
                save_inference_result(image_path, data['detections'])
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")


def save_inference_result(image_path: str, detections: list):
    """保存推理结果图像"""
    try:
        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return
        
        # 定义颜色映射
        class_to_colors = {
            0: (255, 0, 0),    # 红色
            1: (0, 255, 0),    # 绿色
            2: (0, 0, 255),    # 蓝色
            3: (255, 255, 0),  # 黄色
        }
        
        # 在图像上绘制检测结果
        for i, detection in enumerate(detections):
            label = detection['label']
            confidence = detection['confidence']
            quad = detection['quad']
            
            # 获取颜色
            color = class_to_colors.get(label, (255, 255, 255))  # 默认白色
            
            # 转换四边形坐标为numpy数组
            quad_points = np.array(quad, dtype=np.int32)
            
            # 绘制四边形
            cv2.polylines(image, [quad_points], True, color, 2)
            
            # 添加标签和置信度文本
            text = f"L{label}:{confidence:.2f}"
            text_position = (int(quad[0][0]), int(quad[0][1]) - 10)
            
            # 绘制文本背景
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, 
                         (text_position[0], text_position[1] - text_height - 5),
                         (text_position[0] + text_width, text_position[1] + 5),
                         color, -1)
            
            # 绘制文本
            cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 保存结果图像
        output_path = "test.jpg"
        cv2.imwrite(output_path, image)
        print(f"推理结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"保存推理结果图像失败: {e}")


def test_health_check(server_url: str = "http://localhost:22336"):
    """测试健康检查端点"""
    try:
        response = requests.get(f"{server_url}/health")
        if response.status_code == 200:
            result = response.json()
            print(f"健康检查成功: {result}")
        else:
            print(f"健康检查失败，状态码: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"健康检查异常: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RT-DETR推理服务器测试")
    parser.add_argument("--image", type=str, required=True, help="测试图像路径")
    parser.add_argument("--server", type=str, default="http://localhost:22336", help="服务器URL")
    
    args = parser.parse_args()
    
    print("=== RT-DETR推理服务器测试 ===")
    
    # 测试健康检查
    print("\n1. 测试健康检查...")
    test_health_check(args.server)
    
    # 测试推理
    print(f"\n2. 测试推理功能...")
    test_inference_server(args.image, args.server)
