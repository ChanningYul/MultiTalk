#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试配置模块
提供轻量级调试模式，避免每次调试都加载完整模型
"""

import os
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Generator
from PIL import Image

# 调试模式环境变量
DEBUG_MODE = os.getenv('MULTITALK_DEBUG', 'false').lower() == 'true'
MOCK_MODEL_OUTPUTS = os.getenv('MULTITALK_MOCK_OUTPUTS', 'true').lower() == 'true'

class MockWav2VecModel:
    """Mock Wav2Vec2模型，用于调试"""
    
    def __init__(self, *args, **kwargs):
        self.device = kwargs.get('device', 'cpu')
        logging.info("[DEBUG] 使用Mock Wav2Vec2模型")
    
    def __call__(self, audio_array, *args, **kwargs):
        """返回模拟的音频特征"""
        # 返回固定维度的随机特征向量
        batch_size = 1 if len(audio_array.shape) == 1 else audio_array.shape[0]
        feature_dim = 768  # 典型的Wav2Vec2特征维度
        return torch.randn(batch_size, feature_dim, device=self.device)
    
    def to(self, device):
        self.device = device
        return self

class MockWav2VecFeatureExtractor:
    """Mock Wav2Vec2特征提取器"""
    
    def __init__(self, *args, **kwargs):
        logging.info("[DEBUG] 使用Mock Wav2Vec2特征提取器")
    
    def __call__(self, audio_array, sampling_rate=16000, return_tensors="pt"):
        """返回模拟的输入特征"""
        if isinstance(audio_array, list):
            audio_array = audio_array[0]
        
        # 模拟特征提取器的输出格式
        return {
            'input_values': torch.randn(1, len(audio_array) if hasattr(audio_array, '__len__') else 16000)
        }

class MockMultiTalkPipeline:
    """Mock MultiTalk管道，用于调试"""
    
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get('config')
        self.device_id = kwargs.get('device_id', 0)
        self.vram_management = False
        logging.info("[DEBUG] 使用Mock MultiTalk管道")
    
    def enable_vram_management(self, **kwargs):
        """模拟VRAM管理启用"""
        self.vram_management = True
        logging.info("[DEBUG] Mock VRAM管理已启用")
    
    def __call__(self, *args, **kwargs):
        """模拟视频生成"""
        logging.info("[DEBUG] 模拟视频生成中...")
        
        # 返回模拟的视频数据
        # 假设生成16帧，分辨率512x512的视频
        frames = 16
        height, width = 512, 512
        
        # 生成随机视频帧（RGB格式）
        video_frames = np.random.randint(0, 255, (frames, height, width, 3), dtype=np.uint8)
        
        return {
            'video': video_frames,
            'audio': np.random.randn(16000 * 5),  # 5秒音频
            'metadata': {
                'fps': 8,
                'duration': frames / 8,
                'resolution': (width, height)
            }
        }

class MockKPipeline:
    """Mock Kokoro TTS管道"""
    
    def __init__(self, *args, **kwargs):
        logging.info("[DEBUG] 使用Mock Kokoro TTS管道")
    
    def __call__(self, text, voice=None, speed=1.0, split_pattern=None, *args, **kwargs):
        """模拟TTS生成，返回生成器"""
        logging.info(f"[DEBUG] 模拟TTS生成: {text[:50]}...")
        
        # 模拟生成器行为，返回生成器
        def mock_generator():
            # 根据文本长度生成合适的音频长度
            text_length = len(text)
            duration = max(1.0, min(text_length * 0.1, 10.0))  # 0.1秒每字符，最少1秒，最多10秒
            sample_rate = 22050
            audio_length = int(duration * sample_rate / speed)
            
            # 生成模拟的音频波形
            audio = torch.randn(audio_length) * 0.1
            
            # 模拟分段生成，每段1-2秒
            segment_length = int(sample_rate * 2 / speed)  # 2秒每段
            
            for i in range(0, audio_length, segment_length):
                end_idx = min(i + segment_length, audio_length)
                audio_segment = audio[i:end_idx]
                
                # 返回 (gs, ps, audio) 格式，模拟真实的KPipeline输出
                gs = None  # grapheme sequence (不需要)
                ps = None  # phoneme sequence (不需要)
                yield gs, ps, audio_segment
        
        return mock_generator()

class DebugConfig:
    """调试配置类"""
    
    @staticmethod
    def is_debug_mode() -> bool:
        """检查是否为调试模式"""
        return DEBUG_MODE
    
    @staticmethod
    def should_mock_outputs() -> bool:
        """检查是否应该使用Mock输出"""
        return MOCK_MODEL_OUTPUTS
    
    @staticmethod
    def get_mock_classes() -> Dict[str, Any]:
        """获取Mock类的映射"""
        return {
            'Wav2Vec2Model': MockWav2VecModel,
            'Wav2Vec2FeatureExtractor': MockWav2VecFeatureExtractor,
            'MultiTalkPipeline': MockMultiTalkPipeline,
            'KPipeline': MockKPipeline
        }
    
    @staticmethod
    def setup_debug_logging():
        """设置调试日志"""
        if DEBUG_MODE:
            logging.basicConfig(
                level=logging.DEBUG,
                format='[%(asctime)s] [DEBUG] %(message)s',
                datefmt='%H:%M:%S'
            )
            logging.info("调试模式已启用")
            logging.info(f"Mock输出: {MOCK_MODEL_OUTPUTS}")

def create_debug_video_output(output_path: str, duration: float = 5.0) -> str:
    """创建调试用的模拟视频文件"""
    import cv2
    
    # 创建一个简单的测试视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 8
    width, height = 512, 512
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames = int(duration * fps)
    for i in range(frames):
        # 创建渐变色彩的测试帧
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 255 // frames)  # 红色渐变
        frame[:, :, 1] = 128  # 固定绿色
        frame[:, :, 2] = 255 - (i * 255 // frames)  # 蓝色反向渐变
        
        # 添加文本标识
        cv2.putText(frame, f'DEBUG FRAME {i+1}/{frames}', 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    logging.info(f"[DEBUG] 创建调试视频: {output_path}")
    return output_path

def create_debug_audio_output(output_path: str, duration: float = 5.0, sample_rate: int = 22050) -> str:
    """创建调试用的模拟音频文件"""
    import soundfile as sf
    
    # 生成简单的正弦波测试音频
    t = np.linspace(0, duration, int(duration * sample_rate))
    frequency = 440  # A4音符
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # 添加一些变化使其更有趣
    audio += 0.1 * np.sin(2 * np.pi * frequency * 2 * t)
    audio += 0.05 * np.sin(2 * np.pi * frequency * 3 * t)
    
    sf.write(output_path, audio, sample_rate)
    logging.info(f"[DEBUG] 创建调试音频: {output_path}")
    return output_path

# 使用示例和说明
if __name__ == "__main__":
    print("MultiTalk 调试配置模块")
    print("="*50)
    print("使用方法:")
    print("1. 设置环境变量启用调试模式:")
    print("   export MULTITALK_DEBUG=true")
    print("   export MULTITALK_MOCK_OUTPUTS=true")
    print("")
    print("2. 在代码中使用:")
    print("   from debug_config import DebugConfig")
    print("   if DebugConfig.is_debug_mode():")
    print("       # 使用Mock类替代真实模型")
    print("")
    print(f"当前调试模式: {DEBUG_MODE}")
    print(f"当前Mock输出: {MOCK_MODEL_OUTPUTS}")