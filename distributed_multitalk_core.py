# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
import sys
import json
import warnings
from datetime import datetime
import re
import tempfile
import uuid
from pathlib import Path

import gradio as gr
warnings.filterwarnings('ignore')

import random
import torch
import torch.distributed as dist
from PIL import Image
import subprocess

import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import cache_image, cache_video, str2bool
from wan.utils.multitalk_utils import save_video_ffmpeg
from kokoro import KPipeline
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model

import librosa
import pyloudnorm as pyln
import numpy as np
from einops import rearrange
import soundfile as sf


class DialogueScriptParser:
    """解析对话台词脚本的类"""
    
    def __init__(self):
        # 支持多种对话格式
        self.patterns = [
            # 格式: (s1) 台词 (s2) 台词
            r'\(s(\d+)\)\s*(.*?)(?=\s*\(s\d+\)|$)',
            # 格式: 角色1: 台词 角色2: 台词
            r'角色(\d+)[:：]\s*(.*?)(?=\s*角色\d+[:：]|$)',
            # 格式: 人物1: 台词 人物2: 台词
            r'人物(\d+)[:：]\s*(.*?)(?=\s*人物\d+[:：]|$)',
            # 格式: A: 台词 B: 台词 (将A映射为1，B映射为2)
            r'([AB])[:：]\s*(.*?)(?=\s*[AB][:：]|$)',
        ]
    
    def parse_dialogue(self, script_text):
        """
        解析对话脚本，返回分配给两个角色的台词
        
        Args:
            script_text (str): 输入的对话脚本
            
        Returns:
            str: 格式化的TTS文本，格式为 "(s1) 台词1 (s2) 台词2 ..."
        """
        script_text = script_text.strip()
        
        # 尝试不同的解析模式
        for i, pattern in enumerate(self.patterns):
            matches = re.findall(pattern, script_text, re.DOTALL)
            if matches:
                return self._format_dialogue(matches, pattern_index=i)
        
        # 如果没有匹配到任何模式，按行分割并交替分配
        return self._parse_by_lines(script_text)
    
    def _format_dialogue(self, matches, pattern_index):
        """格式化匹配到的对话内容"""
        formatted_dialogue = []
        
        for match in matches:
            if pattern_index == 3:  # A/B格式
                speaker_letter, content = match
                speaker_id = "1" if speaker_letter == "A" else "2"
            else:
                speaker_id, content = match
            
            content = content.strip()
            if content:
                formatted_dialogue.append(f"(s{speaker_id}) {content}")
        
        return " ".join(formatted_dialogue)
    
    def _parse_by_lines(self, script_text):
        """按行分割并交替分配给两个角色"""
        lines = [line.strip() for line in script_text.split('\n') if line.strip()]
        
        if not lines:
            return "(s1) 你好！ (s2) 很高兴见到你！"
        
        formatted_dialogue = []
        for i, line in enumerate(lines):
            speaker_id = (i % 2) + 1
            formatted_dialogue.append(f"(s{speaker_id}) {line}")
        
        return " ".join(formatted_dialogue)


def _parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Distributed MultiTalk Web Service for Dual-Person Dialogue Video Generation"
    )
    
    # 基础参数
    parser.add_argument("--task", type=str, default="multitalk-14B", help="The task to run.")
    parser.add_argument("--size", type=str, default="multitalk-720", help="Video resolution.")
    parser.add_argument("--frame_num", type=int, default=81, help="Number of frames to generate.")
    parser.add_argument("--ckpt_dir", type=str, default='./weights/Wan2.1-I2V-14B-720P', 
                       help="Path to checkpoint directory.")
    parser.add_argument("--quant_dir", type=str, default=None, help="Path to quantized checkpoint directory.")
    parser.add_argument("--wav2vec_dir", type=str, default='./weights/chinese-wav2vec2-base',
                       help="Path to wav2vec checkpoint directory.")
    parser.add_argument("--lora_dir", type=str, nargs='+', default=None, help="Path to LoRA checkpoint.")
    parser.add_argument("--lora_scale", type=float, nargs='+', default=[1.2], help="LoRA scale factors.")
    
    # 分布式参数
    parser.add_argument("--ulysses_size", type=int, default=8, help="Ulysses parallelism size.")
    parser.add_argument("--ring_size", type=int, default=1, help="Ring attention parallelism size.")
    parser.add_argument("--t5_fsdp", action="store_true", default=True, help="Use FSDP for T5.")
    parser.add_argument("--dit_fsdp", action="store_true", default=True, help="Use FSDP for DiT.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Place T5 model on CPU.")
    parser.add_argument("--offload_model", type=str2bool, default=None, help="Offload model to CPU.")
    
    # 生成参数
    parser.add_argument("--motion_frame", type=int, default=25, help="Motion frame length.")
    parser.add_argument("--sample_shift", type=float, default=11, help="Sample shift for 720P.")
    parser.add_argument("--color_correction_strength", type=float, default=1.0, help="Color correction strength.")
    parser.add_argument("--base_seed", type=int, default=42, help="Base seed for generation.")
    parser.add_argument("--quant", type=str, default=None, help="Quantization type.")
    
    # 其他参数
    parser.add_argument("--audio_save_dir", type=str, default='save_audio/distributed',
                       help="Directory to save audio embeddings.")
    parser.add_argument("--num_persistent_param_in_dit", type=int, default=None,
                       help="Maximum parameter quantity retained in VRAM.")
    parser.add_argument("--server_port", type=int, default=8419, help="Web server port.")
    
    args = parser.parse_args()
    
    # 验证参数
    assert args.task == "multitalk-14B", 'You should choose multitalk-14B in args.task.'
    assert args.size == "multitalk-720", 'This app is designed for 720P generation.'
    
    return args


if __name__ == "__main__":
    args = _parse_args()
    print("分布式MultiTalk Web服务启动中...")
    print(f"配置: 任务={args.task}, 分辨率={args.size}, Ulysses并行度={args.ulysses_size}")