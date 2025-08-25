#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

"""
分布式MultiTalk双人对话视频生成Web服务

功能特性:
- 支持两个人物图片上传
- 智能对话脚本解析 
- 自动TTS语音生成
- 720P高清视频输出
- 8张RTX-4090分布式推理
- 友好的Web界面

使用方法:
# 单GPU测试运行
python distributed_multitalk_app.py

# 8GPU分布式运行  
torchrun --nproc_per_node=8 distributed_multitalk_app.py --ulysses_size=8 --dit_fsdp --t5_fsdp

作者: MultiTalk团队
版本: 1.0.0
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from distributed_multitalk_core import _parse_args
from distributed_generator import DistributedMultiTalkGenerator
from distributed_web_interface import create_gradio_interface


def main():
    """主函数"""
    
    # 打印启动信息
    print("=" * 60)
    print("🎬 分布式MultiTalk双人对话视频生成服务")
    print("=" * 60)
    print("功能特性:")
    print("- 👥 双人图片合成")
    print("- 💬 智能对话解析") 
    print("- 🎤 自动TTS生成")
    print("- 📺 720P高清输出")
    print("- 🚀 分布式推理")
    print("=" * 60)
    
    # 解析参数
    args = _parse_args()
    
    # 显示配置信息
    print("配置信息:")
    print(f"- 任务类型: {args.task}")
    print(f"- 视频分辨率: {args.size}")
    print(f"- 帧数: {args.frame_num}")
    print(f"- Ulysses并行度: {args.ulysses_size}")
    print(f"- Ring并行度: {args.ring_size}")
    print(f"- T5 FSDP: {args.t5_fsdp}")
    print(f"- DiT FSDP: {args.dit_fsdp}")
    print(f"- 服务端口: {args.server_port}")
    print("=" * 60)
    
    try:
        # 创建分布式生成器
        print("正在初始化分布式生成器...")
        generator = DistributedMultiTalkGenerator(args)
        print("✅ 分布式生成器初始化完成")
        
        # 创建Web界面
        print("正在创建Web界面...")
        demo = create_gradio_interface(generator)
        print("✅ Web界面创建完成")
        
        # 启动服务
        print(f"正在启动Web服务 (端口: {args.server_port})...")
        print("=" * 60)
        print("🌐 服务启动成功！")
        print(f"📱 访问地址: http://localhost:{args.server_port}")
        print(f"🔗 外网访问: http://0.0.0.0:{args.server_port}")
        print("=" * 60)
        print("🎯 使用说明:")
        print("1. 在浏览器中打开上述地址")
        print("2. 上传两个人物图片")
        print("3. 输入对话台词脚本")
        print("4. 描述场景环境")
        print("5. 点击生成按钮等待720P视频")
        print("=" * 60)
        
        # 启动Gradio服务
        demo.launch(
            server_name="0.0.0.0",
            server_port=args.server_port,
            debug=True,
            share=False,  # 分布式环境不建议开启share
            show_api=False,
            quiet=False
        )
        
    except KeyboardInterrupt:
        print("\n👋 用户中断，正在关闭服务...")
        
    except Exception as e:
        print(f"❌ 启动服务时发生错误: {e}")
        logging.error(f"Service startup error: {e}", exc_info=True)
        sys.exit(1)
        
    finally:
        print("🔚 服务已关闭")


def validate_environment():
    """验证运行环境"""
    
    # 检查必要的目录
    required_dirs = [
        "weights/Wan2.1-I2V-14B-720P",
        "weights/chinese-wav2vec2-base", 
        "weights/Kokoro-82M"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("❌ 缺少必要的模型文件目录:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        print("\n请确保已下载所有必要的模型文件")
        return False
    
    # 检查GPU
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  警告: 未检测到CUDA GPU，将使用CPU运行（速度较慢）")
        else:
            gpu_count = torch.cuda.device_count()
            print(f"✅ 检测到 {gpu_count} 张GPU")
            
            if gpu_count >= 8:
                print("🚀 推荐使用8GPU分布式模式以获得最佳性能")
                print("   启动命令: torchrun --nproc_per_node=8 distributed_multitalk_app.py --ulysses_size=8 --dit_fsdp --t5_fsdp")
            
    except ImportError:
        print("❌ 未找到PyTorch，请先安装PyTorch")
        return False
    
    return True


if __name__ == "__main__":
    
    # 验证环境
    if not validate_environment():
        sys.exit(1)
    
    # 启动服务
    main()