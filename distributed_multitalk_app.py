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
import socket
import subprocess
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

import os
import socket
import subprocess
from distributed_multitalk_core import _parse_args
from distributed_generator import DistributedMultiTalkGenerator
from distributed_web_interface import create_gradio_interface


def check_port_available(port, host='0.0.0.0'):
    """检查端口是否可用
    
    Args:
        port (int): 要检查的端口号
        host (str): 主机地址，默认为'0.0.0.0'
        
    Returns:
        bool: 端口可用返回True，否则返回False
    """
    try:
        # 创建socket对象
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # 尝试绑定端口
        result = sock.bind((host, port))
        sock.close()
        return True
        
    except socket.error as e:
        return False


def find_available_port(start_port, max_attempts=10):
    """寻找可用端口
    
    Args:
        start_port (int): 起始端口号
        max_attempts (int): 最大尝试次数
        
    Returns:
        int: 可用的端口号，如果找不到返回None
    """
    for i in range(max_attempts):
        port = start_port + i
        if check_port_available(port):
            return port
    return None


def get_port_process_info(port):
    """获取占用端口的进程信息
    
    Args:
        port (int): 端口号
        
    Returns:
        str: 进程信息字符串，如果获取失败返回None
    """
    try:
        if os.name == 'nt':  # Windows系统
            # 使用netstat命令查找占用端口的进程
            result = subprocess.run(
                ['netstat', '-ano'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if f':{port}' in line and 'LISTENING' in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            pid = parts[-1]
                            try:
                                # 获取进程名称
                                tasklist_result = subprocess.run(
                                    ['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV'],
                                    capture_output=True,
                                    text=True,
                                    timeout=5
                                )
                                if tasklist_result.returncode == 0:
                                    lines = tasklist_result.stdout.strip().split('\n')
                                    if len(lines) >= 2:
                                        # 解析CSV格式的输出
                                        process_line = lines[1].replace('"', '').split(',')
                                        if len(process_line) >= 1:
                                            process_name = process_line[0]
                                            return f"进程: {process_name} (PID: {pid})"
                                return f"PID: {pid}"
                            except:
                                return f"PID: {pid}"
        else:  # Linux/Mac系统
            result = subprocess.run(
                ['lsof', '-i', f':{port}'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    # 解析lsof输出
                    parts = lines[1].split()
                    if len(parts) >= 2:
                        process_name = parts[0]
                        pid = parts[1]
                        return f"进程: {process_name} (PID: {pid})"
                        
    except Exception as e:
        pass
        
    return None


def main():
    """主函数"""
    
    # 解析参数
    args = _parse_args()
    
    # 检查是否在分布式环境中运行
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    
    # 只有主进程(rank 0)才启动Web服务
    if world_size > 1 and rank != 0:
        print(f"🔄 进程 {rank} 正在等待主进程启动Web服务...")
        # 非主进程只需要初始化分布式生成器，不启动Web服务
        try:
            generator = DistributedMultiTalkGenerator(args)
            print(f"✅ 进程 {rank} 分布式生成器初始化完成")
            # 保持进程运行，等待分布式任务
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n🔚 进程 {rank} 已关闭")
        except Exception as e:
            print(f"❌ 进程 {rank} 发生错误: {e}")
        return
    
    # 主进程执行端口检查和Web服务启动
    print(f"🌟 主进程 (rank {rank}) 正在启动Web服务...")
    
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
    
    # 首先检查端口可用性，避免在模型加载后才发现端口被占用
    print("🔍 检查端口可用性...")
    if not check_port_available(args.server_port):
        print(f"❌ 端口 {args.server_port} 已被占用！")
        
        # 获取占用端口的进程信息
        process_info = get_port_process_info(args.server_port)
        if process_info:
            print(f"📋 占用端口的进程: {process_info}")
        
        # 尝试寻找可用端口
        print("🔍 正在寻找可用端口...")
        available_port = find_available_port(args.server_port, max_attempts=20)
        
        if available_port:
            print(f"✅ 找到可用端口: {available_port}")
            print(f"📝 建议使用以下命令重新启动:")
            
            # 构建建议的启动命令
            if hasattr(args, 'ulysses_size') and args.ulysses_size > 1:
                cmd_parts = [
                    "torchrun",
                    f"--nproc_per_node={args.ulysses_size}",
                    "--master_port=29500",
                    "distributed_multitalk_app.py",
                    f"--ulysses_size={args.ulysses_size}",
                    f"--ring_size={args.ring_size}"
                ]
                
                if args.t5_fsdp:
                    cmd_parts.append("--t5_fsdp")
                if args.dit_fsdp:
                    cmd_parts.append("--dit_fsdp")
                    
                cmd_parts.append(f"--server_port={available_port}")
                
                if hasattr(args, 'num_persistent_param_in_dit') and args.num_persistent_param_in_dit is not None:
                    cmd_parts.append(f"--num_persistent_param_in_dit={args.num_persistent_param_in_dit}")
                    
                print(f"   {' '.join(cmd_parts)}")
            else:
                print(f"   python distributed_multitalk_app.py --server_port={available_port}")
                
            print("\n❓ 是否使用可用端口继续启动？(y/n): ", end="")
            try:
                choice = input().strip().lower()
                if choice in ['y', 'yes', '是', '']:
                    args.server_port = available_port
                    print(f"✅ 已更新服务端口为: {available_port}")
                else:
                    print("❌ 用户取消启动")
                    sys.exit(1)
            except KeyboardInterrupt:
                print("\n❌ 用户中断启动")
                sys.exit(1)
        else:
            print(f"❌ 无法找到可用端口 (尝试范围: {args.server_port}-{args.server_port+19})")
            print("💡 建议:")
            print("   1. 检查并关闭占用端口的程序")
            print("   2. 手动指定其他端口号")
            print("   3. 重启系统释放端口")
            sys.exit(1)
    else:
        print(f"✅ 端口 {args.server_port} 可用")
    
    # 显示配置信息
    print("\n配置信息:")
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