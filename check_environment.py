#!/usr/bin/env python3
"""
分布式MultiTalk环境验证脚本

快速检查部署环境是否准备就绪
"""

import sys
import os
from pathlib import Path
import subprocess

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python版本过低: {version.major}.{version.minor}.{version.micro}")
        print("   需要Python 3.10或更高版本")
        return False

def check_cuda():
    """检查CUDA环境"""
    print("🎮 检查CUDA环境...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"   ✅ 检测到 {gpu_count} 张GPU")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                print(f"   GPU {i}: {gpu_name} ({gpu_memory}GB)")
            return True
        else:
            print("   ⚠️  未检测到CUDA GPU")
            return False
    except ImportError:
        print("   ❌ PyTorch未安装")
        return False

def check_dependencies():
    """检查关键依赖"""
    print("📦 检查关键依赖...")
    
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("transformers", "Transformers"),
        ("gradio", "Gradio"),
        ("librosa", "Librosa"),
        ("soundfile", "SoundFile"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
    ]
    
    missing = []
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name}")
            missing.append(name)
    
    if missing:
        print(f"\n   缺少依赖: {', '.join(missing)}")
        print("   请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_model_files():
    """检查模型文件"""
    print("🧠 检查模型文件...")
    
    required_paths = [
        ("weights/Wan2.1-I2V-14B-480P", "主模型"),
        ("weights/chinese-wav2vec2-base", "音频编码器"),
        ("weights/Kokoro-82M", "TTS模型"),
        ("weights/Kokoro-82M/voices", "语音模型"),
    ]
    
    missing = []
    for path, name in required_paths:
        if Path(path).exists():
            print(f"   ✅ {name}: {path}")
        else:
            print(f"   ❌ {name}: {path}")
            missing.append(name)
    
    # 检查具体语音文件
    voice_files = [
        "weights/Kokoro-82M/voices/af_heart.pt",
        "weights/Kokoro-82M/voices/am_adam.pt", 
        "weights/Kokoro-82M/voices/af_bella.pt",
        "weights/Kokoro-82M/voices/am_freeman.pt",
    ]
    
    for voice_file in voice_files:
        if Path(voice_file).exists():
            print(f"   ✅ 语音模型: {Path(voice_file).name}")
        else:
            print(f"   ❌ 语音模型: {Path(voice_file).name}")
            missing.append(f"语音模型 {Path(voice_file).name}")
    
    return len(missing) == 0

def check_service_files():
    """检查服务文件"""
    print("📁 检查服务文件...")
    
    required_files = [
        "distributed_multitalk_app.py",
        "distributed_generator.py",
        "distributed_web_interface.py", 
        "distributed_multitalk_core.py",
        "start_distributed_service.bat",
        "start_distributed_service.sh",
    ]
    
    missing = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            missing.append(file_path)
    
    return len(missing) == 0

def check_ports():
    """检查端口占用"""
    print("🔌 检查端口状态...")
    
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8419))
        sock.close()
        
        if result == 0:
            print("   ⚠️  端口8419已被占用")
            return False
        else:
            print("   ✅ 端口8419可用")
            return True
    except Exception:
        print("   ⚠️  无法检查端口状态")
        return True

def check_disk_space():
    """检查磁盘空间"""
    print("💾 检查磁盘空间...")
    
    try:
        import shutil
        free_space = shutil.disk_usage('.').free // (1024**3)
        
        if free_space >= 50:
            print(f"   ✅ 可用空间: {free_space}GB")
            return True
        else:
            print(f"   ⚠️  可用空间不足: {free_space}GB (建议至少50GB)")
            return False
    except Exception:
        print("   ⚠️  无法检查磁盘空间")
        return True

def suggest_startup_command():
    """建议启动命令"""
    print("\n🚀 建议的启动命令:")
    print("=" * 50)
    
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        
        if gpu_count >= 8:
            print("8GPU分布式模式（推荐）:")
            print("Windows: start_distributed_service.bat distributed")
            print("Linux:   bash start_distributed_service.sh distributed")
            print("\n手动启动:")
            print("torchrun --nproc_per_node=8 --master_port=29500 \\")
            print("    distributed_multitalk_app.py \\")
            print("    --ulysses_size=8 --ring_size=1 \\")
            print("    --t5_fsdp --dit_fsdp --server_port=8419")
        else:
            print("单GPU测试模式:")
            print("Windows: start_distributed_service.bat single")
            print("Linux:   bash start_distributed_service.sh single")
            print("\n手动启动:")
            print("python distributed_multitalk_app.py \\")
            print("    --ulysses_size=1 --ring_size=1 --server_port=8419")
    except:
        print("请先安装PyTorch后重新检查")

def main():
    """主检查函数"""
    print("=" * 60)
    print("🔍 分布式MultiTalk环境验证")
    print("=" * 60)
    
    checks = [
        ("Python版本", check_python_version),
        ("CUDA环境", check_cuda),
        ("依赖包", check_dependencies),
        ("模型文件", check_model_files),
        ("服务文件", check_service_files),
        ("端口状态", check_ports),
        ("磁盘空间", check_disk_space),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        print("-" * 30)
        
        if check_func():
            passed += 1
            
    print("\n" + "=" * 60)
    print(f"🎯 检查结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 环境检查全部通过！可以启动服务。")
        suggest_startup_command()
        print("\n📱 启动后访问: http://localhost:8419")
    elif passed >= total - 2:
        print("⚠️  大部分检查通过，可以尝试启动服务。")
        suggest_startup_command()
    else:
        print("❌ 环境检查未通过，请解决相关问题后重试。")
        
        print("\n🔧 常见解决方案:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 下载模型: 请参考主项目README")
        print("3. 检查CUDA: nvidia-smi")
        print("4. 释放端口: 关闭占用8419端口的程序")
    
    print("=" * 60)

if __name__ == "__main__":
    main()