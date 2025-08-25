#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试模式使用示例
展示如何在调试模式下快速测试MultiTalk的各种功能
"""

import os
import sys
import tempfile
from pathlib import Path

# 设置调试模式（必须在导入其他模块之前）
os.environ['DEBUG_MODE'] = '1'
os.environ['MOCK_MODEL_OUTPUTS'] = '1'

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    try:
        from distributed_generator import DistributedMultiTalkGenerator
        from debug_run import create_debug_args
        
        # 创建生成器
        args = create_debug_args()
        generator = DistributedMultiTalkGenerator(args)
        
        print("✅ 生成器初始化成功（调试模式）")
        print(f"调试模式状态: {generator.debug_mode}")
        
        return generator
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def example_tts_generation(generator):
    """TTS生成示例"""
    print("\n=== TTS生成示例 ===")
    
    if not generator:
        print("❌ 生成器不可用")
        return
    
    try:
        # 创建临时语音文件
        import torch
        temp_dir = Path(tempfile.mkdtemp())
        
        voice1_path = temp_dir / 'voice1.pt'
        voice2_path = temp_dir / 'voice2.pt'
        
        # 保存假的语音向量
        torch.save(torch.randn(256), voice1_path)
        torch.save(torch.randn(256), voice2_path)
        
        # 测试对话生成
        dialogue = "(s1)你好，欢迎使用MultiTalk调试模式！(s2)Hello, welcome to MultiTalk debug mode!(s1)这样可以快速测试代码逻辑。"
        
        print(f"输入对话: {dialogue}")
        result = generator.generate_tts_audio(dialogue, str(voice1_path), str(voice2_path))
        
        print("✅ TTS生成成功")
        print(f"结果键: {list(result.keys())}")
        print(f"S1音频形状: {result['s1_audio'].shape}")
        print(f"S2音频形状: {result['s2_audio'].shape}")
        
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return result
        
    except Exception as e:
        print(f"❌ TTS生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def example_audio_embedding(generator):
    """音频嵌入示例"""
    print("\n=== 音频嵌入示例 ===")
    
    if not generator:
        print("❌ 生成器不可用")
        return
    
    try:
        import numpy as np
        
        # 创建假的音频数据（1秒，16kHz）
        audio_data = np.random.randn(16000).astype(np.float32)
        
        print(f"输入音频形状: {audio_data.shape}")
        embedding = generator.get_audio_embedding(audio_data)
        
        print("✅ 音频嵌入生成成功")
        print(f"嵌入形状: {embedding.shape}")
        print(f"嵌入数据类型: {embedding.dtype}")
        
        return embedding
        
    except Exception as e:
        print(f"❌ 音频嵌入失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def example_image_processing(generator):
    """图像处理示例"""
    print("\n=== 图像处理示例 ===")
    
    if not generator:
        print("❌ 生成器不可用")
        return
    
    try:
        from PIL import Image
        import numpy as np
        import tempfile
        
        # 创建假的图像文件
        temp_dir = Path(tempfile.mkdtemp())
        
        # 创建两个假的人物图像
        img1 = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        
        img1_path = temp_dir / 'person1.jpg'
        img2_path = temp_dir / 'person2.jpg'
        
        img1.save(img1_path)
        img2.save(img2_path)
        
        print(f"创建图像: {img1_path}, {img2_path}")
        
        # 测试图像合成
        composite = generator.create_composite_image(str(img1_path), str(img2_path))
        
        print("✅ 图像合成成功")
        print(f"合成图像键: {list(composite.keys())}")
        
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return composite
        
    except Exception as e:
        print(f"❌ 图像处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def example_performance_comparison():
    """性能对比示例"""
    print("\n=== 性能对比示例 ===")
    
    import time
    
    # 测试调试模式启动时间
    print("测试调试模式启动时间...")
    start_time = time.time()
    
    try:
        from distributed_generator import DistributedMultiTalkGenerator
        from debug_run import create_debug_args
        
        args = create_debug_args()
        generator = DistributedMultiTalkGenerator(args)
        
        init_time = time.time() - start_time
        print(f"✅ 调试模式初始化时间: {init_time:.2f} 秒")
        
        # 测试TTS生成时间
        import torch
        import tempfile
        
        temp_dir = Path(tempfile.mkdtemp())
        voice1_path = temp_dir / 'voice1.pt'
        voice2_path = temp_dir / 'voice2.pt'
        
        torch.save(torch.randn(256), voice1_path)
        torch.save(torch.randn(256), voice2_path)
        
        dialogue = "(s1)性能测试对话(s2)Performance test dialogue"
        
        start_time = time.time()
        result = generator.generate_tts_audio(dialogue, str(voice1_path), str(voice2_path))
        tts_time = time.time() - start_time
        
        print(f"✅ TTS生成时间: {tts_time:.2f} 秒")
        
        # 清理
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {
            'init_time': init_time,
            'tts_time': tts_time
        }
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return None

def main():
    """主函数"""
    print("MultiTalk 调试模式示例")
    print("=" * 50)
    
    # 基本使用
    generator = example_basic_usage()
    
    if generator:
        # TTS生成
        tts_result = example_tts_generation(generator)
        
        # 音频嵌入
        audio_embedding = example_audio_embedding(generator)
        
        # 图像处理
        image_result = example_image_processing(generator)
        
        # 性能对比
        perf_result = example_performance_comparison()
        
        print("\n=== 总结 ===")
        print("✅ 所有示例运行完成")
        print("调试模式让你可以:")
        print("  - 快速启动和测试")
        print("  - 跳过模型加载时间")
        print("  - 验证代码逻辑")
        print("  - 进行单元测试")
        
        if perf_result:
            print(f"\n性能提升:")
            print(f"  - 初始化时间: {perf_result['init_time']:.2f}s (vs 数分钟的模型加载)")
            print(f"  - TTS生成时间: {perf_result['tts_time']:.2f}s (vs 数秒的真实推理)")
    
    else:
        print("❌ 示例运行失败")
        print("请检查:")
        print("  - 是否正确设置了调试模式环境变量")
        print("  - 是否存在导入错误")
        print("  - 是否缺少依赖包")

if __name__ == '__main__':
    main()