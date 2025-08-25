#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试模式启动脚本
用于快速启动MultiTalk的调试模式，跳过模型加载过程
"""

import os
import sys
import argparse
from pathlib import Path

# 设置调试模式环境变量
os.environ['DEBUG_MODE'] = '1'
os.environ['MOCK_MODEL_OUTPUTS'] = '1'

def create_debug_args():
    """创建调试用的参数"""
    class DebugArgs:
        def __init__(self):
            # 基本参数
            self.task = 'multitalk_512'
            self.base_seed = 42
            self.ulysses_size = 1
            self.ring_size = 1
            
            # 模型路径（调试模式下不会实际加载）
            self.ckpt_dir = 'weights/multitalk'
            self.quant_dir = None
            self.wav2vec_dir = 'weights/wav2vec2'
            self.lora_dir = None
            self.lora_scale = []
            
            # FSDP设置
            self.t5_fsdp = False
            self.dit_fsdp = False
            self.t5_cpu = False
            self.quant = False
            
            # VRAM管理
            self.num_persistent_param_in_dit = None
            
            # 音频保存目录
            self.audio_save_dir = 'debug_output/audio'
            
            # 输出目录
            self.output_dir = 'debug_output'
            
    return DebugArgs()

def main():
    """主函数"""
    print("[DEBUG] 启动MultiTalk调试模式")
    print("[DEBUG] 所有模型将被Mock对象替代，不会进行实际的模型加载")
    
    # 创建输出目录
    os.makedirs('debug_output', exist_ok=True)
    os.makedirs('debug_output/audio', exist_ok=True)
    
    # 导入并启动生成器
    try:
        from distributed_generator import DistributedMultiTalkGenerator
        
        args = create_debug_args()
        generator = DistributedMultiTalkGenerator(args)
        
        print("[DEBUG] 调试模式初始化完成！")
        print("[DEBUG] 现在可以调用generator的各种方法进行调试")
        
        # 示例调用
        print("\n[DEBUG] 示例调用:")
        print("generator.generate_tts_audio('(s1)你好世界(s2)Hello World', 'voice1.pt', 'voice2.pt')")
        
        # 返回生成器实例供交互使用
        return generator
        
    except Exception as e:
        print(f"[ERROR] 调试模式启动失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    generator = main()
    
    # 如果成功创建生成器，进入交互模式
    if generator:
        print("\n[DEBUG] 进入交互模式，generator变量可用")
        print("[DEBUG] 使用 Ctrl+C 退出")
        
        try:
            # 保持脚本运行，允许用户交互
            import code
            code.interact(local=locals())
        except KeyboardInterrupt:
            print("\n[DEBUG] 退出调试模式")