#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试模式单元测试
用于测试各个模块在调试模式下的功能
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path

# 设置调试模式
os.environ['DEBUG_MODE'] = '1'
os.environ['MOCK_MODEL_OUTPUTS'] = '1'

from debug_config import DebugConfig, MockWav2VecModel, MockWav2VecFeatureExtractor, MockMultiTalkPipeline, MockKPipeline

class TestDebugMode(unittest.TestCase):
    """调试模式测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_debug_config(self):
        """测试调试配置"""
        self.assertTrue(DebugConfig.is_debug_mode())
        self.assertTrue(DebugConfig.should_mock_outputs())
        
    def test_mock_wav2vec_model(self):
        """测试Mock Wav2Vec模型"""
        model = MockWav2VecModel(device='cpu')
        
        # 测试模型属性
        self.assertTrue(hasattr(model, 'feature_extractor'))
        self.assertTrue(hasattr(model.feature_extractor, '_freeze_parameters'))
        
        # 测试前向传播
        import torch
        dummy_input = torch.randn(1, 1000)
        output = model(dummy_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 63, 1024))  # 预期的输出形状
        
    def test_mock_feature_extractor(self):
        """测试Mock特征提取器"""
        extractor = MockWav2VecFeatureExtractor()
        
        # 测试处理音频
        import numpy as np
        dummy_audio = np.random.randn(16000)  # 1秒的音频
        result = extractor(dummy_audio, sampling_rate=16000, return_tensors='pt')
        
        self.assertIn('input_values', result)
        self.assertEqual(result['input_values'].shape[1], 1000)  # 预期的输入长度
        
    def test_mock_multitalk_pipeline(self):
        """测试Mock MultiTalk管道"""
        pipeline = MockMultiTalkPipeline(device='cpu')
        
        # 测试属性
        self.assertFalse(pipeline.vram_management)
        
        # 测试方法
        pipeline.enable_vram_management(num_persistent_param_in_dit=100)
        self.assertTrue(pipeline.vram_management)
        
        # 测试生成
        import torch
        dummy_inputs = {
            'prompt': 'test prompt',
            'audio_embedding': torch.randn(1, 100, 512),
            'image': torch.randn(1, 3, 512, 512)
        }
        
        result = pipeline.generate(**dummy_inputs)
        self.assertIn('video', result)
        self.assertIsInstance(result['video'], torch.Tensor)
        
    def test_mock_kpipeline(self):
        """测试Mock K管道"""
        pipeline = MockKPipeline()
        
        # 测试TTS生成
        import torch
        voice_tensor = torch.randn(256)  # 假的语音向量
        
        generator = pipeline('Hello world', voice=voice_tensor, speed=1.0)
        
        # 测试生成器
        results = list(generator)
        self.assertGreater(len(results), 0)
        
        for gs, ps, audio in results:
            self.assertIsInstance(audio, torch.Tensor)
            self.assertEqual(len(audio.shape), 1)  # 1D音频
            
class TestDistributedGenerator(unittest.TestCase):
    """测试分布式生成器"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # 创建测试参数
        class TestArgs:
            def __init__(self):
                self.task = 'multitalk_512'
                self.base_seed = 42
                self.ulysses_size = 1
                self.ring_size = 1
                self.ckpt_dir = 'weights/multitalk'
                self.quant_dir = None
                self.wav2vec_dir = 'weights/wav2vec2'
                self.lora_dir = None
                self.lora_scale = []
                self.t5_fsdp = False
                self.dit_fsdp = False
                self.t5_cpu = False
                self.quant = False
                self.num_persistent_param_in_dit = None
                self.audio_save_dir = str(self.temp_dir / 'audio')
                
        self.args = TestArgs()
        
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_generator_initialization(self):
        """测试生成器初始化"""
        try:
            from distributed_generator import DistributedMultiTalkGenerator
            generator = DistributedMultiTalkGenerator(self.args)
            
            # 验证调试模式
            self.assertTrue(generator.debug_mode)
            
            # 验证Mock对象
            self.assertIsInstance(generator.wan_pipeline, MockMultiTalkPipeline)
            self.assertIsInstance(generator.audio_encoder, MockWav2VecModel)
            self.assertIsInstance(generator.wav2vec_feature_extractor, MockWav2VecFeatureExtractor)
            
        except ImportError:
            self.skipTest("distributed_generator模块不可用")
            
    def test_tts_generation(self):
        """测试TTS生成"""
        try:
            from distributed_generator import DistributedMultiTalkGenerator
            generator = DistributedMultiTalkGenerator(self.args)
            
            # 创建假的语音文件路径
            voice1_path = self.temp_dir / 'voice1.pt'
            voice2_path = self.temp_dir / 'voice2.pt'
            
            # 创建假的语音张量文件
            import torch
            torch.save(torch.randn(256), voice1_path)
            torch.save(torch.randn(256), voice2_path)
            
            # 测试TTS生成
            dialogue = "(s1)你好世界(s2)Hello World"
            result = generator.generate_tts_audio(dialogue, str(voice1_path), str(voice2_path))
            
            self.assertIn('s1_audio', result)
            self.assertIn('s2_audio', result)
            self.assertIsInstance(result['s1_audio'], torch.Tensor)
            self.assertIsInstance(result['s2_audio'], torch.Tensor)
            
        except ImportError:
            self.skipTest("distributed_generator模块不可用")

def run_tests():
    """运行所有测试"""
    print("[DEBUG] 开始运行调试模式测试...")
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestDebugMode))
    suite.addTests(loader.loadTestsFromTestCase(TestDistributedGenerator))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果
    if result.wasSuccessful():
        print("\n[DEBUG] 所有测试通过！")
    else:
        print(f"\n[DEBUG] 测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")
        
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)