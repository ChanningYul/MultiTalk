#!/usr/bin/env python3
"""
分布式MultiTalk Web服务测试脚本

测试项目：
1. 对话脚本解析功能
2. 参数配置验证
3. 模块导入测试
4. Web界面创建测试
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_dialogue_parser():
    """测试对话脚本解析功能"""
    print("🔍 测试对话脚本解析功能...")
    
    try:
        from distributed_multitalk_core import DialogueScriptParser
        
        parser = DialogueScriptParser()
        
        # 测试不同格式的对话
        test_cases = [
            # 标准格式
            "(s1) 你好！ (s2) 很高兴见到你！",
            
            # 角色格式
            "角色1: 今天天气真好 角色2: 是啊，很适合出门",
            
            # A/B格式
            "A: 这个项目怎么样？ B: 我觉得很不错！",
            
            # 自由格式
            "你好，今天天气真好啊！\n是的，很适合出去走走。\n要不要一起去公园？\n好主意，我们走吧！"
        ]
        
        for i, test_input in enumerate(test_cases, 1):
            result = parser.parse_dialogue(test_input)
            print(f"   测试 {i}: ✅")
            print(f"   输入: {test_input[:50]}...")
            print(f"   输出: {result}")
            print()
        
        print("✅ 对话脚本解析功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 对话脚本解析功能测试失败: {e}")
        return False


def test_args_parsing():
    """测试参数解析功能"""
    print("🔍 测试参数解析功能...")
    
    try:
        # 临时修改sys.argv来模拟命令行参数
        original_argv = sys.argv.copy()
        sys.argv = [
            "test_script.py",
            "--task", "multitalk-14B",
            "--size", "multitalk-720", 
            "--ulysses_size", "8",
            "--server_port", "8419"
        ]
        
        from distributed_multitalk_core import _parse_args
        args = _parse_args()
        
        # 验证关键参数
        assert args.task == "multitalk-14B"
        assert args.size == "multitalk-720"
        assert args.ulysses_size == 8
        assert args.server_port == 8419
        
        # 恢复原始argv
        sys.argv = original_argv
        
        print("✅ 参数解析功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 参数解析功能测试失败: {e}")
        return False


def test_module_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        # 测试核心模块
        from distributed_multitalk_core import DialogueScriptParser, _parse_args
        print("   ✅ distributed_multitalk_core")
        
        # 测试生成器模块 (可能因为缺少模型而失败，但至少语法要正确)
        try:
            from distributed_generator import DistributedMultiTalkGenerator
            print("   ✅ distributed_generator")
        except Exception as e:
            if "No module named" in str(e) or "cannot import" in str(e):
                print(f"   ⚠️  distributed_generator (依赖问题，但语法正确): {e}")
            else:
                print(f"   ❌ distributed_generator: {e}")
                return False
        
        # 测试Web界面模块
        from distributed_web_interface import create_gradio_interface
        print("   ✅ distributed_web_interface")
        
        print("✅ 模块导入测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 模块导入测试失败: {e}")
        return False


def test_gradio_interface():
    """测试Gradio界面创建"""
    print("🔍 测试Gradio界面创建...")
    
    try:
        import gradio as gr
        from distributed_web_interface import create_gradio_interface
        
        # 创建一个模拟的生成器对象
        class MockGenerator:
            def generate_video(self, **kwargs):
                return "test_video.mp4"
        
        mock_generator = MockGenerator()
        
        # 创建界面（不启动）
        demo = create_gradio_interface(mock_generator)
        
        # 验证界面对象
        assert hasattr(demo, 'launch'), "Demo对象应该有launch方法"
        
        print("✅ Gradio界面创建测试通过")
        return True
        
    except Exception as e:
        print(f"❌ Gradio界面创建测试失败: {e}")
        return False


def test_file_structure():
    """测试文件结构"""
    print("🔍 测试文件结构...")
    
    required_files = [
        "distributed_multitalk_core.py",
        "distributed_generator.py", 
        "distributed_web_interface.py",
        "distributed_multitalk_app.py",
        "start_distributed_service.sh",
        "start_distributed_service.bat",
        "DISTRIBUTED_README.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"   ✅ {file_path}")
    
    if missing_files:
        print(f"❌ 缺少文件: {missing_files}")
        return False
    
    print("✅ 文件结构测试通过")
    return True


def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 分布式MultiTalk Web服务测试")
    print("=" * 60)
    
    tests = [
        ("文件结构", test_file_structure),
        ("模块导入", test_module_imports),
        ("参数解析", test_args_parsing),
        ("对话解析", test_dialogue_parser),
        ("Gradio界面", test_gradio_interface),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}测试:")
        print("-" * 40)
        
        if test_func():
            passed += 1
        
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print(f"🎯 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！分布式Web服务可以启动。")
        print("\n📝 启动建议:")
        print("1. 确保已下载所有必要的模型文件")
        print("2. 使用启动脚本启动服务:")
        print("   Windows: start_distributed_service.bat distributed")
        print("   Linux:   bash start_distributed_service.sh distributed")
        print("3. 在浏览器中访问: http://localhost:8419")
    else:
        print("⚠️  部分测试未通过，请检查相关问题后再启动服务。")
    
    print("=" * 60)


if __name__ == "__main__":
    main()