# MultiTalk 调试模式使用指南

## 概述

为了解决每次修改代码后都需要长时间加载模型的问题，我们实现了一个轻量级的调试模式。在调试模式下，所有重型模型都会被Mock对象替代，大大加快了启动和测试速度。

## 快速开始

### 方法1: 使用调试启动脚本

```bash
python debug_run.py
```

这将启动一个交互式调试环境，你可以直接使用 `generator` 变量进行测试。

### 方法2: 设置环境变量

```bash
# Windows PowerShell
$env:DEBUG_MODE = "1"
$env:MOCK_MODEL_OUTPUTS = "1"
python distributed_multitalk_app.py

# Windows CMD
set DEBUG_MODE=1
set MOCK_MODEL_OUTPUTS=1
python distributed_multitalk_app.py

# Linux/Mac
export DEBUG_MODE=1
export MOCK_MODEL_OUTPUTS=1
python distributed_multitalk_app.py
```

### 方法3: 在代码中设置

```python
import os
os.environ['DEBUG_MODE'] = '1'
os.environ['MOCK_MODEL_OUTPUTS'] = '1'

# 然后正常导入和使用
from distributed_generator import DistributedMultiTalkGenerator
```

## 调试模式功能

### Mock 组件

调试模式会替换以下重型组件：

1. **MockWav2VecModel**: 替代 Wav2Vec2 音频编码器
2. **MockWav2VecFeatureExtractor**: 替代音频特征提取器
3. **MockMultiTalkPipeline**: 替代 MultiTalk 主管道
4. **MockKPipeline**: 替代 TTS 管道

### 输出特点

- **快速启动**: 跳过所有模型加载，几秒内完成初始化
- **模拟输出**: 生成符合预期格式的假数据
- **保持接口**: 所有方法调用保持原有接口不变
- **调试日志**: 详细的调试信息输出

## 测试和验证

### 运行单元测试

```bash
python test_debug.py
```

这将运行所有调试模式相关的单元测试，验证Mock组件的正确性。

### 手动测试示例

```python
# 启动调试模式
import os
os.environ['DEBUG_MODE'] = '1'

from distributed_generator import DistributedMultiTalkGenerator
from debug_run import create_debug_args

# 创建生成器
args = create_debug_args()
generator = DistributedMultiTalkGenerator(args)

# 测试TTS生成
result = generator.generate_tts_audio(
    "(s1)你好世界(s2)Hello World", 
    "voice1.pt", 
    "voice2.pt"
)
print(f"TTS结果: {result.keys()}")

# 测试音频嵌入
import numpy as np
audio = np.random.randn(16000)  # 1秒音频
embedding = generator.get_audio_embedding(audio)
print(f"音频嵌入形状: {embedding.shape}")
```

## 配置选项

### 环境变量

- `DEBUG_MODE`: 设置为 "1" 启用调试模式
- `MOCK_MODEL_OUTPUTS`: 设置为 "1" 启用Mock输出
- `DEBUG_LOG_LEVEL`: 设置日志级别 (默认: INFO)

### 调试配置类

```python
from debug_config import DebugConfig

# 检查调试模式状态
if DebugConfig.is_debug_mode():
    print("调试模式已启用")

# 检查Mock输出状态
if DebugConfig.should_mock_outputs():
    print("将使用Mock输出")

# 设置调试日志
DebugConfig.setup_debug_logging()
```

## 注意事项

### 限制

1. **仅用于调试**: Mock输出不是真实的模型结果
2. **格式兼容**: 输出格式与真实模型保持一致，但内容是随机生成的
3. **性能测试**: 不能用于性能基准测试
4. **功能验证**: 适合验证代码逻辑，不适合验证模型效果

### 最佳实践

1. **开发阶段**: 使用调试模式快速迭代代码逻辑
2. **单元测试**: 使用调试模式编写和运行单元测试
3. **集成测试**: 在真实环境中进行最终验证
4. **代码审查**: 确保调试代码不会进入生产环境

## 故障排除

### 常见问题

**Q: 调试模式没有生效？**
A: 确保在导入任何模型相关模块之前设置环境变量。

**Q: Mock输出格式不正确？**
A: 检查 `debug_config.py` 中的Mock类实现，确保输出格式匹配。

**Q: 测试失败？**
A: 运行 `python test_debug.py -v` 查看详细错误信息。

### 调试日志

调试模式会输出详细的日志信息：

```
[DEBUG] 调试模式已启用，将使用Mock模型
[DEBUG] 使用Mock音频模型
[DEBUG] 使用Mock MultiTalk管道
[DEBUG] 使用Mock TTS管道
```

## 扩展调试功能

### 添加新的Mock组件

1. 在 `debug_config.py` 中定义新的Mock类
2. 实现必要的方法和属性
3. 在相应的模块中添加调试模式检查
4. 添加相应的单元测试

### 自定义Mock输出

```python
from debug_config import DebugConfig

# 自定义视频输出
DebugConfig.create_debug_video_output(
    output_path="custom_video.mp4",
    duration=10.0,
    fps=25
)

# 自定义音频输出
DebugConfig.create_debug_audio_output(
    output_path="custom_audio.wav",
    duration=5.0,
    sample_rate=22050
)
```

## 总结

调试模式提供了一个轻量级的开发和测试环境，让你可以：

- ✅ 快速启动和测试代码逻辑
- ✅ 跳过耗时的模型加载过程
- ✅ 保持完整的API兼容性
- ✅ 进行单元测试和集成测试
- ✅ 快速迭代和调试功能

记住在生产环境中禁用调试模式，确保使用真实的模型进行推理。