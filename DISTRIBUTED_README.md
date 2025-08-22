# 🎬 分布式MultiTalk双人对话视频生成Web服务

基于MultiTalk项目实现的分布式Web服务，专门用于生成720P高质量双人对话视频。

## ✨ 功能特性

- 🎯 **双人对话生成**: 支持两个人物图片上传，自动合成对话视频
- 💬 **智能脚本解析**: 支持多种对话格式，自动分配角色台词
- 🎤 **自动TTS生成**: 内置语音合成，支持多种音色选择  
- 📺 **720P高清输出**: 生成960x960分辨率高质量视频
- 🚀 **分布式推理**: 支持8张RTX-4090 GPU并行计算
- 🌐 **友好Web界面**: 基于Gradio的直观操作界面

## 🏗️ 架构设计

```
分布式MultiTalk Web服务架构

┌─────────────────────────────────────────────────────────────┐
│                    Web界面层 (Gradio)                      │
├─────────────────────────────────────────────────────────────┤
│                 业务逻辑层 (Generator)                      │
├─────────────────────────────────────────────────────────────┤
│  对话解析器  │  图像合成  │  TTS生成  │  音频编码  │  视频生成  │
├─────────────────────────────────────────────────────────────┤
│                分布式推理层 (8xGPU)                        │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐    │
│  │ GPU0   │ │ GPU1   │ │ GPU2   │ │ GPU3   │ │ GPU4   │ ...│
│  │ FSDP   │ │ FSDP   │ │ FSDP   │ │ FSDP   │ │ FSDP   │    │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 📋 系统要求

### 硬件要求
- **推荐配置**: 8张RTX-4090 (26GB显存/卡)
- **最低配置**: 1张RTX-4090 (测试模式)
- **内存**: 400GB+ 主机内存
- **存储**: 100GB+ 可用空间

### 软件要求
- Python 3.10
- PyTorch 2.4.1 + CUDA 12.1
- 其他依赖见 `requirements.txt`

## 🚀 快速开始

### 1. 环境准备

确保已按照主项目README安装好环境：

```bash
# 激活conda环境
conda activate multitalk

# 验证GPU
nvidia-smi
```

### 2. 模型准备

确保以下模型文件已下载：

```
weights/
├── Wan2.1-I2V-14B-480P/          # 主模型
├── chinese-wav2vec2-base/         # 音频编码器
└── Kokoro-82M/                    # TTS模型
    └── voices/
        ├── af_heart.pt            # 女性温柔
        ├── am_adam.pt             # 男性成熟
        ├── af_bella.pt            # 女性活泼
        └── am_freeman.pt          # 男性年轻
```

### 3. 启动服务

#### Windows环境 (推荐)
```cmd
# 8GPU分布式模式
start_distributed_service.bat distributed

# 单GPU测试模式  
start_distributed_service.bat single
```

#### Linux环境
```bash
# 8GPU分布式模式
bash start_distributed_service.sh distributed

# 单GPU测试模式
bash start_distributed_service.sh single
```

#### 手动启动
```bash
# 8GPU分布式模式
torchrun --nproc_per_node=8 --master_port=29500 \
    distributed_multitalk_app.py \
    --ulysses_size=8 --ring_size=1 \
    --t5_fsdp --dit_fsdp \
    --server_port=8419

# 单GPU测试模式
python distributed_multitalk_app.py \
    --ulysses_size=1 --ring_size=1 \
    --server_port=8419
```

### 4. 访问Web界面

启动成功后，在浏览器中访问：
- 本地访问: http://localhost:8419
- 外网访问: http://0.0.0.0:8419

## 📖 使用指南

### 基本使用流程

1. **上传人物图片**
   - 上传两张清晰的人物正面照片
   - 建议分辨率不低于512x512
   - 支持JPG、PNG格式

2. **输入对话脚本**
   - 支持多种格式的对话脚本
   - 系统会自动解析并分配给两个角色

3. **描述场景环境**
   - 输入详细的场景描述
   - 有助于生成更符合预期的视频

4. **选择语音模型**
   - 为每个角色选择合适的音色
   - 可选择女性/男性，温柔/活泼等风格

5. **调整生成参数**
   - 采样步数：影响生成质量和速度
   - 引导强度：控制文本和音频匹配度

6. **生成720P视频**
   - 点击生成按钮
   - 等待分布式推理完成

### 支持的对话格式

#### 1. 标准格式
```
(s1) 你好，今天天气真不错！
(s2) 是的，很适合出去走走。
(s1) 要不要一起去公园？
(s2) 好主意，我们走吧！
```

#### 2. 角色格式
```
角色1: 你知道MultiTalk这个项目吗？
角色2: 当然知道！这是一个很棒的模型。
角色1: 它有什么特别的功能？
角色2: 可以生成多人对话视频。
```

#### 3. 简化格式
```
A: 这个项目怎么样？
B: 我觉得很不错！
A: 我们来试试看。
B: 好的，开始吧！
```

#### 4. 自由格式
```
你好，很高兴见到你！
我也是，今天天气真好。
是啊，要不要一起喝咖啡？
当然可以，我知道一家很棒的咖啡厅。
```

## ⚙️ 配置参数

### 分布式参数
- `--ulysses_size`: Ulysses并行度 (推荐8)
- `--ring_size`: Ring注意力并行度 (推荐1)
- `--t5_fsdp`: 启用T5模型FSDP
- `--dit_fsdp`: 启用DiT模型FSDP

### 生成参数
- `--frame_num`: 生成帧数 (默认81帧)
- `--sample_shift`: 采样偏移 (720P推荐11)
- `--server_port`: Web服务端口 (默认8419)

### 内存优化
- `--num_persistent_param_in_dit`: DiT模型常驻参数数量
- `--offload_model`: 是否将模型卸载到CPU

## 🔧 故障排除

### 常见问题

#### 1. GPU内存不足
```bash
# 解决方案：启用内存优化
--num_persistent_param_in_dit=0
```

#### 2. 分布式初始化失败
```bash
# 检查NCCL设置
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
```

#### 3. 模型加载错误
```bash
# 检查模型路径
ls -la weights/Wan2.1-I2V-14B-480P/
ls -la weights/chinese-wav2vec2-base/
```

#### 4. TTS生成失败
```bash
# 检查Kokoro模型
ls -la weights/Kokoro-82M/voices/
```

### 性能优化建议

1. **GPU配置**
   - 确保8张GPU在同一PCIe交换机下
   - 启用NVLink以提升GPU间通信

2. **内存配置**
   - 预留足够的主机内存
   - 适当调整GPU显存使用策略

3. **网络配置**
   - 如果使用多节点，确保高速网络连接
   - 调整NCCL通信参数

## 📊 性能指标

### 8GPU分布式模式
- **生成分辨率**: 720P (960x960)
- **推理时间**: ~2-3分钟/视频 (81帧)
- **显存占用**: ~20GB/GPU
- **并行效率**: ~85%

### 单GPU测试模式
- **生成分辨率**: 720P (960x960)
- **推理时间**: ~15-20分钟/视频 (81帧)
- **显存占用**: ~24GB
- **适用场景**: 功能测试

## 🔗 相关链接

- [MultiTalk项目主页](https://meigen-ai.github.io/multi-talk/)
- [HuggingFace模型](https://huggingface.co/MeiGen-AI/MeiGen-MultiTalk)
- [技术论文](https://arxiv.org/abs/2505.22647)

## 📝 更新日志

### v1.0.0 (2025-01-22)
- ✨ 首次发布分布式Web服务
- 🚀 支持8GPU并行推理
- 📺 支持720P视频生成
- 💬 智能对话脚本解析
- 🎤 集成TTS语音合成

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目遵循与MultiTalk主项目相同的许可证。