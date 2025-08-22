@echo off
REM 分布式MultiTalk双人对话视频生成服务启动脚本 (Windows版本)
REM 适用于8张RTX-4090 GPU环境

echo ============================================================
echo 🎬 分布式MultiTalk双人对话视频生成服务启动脚本
echo ============================================================

REM 检查参数
if "%1"=="--help" goto :help
if "%1"=="-h" goto :help
if "%1"=="/?" goto :help

REM 设置模式
set MODE=%1
if "%MODE%"=="" set MODE=distributed

REM 设置环境变量
set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set NCCL_IB_DISABLE=1
set NCCL_P2P_DISABLE=1

REM 检查必要文件
echo 🔍 检查必要文件...

if not exist "distributed_multitalk_app.py" (
    echo ❌ 缺少文件: distributed_multitalk_app.py
    exit /b 1
)
if not exist "distributed_generator.py" (
    echo ❌ 缺少文件: distributed_generator.py
    exit /b 1
)
if not exist "distributed_web_interface.py" (
    echo ❌ 缺少文件: distributed_web_interface.py
    exit /b 1
)
if not exist "distributed_multitalk_core.py" (
    echo ❌ 缺少文件: distributed_multitalk_core.py
    exit /b 1
)

echo ✅ 文件检查完成

REM 检查模型目录
echo 🔍 检查模型目录...

if not exist "weights\Wan2.1-I2V-14B-480P" (
    echo ❌ 缺少目录: weights\Wan2.1-I2V-14B-480P
    echo 请确保已下载所有必要的模型文件
    exit /b 1
)
if not exist "weights\chinese-wav2vec2-base" (
    echo ❌ 缺少目录: weights\chinese-wav2vec2-base
    echo 请确保已下载所有必要的模型文件
    exit /b 1
)
if not exist "weights\Kokoro-82M" (
    echo ❌ 缺少目录: weights\Kokoro-82M
    echo 请确保已下载所有必要的模型文件
    exit /b 1
)

echo ✅ 模型目录检查完成

REM 根据模式启动
if "%MODE%"=="single" goto :single
if "%MODE%"=="distributed" goto :distributed

echo ❌ 未知模式: %MODE%
echo 支持的模式: single, distributed
exit /b 1

:single
echo 🔧 启动模式: 单GPU (测试)
echo ⚠️  注意: 单GPU模式仅供测试，生成速度较慢
echo ============================================================

python distributed_multitalk_app.py ^
    --ulysses_size=1 ^
    --ring_size=1 ^
    --server_port=8419

goto :end

:distributed
echo 🚀 启动模式: 8GPU分布式 (推荐)
echo 📊 GPU配置: 8张RTX-4090
echo ⚡ 并行配置: Ulysses=8, Ring=1, FSDP=True
echo 🎯 分辨率: 720P (960x960)
echo ============================================================

torchrun --nproc_per_node=8 ^
    --master_port=29500 ^
    distributed_multitalk_app.py ^
    --ulysses_size=8 ^
    --ring_size=1 ^
    --t5_fsdp ^
    --dit_fsdp ^
    --server_port=8419 ^
    --num_persistent_param_in_dit=0

goto :end

:help
echo 使用方法:
echo   %0 [single^|distributed]
echo.
echo 参数说明:
echo   single       - 单GPU模式 (测试用)
echo   distributed  - 8GPU分布式模式 (推荐)
echo.
echo 示例:
echo   %0 single       # 单GPU测试
echo   %0 distributed  # 8GPU分布式
echo   %0              # 默认8GPU分布式
exit /b 0

:end
echo ============================================================
echo 🔚 服务已停止
pause