import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path
import re

import torch
import torch.distributed as dist
from PIL import Image
import numpy as np
from einops import rearrange
import soundfile as sf
import librosa

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.multitalk_utils import save_video_ffmpeg
from kokoro import KPipeline
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from distributed_multitalk_core import DialogueScriptParser


class DistributedMultiTalkGenerator:
    """分布式MultiTalk视频生成器"""
    
    def __init__(self, args):
        self.args = args
        self.setup_distributed()
        self.setup_models()
        self.dialogue_parser = DialogueScriptParser()
        
        # 创建临时目录
        self.temp_dir = Path(tempfile.gettempdir()) / "multitalk_distributed"
        self.temp_dir.mkdir(exist_ok=True)
    
    def setup_distributed(self):
        """设置分布式环境"""
        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.device = self.local_rank
        
        self._init_logging(self.rank)
        
        if self.args.offload_model is None:
            self.args.offload_model = False if self.world_size > 1 else True
            logging.info(f"offload_model设置为: {self.args.offload_model}")
        
        if self.world_size > 1:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=self.rank,
                world_size=self.world_size
            )
            logging.info(f"分布式环境初始化完成: rank={self.rank}, world_size={self.world_size}")
        else:
            assert not (self.args.t5_fsdp or self.args.dit_fsdp), \
                "单GPU环境不支持FSDP"
            assert not (self.args.ulysses_size > 1 or self.args.ring_size > 1), \
                "单GPU环境不支持并行计算"
        
        if self.args.ulysses_size > 1 or self.args.ring_size > 1:
            assert self.args.ulysses_size * self.args.ring_size == self.world_size, \
                f"ulysses_size * ring_size必须等于world_size"
            from xfuser.core.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )
            init_distributed_environment(
                rank=dist.get_rank(), world_size=dist.get_world_size())

            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=self.args.ring_size,
                ulysses_degree=self.args.ulysses_size,
            )
            logging.info(f"并行计算环境初始化完成: ulysses={self.args.ulysses_size}, ring={self.args.ring_size}")
    
    def setup_models(self):
        """初始化模型"""
        cfg = WAN_CONFIGS[self.args.task]
        
        if self.args.ulysses_size > 1:
            assert cfg.num_heads % self.args.ulysses_size == 0, \
                f"注意力头数{cfg.num_heads}无法被ulysses_size{self.args.ulysses_size}整除"
        
        logging.info(f"生成任务参数: {self.args}")
        logging.info(f"模型配置: {cfg}")
        
        if dist.is_initialized():
            base_seed = [self.args.base_seed] if self.rank == 0 else [None]
            dist.broadcast_object_list(base_seed, src=0)
            self.args.base_seed = base_seed[0]
        
        # 初始化音频相关模型
        self.wav2vec_feature_extractor, self.audio_encoder = self._init_audio_models()
        
        # 创建音频保存目录
        os.makedirs(self.args.audio_save_dir, exist_ok=True)
        
        # 初始化MultiTalk管道
        logging.info("正在创建MultiTalk管道...")
        self.wan_pipeline = wan.MultiTalkPipeline(
            config=cfg,
            checkpoint_dir=self.args.ckpt_dir,
            quant_dir=self.args.quant_dir,
            device_id=self.device,
            rank=self.rank,
            t5_fsdp=self.args.t5_fsdp,
            dit_fsdp=self.args.dit_fsdp,
            use_usp=(self.args.ulysses_size > 1 or self.args.ring_size > 1),
            t5_cpu=self.args.t5_cpu,
            lora_dir=self.args.lora_dir,
            lora_scales=self.args.lora_scale,
            quant=self.args.quant
        )
        
        if self.args.num_persistent_param_in_dit is not None:
            self.wan_pipeline.vram_management = True
            self.wan_pipeline.enable_vram_management(
                num_persistent_param_in_dit=self.args.num_persistent_param_in_dit
            )
        
        logging.info("模型初始化完成！")
    
    def _init_logging(self, rank):
        """初始化日志"""
        if rank == 0:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] %(levelname)s: %(message)s",
                handlers=[logging.StreamHandler(stream=sys.stdout)]
            )
        else:
            logging.basicConfig(level=logging.ERROR)
    
    def _init_audio_models(self):
        """初始化音频相关模型"""
        logging.info("正在加载音频模型...")
        audio_encoder = Wav2Vec2Model.from_pretrained(
            self.args.wav2vec_dir, local_files_only=True
        ).to('cpu')
        audio_encoder.feature_extractor._freeze_parameters()
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.args.wav2vec_dir, local_files_only=True
        )
        logging.info("音频模型加载完成")
        return wav2vec_feature_extractor, audio_encoder
    
    def create_composite_image(self, person1_image_path, person2_image_path):
        """创建包含两个人物的合成图像"""
        try:
            logging.info("正在创建合成图像...")
            
            # 读取两个图像
            img1 = Image.open(person1_image_path).convert('RGB')
            img2 = Image.open(person2_image_path).convert('RGB')
            
            # 调整图像大小，保持宽高比
            target_height = 960  # 720P对应的高度
            
            def resize_keep_ratio(img, target_height):
                ratio = target_height / img.height
                new_width = int(img.width * ratio)
                return img.resize((new_width, target_height), Image.Resampling.LANCZOS)
            
            img1_resized = resize_keep_ratio(img1, target_height)
            img2_resized = resize_keep_ratio(img2, target_height)
            
            # 创建横向拼接的合成图像
            total_width = img1_resized.width + img2_resized.width
            composite = Image.new('RGB', (total_width, target_height))
            
            # 拼接图像
            composite.paste(img1_resized, (0, 0))
            composite.paste(img2_resized, (img1_resized.width, 0))
            
            # 保存合成图像
            composite_path = self.temp_dir / f"composite_{uuid.uuid4().hex}.png"
            composite.save(composite_path)
            
            # 生成bbox信息用于指定人物位置
            bbox_info = {
                "person1": [0, 0, img1_resized.width, target_height],
                "person2": [img1_resized.width, 0, total_width, target_height]
            }
            
            logging.info(f"合成图像创建完成: {composite_path}")
            return str(composite_path), bbox_info
            
        except Exception as e:
            logging.error(f"创建合成图像时发生错误: {e}")
            raise
    
    def generate_tts_audio(self, dialogue_text, voice1_path, voice2_path):
        """生成TTS音频"""
        try:
            logging.info("正在生成TTS音频...")
            
            # 解析对话
            pattern = r'\(s(\d+)\)\s*(.*?)(?=\s*\(s\d+\)|$)'
            matches = re.findall(pattern, dialogue_text, re.DOTALL)
            
            s1_sentences = []
            s2_sentences = []
            
            pipeline = KPipeline(lang_code='a', repo_id='weights/Kokoro-82M')
            
            for speaker, content in matches:
                content = content.strip()
                if not content:
                    continue
                    
                if speaker == '1':
                    voice_tensor = torch.load(voice1_path, weights_only=True)
                    generator = pipeline(content, voice=voice_tensor, speed=1, split_pattern=r'\n+')
                    audios = []
                    for gs, ps, audio in generator:
                        audios.append(audio)
                    audios = torch.concat(audios, dim=0)
                    s1_sentences.append(audios)
                    s2_sentences.append(torch.zeros_like(audios))
                    
                elif speaker == '2':
                    voice_tensor = torch.load(voice2_path, weights_only=True)
                    generator = pipeline(content, voice=voice_tensor, speed=1, split_pattern=r'\n+')
                    audios = []
                    for gs, ps, audio in generator:
                        audios.append(audio)
                    audios = torch.concat(audios, dim=0)
                    s2_sentences.append(audios)
                    s1_sentences.append(torch.zeros_like(audios))
            
            if not s1_sentences:
                raise ValueError("对话脚本中没有找到有效的对话内容")
            
            s1_sentences = torch.concat(s1_sentences, dim=0)
            s2_sentences = torch.concat(s2_sentences, dim=0)
            sum_sentences = s1_sentences + s2_sentences
            
            # 保存音频文件
            session_id = uuid.uuid4().hex
            save_dir = self.temp_dir / session_id
            save_dir.mkdir(exist_ok=True)
            
            save_path1 = save_dir / 's1.wav'
            save_path2 = save_dir / 's2.wav'
            save_path_sum = save_dir / 'sum.wav'
            
            sf.write(save_path1, s1_sentences, 24000)
            sf.write(save_path2, s2_sentences, 24000)
            sf.write(save_path_sum, sum_sentences, 24000)
            
            # 转换为16kHz用于embedding
            s1, _ = librosa.load(save_path1, sr=16000)
            s2, _ = librosa.load(save_path2, sr=16000)
            
            logging.info(f"TTS音频生成完成: {save_dir}")
            return s1, s2, str(save_path_sum), str(save_dir)
            
        except Exception as e:
            logging.error(f"生成TTS音频时发生错误: {e}")
            raise
    
    def get_audio_embedding(self, speech_array, sr=16000):
        """获取音频embedding"""
        try:
            audio_duration = len(speech_array) / sr
            video_length = audio_duration * 25  # 假设视频fps为25
            
            # wav2vec特征提取
            audio_feature = np.squeeze(
                self.wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
            )
            audio_feature = torch.from_numpy(audio_feature).float().to(device='cpu')
            audio_feature = audio_feature.unsqueeze(0)
            
            # 音频编码
            with torch.no_grad():
                embeddings = self.audio_encoder(
                    audio_feature, seq_len=int(video_length), output_hidden_states=True
                )
            
            if len(embeddings) == 0:
                logging.error("提取音频embedding失败")
                return None
            
            audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
            audio_emb = rearrange(audio_emb, "b s d -> s b d")
            audio_emb = audio_emb.cpu().detach()
            
            return audio_emb
            
        except Exception as e:
            logging.error(f"获取音频embedding时发生错误: {e}")
            raise
    
    def generate_video(self, person1_image, person2_image, dialogue_script, 
                      prompt_text, voice1_path, voice2_path, 
                      sampling_steps=8, seed=42, text_guide_scale=5.0, 
                      audio_guide_scale=4.0, negative_prompt=""):
        """生成双人对话视频"""
        try:
            logging.info("开始生成视频...")
            
            # 1. 解析对话脚本
            formatted_dialogue = self.dialogue_parser.parse_dialogue(dialogue_script)
            logging.info(f"解析后的对话: {formatted_dialogue}")
            
            # 2. 创建合成图像
            composite_image_path, bbox_info = self.create_composite_image(
                person1_image, person2_image
            )
            
            # 3. 生成TTS音频
            s1_audio, s2_audio, sum_audio_path, audio_dir = self.generate_tts_audio(
                formatted_dialogue, voice1_path, voice2_path
            )
            
            # 4. 获取音频embeddings
            logging.info("正在提取音频特征...")
            audio_embedding_1 = self.get_audio_embedding(s1_audio)
            audio_embedding_2 = self.get_audio_embedding(s2_audio)
            
            # 保存embeddings
            emb1_path = Path(audio_dir) / '1.pt'
            emb2_path = Path(audio_dir) / '2.pt'
            torch.save(audio_embedding_1, emb1_path)
            torch.save(audio_embedding_2, emb2_path)
            
            # 5. 构建输入数据
            input_data = {
                "prompt": prompt_text,
                "cond_image": composite_image_path,
                "audio_type": "para",  # 并行音频模式
                "cond_audio": {
                    "person1": str(emb1_path),
                    "person2": str(emb2_path)
                },
                "video_audio": sum_audio_path,
                "bbox": bbox_info
            }
            
            # 6. 生成视频
            logging.info("正在使用MultiTalk管道生成视频...")
            video = self.wan_pipeline.generate(
                input_data,
                size_buckget="multitalk-720",  # 使用720P分辨率
                motion_frame=self.args.motion_frame,
                frame_num=self.args.frame_num,
                shift=self.args.sample_shift,
                sampling_steps=sampling_steps,
                text_guide_scale=text_guide_scale,
                audio_guide_scale=audio_guide_scale,
                seed=seed,
                n_prompt=negative_prompt,
                offload_model=self.args.offload_model,
                max_frames_num=self.args.frame_num,
                color_correction_strength=self.args.color_correction_strength,
                extra_args=self.args,
            )
            
            # 7. 保存视频
            from datetime import datetime
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"distributed_multitalk_720p_{formatted_time}"
            
            logging.info(f"正在保存生成的视频: {video_filename}.mp4")
            save_video_ffmpeg(
                video, video_filename, [sum_audio_path], high_quality_save=True
            )
            
            video_path = f"{video_filename}.mp4"
            logging.info("视频生成完成！")
            
            return video_path
            
        except Exception as e:
            logging.error(f"生成视频时发生错误: {e}")
            raise