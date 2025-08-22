import gradio as gr
import logging
from distributed_generator import DistributedMultiTalkGenerator


def create_gradio_interface(generator):
    """创建Gradio Web界面"""
    
    def generate_video_wrapper(person1_image, person2_image, dialogue_script, 
                             prompt_text, voice1_choice, voice2_choice,
                             sampling_steps, seed, text_guide_scale, audio_guide_scale,
                             negative_prompt):
        """Web界面的视频生成包装函数"""
        try:
            # 语音模型路径映射
            voice_mapping = {
                "女性温柔": "weights/Kokoro-82M/voices/af_heart.pt",
                "男性成熟": "weights/Kokoro-82M/voices/am_adam.pt",
                "女性活泼": "weights/Kokoro-82M/voices/af_bella.pt",
                "男性年轻": "weights/Kokoro-82M/voices/am_freeman.pt",
            }
            
            voice1_path = voice_mapping.get(voice1_choice, voice_mapping["女性温柔"])
            voice2_path = voice_mapping.get(voice2_choice, voice_mapping["男性成熟"])
            
            if not person1_image or not person2_image:
                return "错误：请上传两个人物图片"
            
            if not dialogue_script.strip():
                return "错误：请输入对话台词"
            
            if not prompt_text.strip():
                return "错误：请输入场景描述"
            
            # 生成视频
            video_path = generator.generate_video(
                person1_image=person1_image,
                person2_image=person2_image,
                dialogue_script=dialogue_script,
                prompt_text=prompt_text,
                voice1_path=voice1_path,
                voice2_path=voice2_path,
                sampling_steps=sampling_steps,
                seed=seed,
                text_guide_scale=text_guide_scale,
                audio_guide_scale=audio_guide_scale,
                negative_prompt=negative_prompt
            )
            
            return video_path
            
        except Exception as e:
            error_msg = f"生成视频时发生错误: {str(e)}"
            logging.error(error_msg)
            return error_msg
    
    # 创建Gradio界面
    with gr.Blocks(title="分布式MultiTalk双人对话视频生成") as demo:
        
        gr.Markdown("""
        <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
            🎬 分布式MultiTalk双人对话视频生成器 (720P)
        </div>
        <div style="text-align: center; font-size: 16px; font-weight: normal; margin-bottom: 20px;">
            基于8张RTX-4090分布式推理，生成高质量720P双人对话视频
        </div>
        <div style="text-align: center; font-size: 14px; color: #666; margin-bottom: 30px;">
            支持多种对话脚本格式 | 自动合成双人图像 | 智能语音生成 | 720P高清输出
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                
                # 人物图片上传
                with gr.Group():
                    gr.Markdown("### 👥 人物图片上传")
                    person1_image = gr.Image(
                        type="filepath",
                        label="人物1图片 (左侧)",
                        elem_id="person1_upload",
                    )
                    person2_image = gr.Image(
                        type="filepath", 
                        label="人物2图片 (右侧)",
                        elem_id="person2_upload",
                    )
                
                # 对话脚本输入
                with gr.Group():
                    gr.Markdown("### 💬 对话台词脚本")
                    dialogue_script = gr.Textbox(
                        label="对话内容",
                        placeholder="""请输入对话脚本，支持多种格式：

1. 标准格式：(s1) 你好！ (s2) 很高兴见到你！
2. 角色格式：角色1: 今天天气真好 角色2: 是啊，很适合出门
3. 简化格式：A: 这个项目怎么样？ B: 我觉得很不错！
4. 或者直接按行输入，系统会自动分配给两个角色

示例对话：
(s1) 你好，今天天气真不错啊！
(s2) 是的，很适合出去走走。
(s1) 要不要一起去公园？
(s2) 好主意，我们走吧！""",
                        lines=6,
                        max_lines=10
                    )
                
                # 场景描述
                prompt_text = gr.Textbox(
                    label="场景描述",
                    placeholder="""描述视频场景环境和氛围，例如：

在温馨的咖啡厅里，两个朋友坐在窗边的座位上进行愉快的对话。阳光透过大窗户洒在桌子上，营造出温暖舒适的氛围。他们面带微笑，眼神交流自然，体现出友好的互动关系。镜头采用中景拍摄，捕捉两人的表情和肢体语言。""",
                    lines=4,
                    value="在一个温馨舒适的环境中，两个人进行着友好自然的对话，他们面带微笑，眼神交流，展现出良好的互动关系。"
                )
                
                # 语音选择
                with gr.Group():
                    gr.Markdown("### 🎤 语音模型选择")
                    with gr.Row():
                        voice1_choice = gr.Dropdown(
                            choices=["女性温柔", "男性成熟", "女性活泼", "男性年轻"],
                            label="人物1语音",
                            value="女性温柔"
                        )
                        voice2_choice = gr.Dropdown(
                            choices=["女性温柔", "男性成熟", "女性活泼", "男性年轻"],
                            label="人物2语音", 
                            value="男性成熟"
                        )
                
                # 生成按钮
                generate_button = gr.Button(
                    "🚀 开始生成720P对话视频", 
                    variant="primary", 
                    size="lg"
                )
                
                # 高级设置
                with gr.Accordion("🔧 高级设置", open=False):
                    with gr.Row():
                        sampling_steps = gr.Slider(
                            label="采样步数",
                            minimum=4,
                            maximum=50,
                            value=8,
                            step=1,
                            info="更多步数生成质量更高，但耗时更长"
                        )
                        seed = gr.Slider(
                            label="随机种子",
                            minimum=-1,
                            maximum=2147483647,
                            value=42,
                            step=1,
                            info="-1表示随机种子"
                        )
                    
                    with gr.Row():
                        text_guide_scale = gr.Slider(
                            label="文本引导强度",
                            minimum=1.0,
                            maximum=20.0,
                            value=5.0,
                            step=0.5,
                            info="控制生成内容与文本描述的相符程度"
                        )
                        audio_guide_scale = gr.Slider(
                            label="音频引导强度", 
                            minimum=1.0,
                            maximum=20.0,
                            value=4.0,
                            step=0.5,
                            info="控制唇形同步和音频匹配程度"
                        )
                    
                    negative_prompt = gr.Textbox(
                        label="负面提示词",
                        placeholder="描述不希望出现的内容",
                        value="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                        lines=3
                    )

            with gr.Column(scale=2):
                result_video = gr.Video(
                    label='生成的720P对话视频', 
                    interactive=False, 
                    height=600
                )
                
                # 状态信息
                status_text = gr.Textbox(
                    label="生成状态",
                    value="等待开始生成...",
                    interactive=False,
                    lines=2
                )
                
                # 示例
                with gr.Accordion("📋 示例", open=True):
                    gr.Markdown("""
                    **示例对话脚本格式：**
                    
                    ```
                    (s1) 你知道MultiTalk这个项目吗？
                    (s2) 当然知道！这是一个很棒的音频驱动视频生成模型。
                    (s1) 它有什么特别的功能？
                    (s2) 它可以生成多人对话视频，支持720P高清输出。
                    (s1) 听起来很厉害，我想试试看。
                    (s2) 好的，我们一起来体验一下吧！
                    ```
                    
                    **生成过程说明：**
                    1. 上传两个人物图片（建议清晰的正面照）
                    2. 输入对话脚本（支持多种格式）
                    3. 描述场景环境和氛围
                    4. 选择合适的语音模型
                    5. 点击生成按钮，等待720P视频输出
                    
                    **注意事项：**
                    - 图片建议分辨率不低于512x512
                    - 对话长度建议控制在15秒以内
                    - 720P生成需要较长时间，请耐心等待
                    """)

        # 绑定生成事件
        generate_button.click(
            fn=generate_video_wrapper,
            inputs=[
                person1_image, person2_image, dialogue_script, prompt_text,
                voice1_choice, voice2_choice, sampling_steps, seed,
                text_guide_scale, audio_guide_scale, negative_prompt
            ],
            outputs=[result_video],
        )
        
        # 实时状态更新
        def update_status():
            return "正在生成中，请稍候..."
        
        generate_button.click(
            fn=update_status,
            outputs=[status_text]
        )
    
    return demo