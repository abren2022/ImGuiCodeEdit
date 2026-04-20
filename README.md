import os
from typing import List, Optional, Union
from peft import LoraConfig
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModel
import torch.nn as nn
import math
from .utils import is_torch2_available, get_generator
import numpy as np
if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
        CNAttnProcessor2_0 as CNAttnProcessor,
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler
def load_lora_model(unet, device, diffusion_model_learning_rate, dtype):
    for param in unet.parameters():
        param.requires_grad_(False)
    unet_lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=diffusion_model_learning_rate,
    )
    # 确保LoRA层使用正确的dtype
    for layer in lora_layers:
        layer.data = layer.data.to(dtype)
    return unet, lora_layers
class ImageProjModel(torch.nn.Module):
    """Projection Model"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
    def forward(self, image_embeds):
        embeds = image_embeds
        b = embeds.shape[0]
        clip_extra_context_tokens = self.proj(embeds).reshape(
            b, -1, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens
class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens
class SelfAttention(nn.Module):
    def __init__(self, in_channels, device, dtype=torch.float16):
        super(SelfAttention, self).__init__()
        self.dtype = dtype
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1).to(device, dtype=dtype)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1).to(device, dtype=dtype)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1).to(device, dtype=dtype)
        self.gamma = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
        self.softmax = nn.Softmax(dim=-1)
        self.proj_out = nn.Linear(1280, 1024).to(device, dtype=dtype)
    def forward(self, x, mask=None):
        # 统一转换为模型dtype
        if torch.isnan(x).any():
            print("Warning: Input to SelfAttention contains NaN!")
            x = torch.nan_to_num(x, nan=0.0)
        x = x.to(dtype=self.dtype)
        x = x.permute(0, 2, 1)
        batch_size, channels, h = x.size()
        height = int(math.sqrt(h))
        width = height
        x = x.view(batch_size, channels, width, height)
        batch_size, channels, height, width = x.size()
        # 计算 query, key, value
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)
        # 计算注意力分数
        attention_scores = torch.bmm(q, k)
        if mask is not None:
            # 将 mask 转换为正确的dtype并移到正确设备
            mask = mask.to(device=x.device, dtype=self.dtype)
            # 将 mask 的尺寸调整为和 x 一致
            mask = nn.functional.interpolate(mask, size=(height, width), mode='nearest')
            mask = mask.view(batch_size, 1, height * width)
            # 应用mask
            large_constant = torch.tensor(1e6, dtype=self.dtype, device=x.device)
            attention_scores = attention_scores - (1 - mask) * large_constant
        # 计算注意力权重
        attention_weights = self.softmax(attention_scores)
        # 应用注意力权重
        out = torch.bmm(v, attention_weights.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        # 加权求和
        out = self.gamma * out + x
        out = out.view(batch_size, channels, height * width)
        out = out.permute(0, 2, 1)
        out = self.proj_out(out)
        return out
import requests
import io
class Anomagic:
    def __init__(self, sd_pipe, image_encoder, ip_ckpt_url, att_ckpt_url, device, num_tokens=4, dtype=torch.float16):
        self.device = device
        self.num_tokens = num_tokens
        if str(device).startswith('cpu'):
            self.dtype = torch.float32
        else:
            self.dtype = dtype
        # 1. 初始化Attention模块（统一dtype）
        self.attention_module = SelfAttention(1280, device, dtype=self.dtype)
        # 2. 初始化SD管道（统一dtype）
        self.pipe = sd_pipe.to(self.device, dtype=self.dtype)
        self.set_anomagic()
        # 3. 处理image_encoder（优先使用传入的模型，而非重新加载）
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "/home/user/yaojie/Anomagic-main/models/image_encoder/", # 完整仓库路径
            torch_dtype=self.dtype, local_files_only=True
        ).to(self.device, dtype=self.dtype)
        self.clip_image_processor = CLIPImageProcessor()
        # 4. 初始化image_proj模型（统一dtype）
        self.image_proj_model = self.init_proj()
        # 5. 从URL加载权重到内存（核心修正）
        self.ip_state_dict = self.load_weight_from_url(ip_ckpt_url)
        self.att_state_dict = self.load_weight_from_url(att_ckpt_url)
        # 6. 加载权重到模型
        self.load_anomagic()
    def load_weight_from_url(self, path_or_url):
        """从URL下载权重到内存并返回state_dict（处理异常）"""
        try:
            if os.path.isfile(path_or_url):
                # --- 本地加载逻辑 ---
                print(f"Loading weights from local file: {path_or_url}")
                # 直接加载本地文件
                return torch.load(path_or_url, map_location="cpu")

            elif path_or_url.startswith(("http://", "https://")):
                # --- URL 下载逻辑 ---
                print(f"Downloading weights from URL: {path_or_url}")
                response = requests.get(path_or_url, stream=True, timeout=30)
                response.raise_for_status()
                buffer = io.BytesIO(response.content)
                return torch.load(buffer, map_location="cpu")

            else:
                raise FileNotFoundError(f"Invalid path or URL: {path_or_url}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"权重URL请求失败: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"权重加载失败: {str(e)}")
    def init_proj(self):
        """初始化image_proj模型（绑定dtype和device）"""
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=self.dtype)
        return image_proj_model
    def set_anomagic(self):
        """配置UNet的Attention处理器和LoRA"""
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            # 判断是否为cross attention
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            # 获取对应层的hidden_size
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                hidden_size = unet.config.cross_attention_dim # 兜底
            # 分配Attention处理器
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=self.dtype)
        # 应用处理器并加载LoRA
        unet.set_attn_processor(attn_procs)
        unet, lora_layers = load_lora_model(unet, self.device, 4e-4, self.dtype)
        # 处理ControlNet（若存在）
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
                    controlnet.to(self.device, dtype=self.dtype)
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
                self.pipe.controlnet.to(self.device, dtype=self.dtype)
    def load_anomagic(self):
        """统一加载IP Adapter和Attention权重（修复类型和冗余问题）"""
        # ========== 处理IP Adapter权重 ==========
        if isinstance(self.ip_state_dict, dict):
            # 内存权重字典（非safetensors）
            state_dict = self.ip_state_dict
            # 转换张量精度（兼容嵌套字典）
            self._convert_state_dict_dtype(state_dict)
            # 加载到对应模块（仅执行一次，删除冗余代码）
            def print_param_shapes(model, state_dict, prefix=""):
                """打印模型和state_dict的参数形状"""
                print(f"\n===== {prefix} 参数形状对比 =====")
                # 1. 打印模型的参数形状
                print("【模型参数】")
                for name, param in model.named_parameters():
                    print(f" {name}: {param.shape}")
                # 2. 打印state_dict的参数形状
                print("\n【StateDict参数】")
                for key, tensor in state_dict.items():
                    print(f" {key}: {tensor.shape}")
            # 在self.image_proj_model.load_state_dict(state_dict["image_proj"])前调用
            print_param_shapes(self.image_proj_model, state_dict["image_proj"], "image_proj_model")
            self.image_proj_model.load_state_dict(state_dict["image_proj"])
            ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
            ip_layers.load_state_dict(state_dict["ip_adapter"])
            # 加载UNet额外权重（若有）
            if "unet" in state_dict:
                self.pipe.unet.load_state_dict(state_dict["unet"], strict=False)
        else:
            raise TypeError("ip_state_dict必须是内存中的权重字典，而非文件路径")
        # ========== 处理Attention模块权重 ==========
        if isinstance(self.att_state_dict, dict):
            att_state_dict = self.att_state_dict.get("att", self.att_state_dict)
            # 转换Attention权重精度
            self._convert_state_dict_dtype(att_state_dict)
            self.attention_module.load_state_dict(att_state_dict, strict=True)
        else:
            raise TypeError("att_state_dict必须是内存中的权重字典")
    def _convert_state_dict_dtype(self, state_dict):
        """递归转换state_dict中所有张量的dtype（工具函数）"""
        for key in list(state_dict.keys()):
            value = state_dict[key]
            if isinstance(value, torch.Tensor):
                state_dict[key] = value.to(self.dtype)
            elif isinstance(value, dict):
                self._convert_state_dict_dtype(value) # 递归处理嵌套字典
    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None, mask_image_0=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image = clip_image.to(self.device, dtype=self.dtype)
            outputs = self.image_encoder(clip_image)
            clip_image_embeds = outputs.image_embeds
            last_feature_layer_output = outputs.last_hidden_state
            print(f"CLIP last_hidden_state NaN: {torch.isnan(last_feature_layer_output).any()}")
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=self.dtype)
        # 处理mask_image_0

        if mask_image_0 is not None:
            if isinstance(mask_image_0, Image.Image):
                mask_image_0 = mask_image_0.resize((64, 64))
                mask_image_0 = mask_image_0.convert('L')
                mask_tensor = torch.tensor(np.array(mask_image_0), dtype=self.dtype, device=self.device)
                mask_tensor = (mask_tensor / 255.0)  # 归一化到 [0, 1]
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 64, 64]
            else:
                mask_tensor = mask_image_0
        else:
            mask_tensor = None
        mask_tensor = None
        # 使用统一的dtype处理特征
        attn_input = last_feature_layer_output[:, :256, :]
        print(f"Attention input NaN: {torch.isnan(attn_input).any()}")
        image_embeds = self.attention_module(
            last_feature_layer_output[:, :256, :],
            mask_tensor
        )
        # 生成image_prompt_embeds
        image_prompt_embeds = self.image_proj_model(image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds
    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale
    def encode_long_text(self,
                         input_ids: torch.Tensor,
                         tokenizer: CLIPTokenizer,
                         text_encoder: CLIPTextModel,
                         max_length: int = 77,
                         device: str = None
                         ) -> torch.Tensor:
        device = device or self.device
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        batch_size = input_ids.size(0)
        hidden_dim = text_encoder.config.hidden_size
        combined_embeddings = torch.zeros(batch_size, hidden_dim, device=device, dtype=self.dtype)
        for batch_idx in range(batch_size):
            current_input_ids = input_ids[batch_idx]
            chunks = [
                current_input_ids[i:i + max_length]
                for i in range(0, len(current_input_ids), max_length)
            ]
            embeddings = []
            for chunk in chunks:
                chunk_len = len(chunk)
                padding_len = max_length - chunk_len
                chunk_input = {
                    "input_ids": torch.cat([
                        chunk.unsqueeze(0).to(device),
                        torch.zeros(1, padding_len, dtype=torch.long, device=device)
                    ], dim=1),
                    "attention_mask": torch.cat([
                        torch.ones(1, chunk_len, dtype=torch.long, device=device),
                        torch.zeros(1, padding_len, dtype=torch.long, device=device)
                    ], dim=1)
                }
                with torch.no_grad():
                    chunk_emb = text_encoder(**chunk_input).last_hidden_state
                    embeddings.append(chunk_emb[:, :chunk_len, :].mean(dim=1))
            if embeddings:
                combined_embeddings[batch_idx] = torch.mean(torch.cat(embeddings, dim=0), dim=0)
        return combined_embeddings.unsqueeze(1)
    def generate(
            self,
            pil_image=None,
            clip_image_embeds=None,
            prompt=None,
            negative_prompt=None,
            scale=0.6,
            num_samples=4,
            seed=None,
            guidance_scale=4,
            num_inference_steps=50,
            mask_image_0=None,
            **kwargs,
    ):
        self.set_scale(scale)
        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0) if clip_image_embeds is not None else 1
        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds, mask_image_0=mask_image_0,
        )
        # 数值检查
        print(f"Image embeds stats - mean: {image_prompt_embeds.mean().item():.4f}, "
              f"std: {image_prompt_embeds.std().item():.4f}, "
              f"min: {image_prompt_embeds.min().item():.4f}, "
              f"max: {image_prompt_embeds.max().item():.4f}")
        if torch.isnan(image_prompt_embeds).any():
            print("NaN detected in image_prompt_embeds, applying fix")
            image_prompt_embeds = torch.nan_to_num(image_prompt_embeds, nan=0.0)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        print(f"提示词数量：{len(prompt)}")
        with torch.inference_mode():
            # 编码文本提示
            prompt_embeds_list = []
            for p in prompt:
                inputs = self.pipe.tokenizer(
                    p,
                    padding="max_length",
                    max_length=self.pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = inputs.input_ids.to(self.device)
                prompt_embed = self.encode_long_text(
                    input_ids=input_ids,
                    tokenizer=self.pipe.tokenizer,
                    text_encoder=self.pipe.text_encoder,
                    device=self.device
                )
                prompt_embeds_list.append(prompt_embed)
            prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
            # 编码负向提示
            negative_prompt_embeds_list = []
            for p in negative_prompt:
                inputs = self.pipe.tokenizer(
                    p,
                    padding="max_length",
                    max_length=self.pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = inputs.input_ids.to(self.device)
                negative_prompt_embed = self.encode_long_text(
                    input_ids=input_ids,
                    tokenizer=self.pipe.tokenizer,
                    text_encoder=self.pipe.text_encoder,
                    device=self.device
                )
                negative_prompt_embeds_list.append(negative_prompt_embed)
            negative_prompt_embeds = torch.cat(negative_prompt_embeds_list, dim=0)
            # 合并图像嵌入与文本嵌入
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
        generator = get_generator(seed, self.device)
        print(f"生成图片数量：{num_samples}")
        try:
            # 尝试以 tensor 形式获取输出，避免内部自动 cast 报错
            output = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pt",  # 关键修改：返回 torch tensor
                **kwargs,
            )
            images_tensor = output.images

            # 2. 清理 NaN 和 Inf
            if torch.isnan(images_tensor).any() or torch.isinf(images_tensor).any():
                print("[Warning] Detected NaN or Inf in generated images. Replacing with 0.")
                images_tensor = torch.nan_to_num(images_tensor, nan=0.0, posinf=1.0, neginf=0.0)

            # 3. 手动转换为 PIL 图像 (模拟 diffusers 的行为)
            from diffusers.utils import pt_to_pil
            images = pt_to_pil(images_tensor)

        except Exception as e:
            # 如果 output_type="pt" 不可用或出错，回退到默认行为，但需捕获警告
            print(f"Using default pipeline output due to error: {e}")
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                **kwargs,
            ).images
        ###########################
        import os
        from datetime import datetime

        # 创建保存目录
        save_dir = "generated_outputs"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, img in enumerate(images):
            # 检查图像是否有效（避免保存 NaN/Inf 导致的损坏文件）
            try:
                import numpy as np
                arr = np.array(img)
                if np.isnan(arr).any() or np.isinf(arr).any():
                    print(f"[Warning] Image {i} contains invalid data, skipping save.")
                    continue

                filename = f"anomagic_{timestamp}_seed{seed}_{i}.png"
                save_path = os.path.join(save_dir, filename)
                img.save(save_path)
                print(f"Image saved to: {save_path}")
            except Exception as e:
                print(f"Failed to save image {i}: {e}")
        # --------------------------

        return images
    
class AnomagicXL(Anomagic):
    """SDXL"""
    def generate(
            self,
            pil_image,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            num_inference_steps=30, **kwargs,
    ):
        self.set_scale(scale)
        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
        self.generator = get_generator(seed, self.device)
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator, **kwargs,
        ).images
        return images
class AnomagicPlus(Anomagic):
    """Anomagic with fine-grained features"""
    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=self.dtype)
        return image_proj_model
    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=self.dtype)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds
class AnomagicFull(AnomagicPlus):
    """Anomagic with full features"""
    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=self.dtype)
        return image_proj_model
class AnomagicPlusXL(Anomagic):
    """SDXL"""
    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=self.dtype)
        return image_proj_model
    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=self.dtype)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds
    def generate(
            self,
            pil_image,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            num_inference_steps=30, **kwargs,
    ):
        self.set_scale(scale)
        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
        generator = get_generator(seed, self.device)
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator, **kwargs,
        ).images
        return images
