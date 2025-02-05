"""Modified from https://github.com/kijai/ComfyUI-EasyAnimateWrapper/blob/main/nodes.py
"""
import gc
import os
import json

import comfy.model_management as mm
import cv2
import folder_paths
import numpy as np
import torch
import torch.nn.functional as F
import copy
from comfy.utils import ProgressBar, load_torch_file
from diffusers import (AutoencoderKL, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from transformers import (BertModel, BertTokenizer,
                          CLIPImageProcessor, CLIPVisionModelWithProjection,
                          Qwen2Tokenizer, Qwen2VLForConditionalGeneration,
                          T5EncoderModel, T5Tokenizer)

from .utils import combine_camera_motion, get_camera_motion, CAMERA
from ..easyanimate.data.bucket_sampler import (ASPECT_RATIO_512,
                                               get_closest_ratio)
from ..easyanimate.data.dataset_image_video import process_pose_params
from ..easyanimate.models import (name_to_autoencoder_magvit,
                                  name_to_transformer3d)
from ..easyanimate.pipeline.pipeline_easyanimate import \
    EasyAnimatePipeline
from ..easyanimate.pipeline.pipeline_easyanimate_inpaint import \
    EasyAnimateInpaintPipeline
from ..easyanimate.pipeline.pipeline_easyanimate_control import \
    EasyAnimateControlPipeline
from ..easyanimate.utils.lora_utils import merge_lora, unmerge_lora
from ..easyanimate.utils.utils import (get_image_to_video_latent, get_image_latent,
                                       get_video_to_video_latent)
from ..easyanimate.utils.fp8_optimization import convert_weight_dtype_wrapper
from ..easyanimate.ui.ui import ddpm_scheduler_dict, flow_scheduler_dict, all_cheduler_dict

# Compatible with Alibaba EAS for quick launch
eas_cache_dir           = '/stable-diffusion-cache/models'
# The directory of the easyanimate
script_directory        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Used in lora cache
transformer_cpu_cache   = {}
# lora path before
lora_path_before        = ""

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))

def numpy2pil(image):
    return Image.fromarray(np.clip(255. * image, 0, 255).astype(np.uint8))

def to_pil(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        return tensor2pil(image)
    if isinstance(image, np.ndarray):
        return numpy2pil(image)
    raise ValueError(f"Cannot convert {type(image)} to PIL.Image")

def gen_gaussian_heatmap(imgSize=200):
    circle_img = np.zeros((imgSize, imgSize,), np.float32) 
    circle_mask = cv2.circle(circle_img, (imgSize//2, imgSize//2), imgSize//2 - 1, 1, -1)

    isotropicGrayscaleImage = np.zeros((imgSize, imgSize), np.float32)

    # 生成高斯图
    for i in range(imgSize):
        for j in range(imgSize):
            isotropicGrayscaleImage[i, j] = 1 / (2 * np.pi * (40 ** 2)) * np.exp(
                -1 / 2 * ((i - imgSize / 2) ** 2 / (40 ** 2) + (j - imgSize / 2) ** 2 / (40 ** 2)))

    isotropicGrayscaleImage = isotropicGrayscaleImage * circle_mask
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage) * 255).astype(np.uint8)
    return isotropicGrayscaleImage

class LoadEasyAnimateModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [ 
                        'EasyAnimateV3-XL-2-InP-512x512',
                        'EasyAnimateV3-XL-2-InP-768x768',
                        'EasyAnimateV3-XL-2-InP-960x960',
                        'EasyAnimateV4-XL-2-InP',
                        'EasyAnimateV5-7b-zh-InP',
                        'EasyAnimateV5-7b-zh',
                        'EasyAnimateV5-12b-zh-InP',
                        'EasyAnimateV5-12b-zh-Control',
                        'EasyAnimateV5-12b-zh',
                        'EasyAnimateV5.1-12b-zh-InP',
                        'EasyAnimateV5.1-12b-zh-Control',
                        'EasyAnimateV5.1-12b-zh-Control-Camera',
                    ],
                    {
                        "default": 'EasyAnimateV5.1-12b-zh-InP',
                    }
                ),
                "GPU_memory_mode":(
                    ["model_cpu_offload", "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"],
                    {
                        "default": "model_cpu_offload",
                    }
                ),
                "model_type": (
                    ["Inpaint", "Control"],
                    {
                        "default": "Inpaint",
                    }
                ),
                "config": (
                    [
                        "easyanimate_video_v3_slicevae_motion_module.yaml",
                        "easyanimate_video_v4_slicevae_multi_text_encoder.yaml",
                        "easyanimate_video_v5_magvit_multi_text_encoder.yaml",
                        "easyanimate_video_v5.1_magvit_qwen.yaml",
                    ],
                    {
                        "default": "easyanimate_video_v5.1_magvit_qwen.yaml",
                    }
                ),
                "precision": (
                    ['fp16', 'bf16'],
                    {
                        "default": 'bf16'
                    }
                ),
                
            },
        }

    RETURN_TYPES = ("EASYANIMATESMODEL",)
    RETURN_NAMES = ("easyanimate_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "EasyAnimateWrapper"

    def loadmodel(self, GPU_memory_mode, model, precision, model_type, config):
        # Init weight_dtype and device
        device          = mm.get_torch_device()
        offload_device  = mm.unet_offload_device()
        weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        # Init processbar
        pbar = ProgressBar(4)

        # Load config
        config_path = f"{script_directory}/config/{config}"
        config = OmegaConf.load(config_path)

        # Detect model is existing or not 
        model_name = os.path.join(folder_paths.models_dir, "EasyAnimate", model)
      
        if not os.path.exists(model_name):
            if os.path.exists(eas_cache_dir):
                model_name = os.path.join(eas_cache_dir, 'EasyAnimate', model)
            else:
                print(f"Please download easyanimate model to: {model_name}")

        # Get Vae
        Choosen_AutoencoderKL = name_to_autoencoder_magvit[
            config['vae_kwargs'].get('vae_type', 'AutoencoderKL')
        ]
        vae = Choosen_AutoencoderKL.from_pretrained(
            model_name, 
            subfolder="vae"
        ).to(weight_dtype)
        if config['vae_kwargs'].get('vae_type', 'AutoencoderKL') == 'AutoencoderKLMagvit' and weight_dtype == torch.float16:
            vae.upcast_vae = True
        # Update pbar
        pbar.update(1)

        # Get Transformer
        Choosen_Transformer3DModel = name_to_transformer3d[
            config['transformer_additional_kwargs'].get('transformer_type', 'Transformer3DModel')
        ]

        transformer_additional_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
        if weight_dtype == torch.float16:
            transformer_additional_kwargs["upcast_attention"] = True

        transformer = Choosen_Transformer3DModel.from_pretrained_2d(
            model_name, 
            subfolder="transformer",
            transformer_additional_kwargs=transformer_additional_kwargs,
            torch_dtype=torch.float8_e4m3fn if GPU_memory_mode == "model_cpu_offload_and_qfloat8" else weight_dtype,
            low_cpu_mem_usage=True,
        )
        # Update pbar
        pbar.update(1) 

        if config['text_encoder_kwargs'].get('enable_multi_text_encoder', False):
            tokenizer = BertTokenizer.from_pretrained(
                model_name, subfolder="tokenizer"
            )
            if config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
                tokenizer_2 = Qwen2Tokenizer.from_pretrained(
                    os.path.join(model_name, "tokenizer_2")
                )
            else:
                tokenizer_2 = T5Tokenizer.from_pretrained(
                    model_name, subfolder="tokenizer_2"
                )
        else:
            if config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
                tokenizer = Qwen2Tokenizer.from_pretrained(
                    os.path.join(model_name, "tokenizer")
                )
            else:
                tokenizer = T5Tokenizer.from_pretrained(
                    model_name, subfolder="tokenizer"
                )
            tokenizer_2 = None

        if config['text_encoder_kwargs'].get('enable_multi_text_encoder', False):
            text_encoder = BertModel.from_pretrained(
                model_name, subfolder="text_encoder", torch_dtype=weight_dtype
            )
            if config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
                text_encoder_2 = Qwen2VLForConditionalGeneration.from_pretrained(
                    os.path.join(model_name, "text_encoder_2"),
                    torch_dtype=weight_dtype,
                )
            else:
                text_encoder_2 = T5EncoderModel.from_pretrained(
                    model_name, subfolder="text_encoder_2"
                ).to(weight_dtype)
        else:
            if config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
                text_encoder = Qwen2VLForConditionalGeneration.from_pretrained(
                    os.path.join(model_name, "text_encoder"),
                    torch_dtype=weight_dtype,
                )
            else:
                text_encoder = T5EncoderModel.from_pretrained(
                    model_name, subfolder="text_encoder"
                ).to(weight_dtype)
            text_encoder_2 = None

        # Load Clip
        if transformer.config.in_channels != vae.config.latent_channels and config['transformer_additional_kwargs'].get('enable_clip_in_inpaint', True):
            clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                model_name, subfolder="image_encoder"
            ).to("cuda", weight_dtype)
            clip_image_processor = CLIPImageProcessor.from_pretrained(
                model_name, subfolder="image_encoder"
            )
        else:
            clip_image_encoder = None
            clip_image_processor = None
        # Update pbar
        pbar.update(1)

        # Load Sampler
        print("Load Sampler.")
        scheduler = DDIMScheduler.from_pretrained(model_name, subfolder='scheduler')
        # Update pbar
        pbar.update(1)

        if model_type == "Inpaint":
            if transformer.config.in_channels != vae.config.latent_channels:
                pipeline = EasyAnimateInpaintPipeline(
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    vae=vae,
                    transformer=transformer,
                    scheduler=scheduler,
                    clip_image_encoder=clip_image_encoder,
                    clip_image_processor=clip_image_processor,
                )
            else:
                pipeline = EasyAnimatePipeline(
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    vae=vae,
                    transformer=transformer,
                    scheduler=scheduler,
                )
        else:
            pipeline = EasyAnimateControlPipeline(
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                vae=vae,
                transformer=transformer,
                scheduler=scheduler,
            )
        if GPU_memory_mode == "sequential_cpu_offload":
            pipeline.enable_sequential_cpu_offload()
        elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            pipeline.enable_model_cpu_offload()
            convert_weight_dtype_wrapper(transformer, weight_dtype)
        else:
            pipeline.enable_model_cpu_offload()

        easyanimate_model = {
            'pipeline': pipeline, 
            'dtype': weight_dtype,
            'model_name': model_name,
            'model_type': model_type,
            'loras': [],
            'strength_model': [],
        }
        return (easyanimate_model,)

class LoadEasyAnimateLora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "easyanimate_model": ("EASYANIMATESMODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"), {"default": None,}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_cache":([False, True],  {"default": False,}),
            }
        }
    RETURN_TYPES = ("EASYANIMATESMODEL",)
    RETURN_NAMES = ("easyanimate_model",)
    FUNCTION = "load_lora"
    CATEGORY = "EasyAnimateWrapper"

    def load_lora(self, easyanimate_model, lora_name, strength_model, lora_cache):
        if lora_name is not None:
            return (
                {
                    'pipeline': easyanimate_model["pipeline"], 
                    'dtype': easyanimate_model["dtype"],
                    'model_name': easyanimate_model["model_name"],
                    'loras': easyanimate_model.get("loras", []) + [folder_paths.get_full_path("loras", lora_name)],
                    'strength_model': easyanimate_model.get("strength_model", []) + [strength_model],
                    'lora_cache': lora_cache,
                }, 
            )
        else:
            return (easyanimate_model,)

class TextBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "",}),
            }
        }
    
    RETURN_TYPES = ("STRING_PROMPT",)
    RETURN_NAMES =("prompt",)
    FUNCTION = "process"
    CATEGORY = "EasyAnimateWrapper"

    def process(self, prompt):
        return (prompt, )

class EasyAnimate_TextBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "",}),
            }
        }
    
    RETURN_TYPES = ("STRING_PROMPT",)
    RETURN_NAMES =("prompt",)
    FUNCTION = "process"
    CATEGORY = "EasyAnimateWrapper"

    def process(self, prompt):
        return (prompt, )

class EasyAnimateT2VSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "easyanimate_model": (
                    "EASYANIMATESMODEL", 
                ),
                "prompt": (
                    "STRING_PROMPT", 
                ),
                "negative_prompt": (
                    "STRING_PROMPT", 
                ),
                "video_length": (
                    "INT", {"default": 72, "min": 8, "max": 144, "step": 8}
                ),
                "width": (
                    "INT", {"default": 1008, "min": 64, "max": 2048, "step": 16}
                ),
                "height": (
                    "INT", {"default": 576, "min": 64, "max": 2048, "step": 16}
                ),
                "is_image":(
                    [
                        False,
                        True
                    ], 
                    {
                        "default": False,
                    }
                ),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "steps": (
                    "INT", {"default": 25, "min": 1, "max": 200, "step": 1}
                ),
                "cfg": (
                    "FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.01}
                ),
                "scheduler": (
                    [ 
                        "Euler",
                        "Euler A",
                        "DPM++",
                        "PNDM",
                        "DDIM",
                    ],
                    {
                        "default": 'DDIM'
                    }
                ),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "EasyAnimateWrapper"

    def process(self, easyanimate_model, prompt, negative_prompt, video_length, width, height, is_image, seed, steps, cfg, scheduler, teacache_threshold=0.10, enable_teacache=False):
        global transformer_cpu_cache
        global lora_path_before
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        gc.collect()

        # Get Pipeline
        pipeline = easyanimate_model['pipeline']
        model_name = easyanimate_model['model_name']
        weight_dtype = easyanimate_model['dtype']

        # Load Sampler
        pipeline.scheduler = all_cheduler_dict[scheduler].from_pretrained(model_name, subfolder='scheduler')

        if enable_teacache:
            pipeline.transformer.enable_teacache(steps, teacache_threshold)

        generator= torch.Generator(device).manual_seed(seed)
        
        video_length = 1 if is_image else video_length
        with torch.no_grad():
            if pipeline.vae.cache_mag_vae:
                video_length = int((video_length - 1) // pipeline.vae.mini_batch_encoder * pipeline.vae.mini_batch_encoder) + 1 if video_length != 1 else 1
            else:
                video_length = int(video_length // pipeline.vae.mini_batch_encoder * pipeline.vae.mini_batch_encoder) if video_length != 1 else 1

            # Apply lora
            if easyanimate_model.get("lora_cache", False):
                if len(easyanimate_model.get("loras", [])) != 0:
                    # Save the original weights to cpu
                    if len(transformer_cpu_cache) == 0:
                        print('Save transformer state_dict to cpu memory')
                        transformer_state_dict = pipeline.transformer.state_dict()
                        for key in transformer_state_dict:
                            transformer_cpu_cache[key] = transformer_state_dict[key].clone().cpu()
                    
                    lora_path_now = str(easyanimate_model.get("loras", []) + easyanimate_model.get("strength_model", []))
                    if lora_path_now != lora_path_before:
                        print('Merge Lora with Cache')
                        lora_path_before = copy.deepcopy(lora_path_now)
                        pipeline.transformer.load_state_dict(transformer_cpu_cache)
                        for _lora_path, _lora_weight in zip(easyanimate_model.get("loras", []), easyanimate_model.get("strength_model", [])):
                            pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)
            else:
                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if len(transformer_cpu_cache) != 0:
                    print('Delete cpu state_dict')
                    pipeline.transformer.load_state_dict(transformer_cpu_cache)
                    transformer_cpu_cache = {}
                    lora_path_before = ""
                    gc.collect()
                print('Merge Lora')
                for _lora_path, _lora_weight in zip(easyanimate_model.get("loras", []), easyanimate_model.get("strength_model", [])):
                    pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)

            if pipeline.transformer.config.in_channels != pipeline.vae.config.latent_channels:
                input_video, input_video_mask, clip_image = get_image_to_video_latent(None, None, video_length=video_length, sample_size=(height, width))

                sample = pipeline(
                    prompt, 
                    video_length = video_length,
                    negative_prompt = negative_prompt,
                    height      = height,
                    width       = width,
                    generator   = generator,
                    guidance_scale = cfg,
                    num_inference_steps = steps,

                    video        = input_video,
                    mask_video   = input_video_mask,
                    clip_image   = clip_image, 
                    comfyui_progressbar = True,
                ).frames
            else:
                sample = pipeline(
                    prompt, 
                    video_length = video_length,
                    negative_prompt = negative_prompt,
                    height      = height,
                    width       = width,
                    generator   = generator,
                    guidance_scale = cfg,
                    num_inference_steps = steps,
                ).frames
            videos = rearrange(sample, "b c t h w -> (b t) h w c")

            if not easyanimate_model.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(easyanimate_model.get("loras", []), easyanimate_model.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)
        return (videos,)   

class EasyAnimateV5_T2VSampler(EasyAnimateT2VSampler):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "easyanimate_model": (
                    "EASYANIMATESMODEL", 
                ),
                "prompt": (
                    "STRING_PROMPT", 
                ),
                "negative_prompt": (
                    "STRING_PROMPT", 
                ),
                "video_length": (
                    "INT", {"default": 49, "min": 1, "max": 49, "step": 4}
                ),
                "width": (
                    "INT", {"default": 1008, "min": 64, "max": 2048, "step": 16}
                ),
                "height": (
                    "INT", {"default": 576, "min": 64, "max": 2048, "step": 16}
                ),
                "is_image":(
                    [
                        False,
                        True
                    ], 
                    {
                        "default": False,
                    }
                ),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "steps": (
                    "INT", {"default": 25, "min": 1, "max": 200, "step": 1}
                ),
                "cfg": (
                    "FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.01}
                ),
                "scheduler": (
                    [ 
                        "Euler",
                        "Euler A",
                        "DPM++",
                        "PNDM",
                        "DDIM",
                        "Flow",
                    ],
                    {
                        "default": 'Flow'
                    }
                ),
                "teacache_threshold": ("FLOAT", {"default": 0.10, "min": 0.00, "max": 1.00, "step": 0.005}),
                "enable_teacache":([False, True],  {"default": True,}),
            },
        }

class EasyAnimateI2VSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "easyanimate_model": (
                    "EASYANIMATESMODEL", 
                ),
                "prompt": (
                    "STRING_PROMPT",
                ),
                "negative_prompt": (
                    "STRING_PROMPT",
                ),
                "video_length": (
                    "INT", {"default": 72, "min": 8, "max": 144, "step": 8}
                ),
                "base_resolution": (
                    [ 
                        512,
                        768,
                        960,
                        1024,
                    ], {"default": 768}
                ),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "steps": (
                    "INT", {"default": 25, "min": 1, "max": 200, "step": 1}
                ),
                "cfg": (
                    "FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.01}
                ),
                "scheduler": (
                    [ 
                        "Euler",
                        "Euler A",
                        "DPM++",
                        "PNDM",
                        "DDIM",
                    ],
                    {
                        "default": 'DDIM'
                    }
                )
            },
            "optional":{
                "start_img": ("IMAGE",),
                "end_img": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "EasyAnimateWrapper"

    def process(self, easyanimate_model, prompt, negative_prompt, video_length, base_resolution, seed, steps, cfg, scheduler, start_img=None, end_img=None, teacache_threshold=0.10, enable_teacache=False):
        global transformer_cpu_cache
        global lora_path_before
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        gc.collect()

        start_img = [to_pil(_start_img) for _start_img in start_img] if start_img is not None else None
        end_img = [to_pil(_end_img) for _end_img in end_img] if end_img is not None else None
        # Count most suitable height and width
        aspect_ratio_sample_size = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        original_width, original_height = start_img[0].size if type(start_img) is list else Image.open(start_img).size
        closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
        height, width = [int(x / 16) * 16 for x in closest_size]
        
        # Get Pipeline
        pipeline = easyanimate_model['pipeline']
        model_name = easyanimate_model['model_name']
        weight_dtype = easyanimate_model['dtype']

        # Load Sampler
        pipeline.scheduler = all_cheduler_dict[scheduler].from_pretrained(model_name, subfolder='scheduler')

        if enable_teacache:
            pipeline.transformer.enable_teacache(steps, teacache_threshold)

        generator= torch.Generator(device).manual_seed(seed)

        with torch.no_grad():
            if pipeline.vae.cache_mag_vae:
                video_length = int((video_length - 1) // pipeline.vae.mini_batch_encoder * pipeline.vae.mini_batch_encoder) + 1 if video_length != 1 else 1
            else:
                video_length = int(video_length // pipeline.vae.mini_batch_encoder * pipeline.vae.mini_batch_encoder) if video_length != 1 else 1
            input_video, input_video_mask, clip_image = get_image_to_video_latent(start_img, end_img, video_length=video_length, sample_size=(height, width))

            # Apply lora
            if easyanimate_model.get("lora_cache", False):
                if len(easyanimate_model.get("loras", [])) != 0:
                    # Save the original weights to cpu
                    if len(transformer_cpu_cache) == 0:
                        print('Save transformer state_dict to cpu memory')
                        transformer_state_dict = pipeline.transformer.state_dict()
                        for key in transformer_state_dict:
                            transformer_cpu_cache[key] = transformer_state_dict[key].clone().cpu()
                    
                    lora_path_now = str(easyanimate_model.get("loras", []) + easyanimate_model.get("strength_model", []))
                    if lora_path_now != lora_path_before:
                        print('Merge Lora with Cache')
                        lora_path_before = copy.deepcopy(lora_path_now)
                        pipeline.transformer.load_state_dict(transformer_cpu_cache)
                        for _lora_path, _lora_weight in zip(easyanimate_model.get("loras", []), easyanimate_model.get("strength_model", [])):
                            pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)
            else:
                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if len(transformer_cpu_cache) != 0:
                    pipeline.transformer.load_state_dict(transformer_cpu_cache)
                    transformer_cpu_cache = {}
                    lora_path_before = ""
                    gc.collect()
                print('Merge Lora')
                for _lora_path, _lora_weight in zip(easyanimate_model.get("loras", []), easyanimate_model.get("strength_model", [])):
                    pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)

            sample = pipeline(
                prompt, 
                video_length = video_length,
                negative_prompt = negative_prompt,
                height      = height,
                width       = width,
                generator   = generator,
                guidance_scale = cfg,
                num_inference_steps = steps,

                video        = input_video,
                mask_video   = input_video_mask,
                clip_image   = clip_image, 
                comfyui_progressbar = True,
            ).frames
            videos = rearrange(sample, "b c t h w -> (b t) h w c")

            if not easyanimate_model.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(easyanimate_model.get("loras", []), easyanimate_model.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)
        return (videos,)   

class EasyAnimateV5_I2VSampler(EasyAnimateI2VSampler):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "easyanimate_model": (
                    "EASYANIMATESMODEL", 
                ),
                "prompt": (
                    "STRING_PROMPT",
                ),
                "negative_prompt": (
                    "STRING_PROMPT",
                ),
                "video_length": (
                    "INT", {"default": 49, "min": 1, "max": 49, "step": 4}
                ),
                "base_resolution": (
                    [ 
                        512,
                        768,
                        960,
                        1024,
                    ], {"default": 768}
                ),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "steps": (
                    "INT", {"default": 25, "min": 1, "max": 200, "step": 1}
                ),
                "cfg": (
                    "FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.01}
                ),
                "scheduler": (
                    [ 
                        "Euler",
                        "Euler A",
                        "DPM++",
                        "PNDM",
                        "DDIM",
                        "Flow",
                    ],
                    {
                        "default": 'Flow'
                    }
                ),
                "teacache_threshold": ("FLOAT", {"default": 0.10, "min": 0.00, "max": 1.00, "step": 0.005}),
                "enable_teacache":([False, True],  {"default": True,}),
            },
            "optional":{
                "start_img": ("IMAGE",),
                "end_img": ("IMAGE",),
            },
        }

class EasyAnimateV2VSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "easyanimate_model": (
                    "EASYANIMATESMODEL", 
                ),
                "prompt": (
                    "STRING_PROMPT",
                ),
                "negative_prompt": (
                    "STRING_PROMPT",
                ),
                "video_length": (
                    "INT", {"default": 72, "min": 8, "max": 144, "step": 8}
                ),
                "base_resolution": (
                    [ 
                        512,
                        768,
                        960,
                        1024,
                    ], {"default": 768}
                ),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "steps": (
                    "INT", {"default": 25, "min": 1, "max": 200, "step": 1}
                ),
                "cfg": (
                    "FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.01}
                ),
                "denoise_strength": (
                    "FLOAT", {"default": 0.70, "min": 0.05, "max": 1.00, "step": 0.01}
                ),
                "scheduler": (
                    [ 
                        "Euler",
                        "Euler A",
                        "DPM++",
                        "PNDM",
                        "DDIM",
                    ],
                    {
                        "default": 'DDIM'
                    }
                ),
            },
            "optional":{
                "validation_video": ("IMAGE",),
                "control_video": ("IMAGE",),
                "start_image": ("IMAGE",),
                "start_index": (
                    "INT", {"default": 0, "min": 0, "max": 300, "step": 1}
                ),
                "end_image": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "EasyAnimateWrapper"

    def process(self, easyanimate_model, prompt, negative_prompt, video_length, base_resolution, seed, steps, cfg, denoise_strength, scheduler, validation_video=None, control_video=None, ref_image=None, start_image=None, start_index=0, end_image=None, camera_conditions=None, teacache_threshold=0.10, enable_teacache=False):
        global transformer_cpu_cache
        global lora_path_before

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        gc.collect()
        
        # Get Pipeline and model info
        pipeline = easyanimate_model['pipeline']
        model_name = easyanimate_model['model_name']
        weight_dtype = easyanimate_model['dtype']
        model_type = easyanimate_model['model_type']

        # --- Resolution & Aspect Ratio Calculation (unchanged) ---
        aspect_ratio_sample_size = {key: [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        if model_type == "Inpaint":
            if type(validation_video) is str:
                original_width, original_height = Image.fromarray(cv2.VideoCapture(validation_video).read()[1]).size
            else:
                validation_video = np.array(validation_video.cpu().numpy() * 255, np.uint8)
                original_width, original_height = Image.fromarray(validation_video[0]).size
        else:
            if control_video is not None and type(control_video) is str:
                original_width, original_height = Image.fromarray(cv2.VideoCapture(control_video).read()[1]).size
            elif control_video is not None:
                control_video = np.array(control_video.cpu().numpy() * 255, np.uint8)
                original_width, original_height = Image.fromarray(control_video[0]).size
            else:
                original_width, original_height = 384 / 512 * base_resolution, 672 / 512 * base_resolution

            # Validate start_image (and end_image, though not used)
            if start_image is not None:
                start_image = [to_pil(img) for img in start_image]
                original_width, original_height = start_image[0].size if type(start_image) is list else Image.open(start_image).size
            
            if end_image is not None:
                end_image = [to_pil(img) for img in end_image]
                original_width, original_height = end_image[0].size if type(end_image) is list else Image.open(end_image).size

        closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
        height, width = [int(x / 16) * 16 for x in closest_size]

        # --- Load Scheduler & Setup TEA cache ---
        pipeline.scheduler = all_cheduler_dict[scheduler].from_pretrained(model_name, subfolder='scheduler')
        if enable_teacache:
            pipeline.transformer.enable_teacache(steps, teacache_threshold)

        generator = torch.Generator(device).manual_seed(seed)

        with torch.no_grad():
            # Adjust overall video_length if necessary (keep as is if video_length==1)
            if pipeline.vae.cache_mag_vae:
                video_length = int((video_length - 1) // pipeline.vae.mini_batch_encoder * pipeline.vae.mini_batch_encoder) + 1 if video_length != 1 else 1
            else:
                video_length = int(video_length // pipeline.vae.mini_batch_encoder * pipeline.vae.mini_batch_encoder) if video_length != 1 else 1

            # --- For non-Inpaint (using control_video) ---
            if model_type != "Inpaint":
                # Process the full control video once into latents.
                input_video, input_video_mask, clip_image = get_video_to_video_latent(control_video, video_length=video_length, sample_size=(height, width), fps=8)

                # Convert the provided start_image into latent form.
                if start_image is not None:
                    start_image_latent = get_image_latent(sample_size=(height, width), ref_image=start_image[0])
                else:
                    start_image_latent = None  # (Should normally be provided.)

                if camera_conditions is not None and len(camera_conditions) > 0: 
                    poses      = json.loads(camera_conditions)
                    cam_params = np.array([[float(x) for x in pose] for pose in poses])
                    cam_params = np.concatenate([np.zeros_like(cam_params[:, :1]), cam_params], 1)
                    control_camera_video = process_pose_params(cam_params, width=width, height=height)
                    control_camera_video = control_camera_video[:video_length].permute([3, 0, 1, 2]).unsqueeze(0)
                else:
                    control_camera_video = None

                # --- Helper: Allowed segment lengths based on VDM constraint ---
                def get_next_allowed_value(n):
                    # For n > 1, allowed values are 5, 9, 13, ... 49.
                    if n <= 1:
                        return 1
                    allowed = [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]
                    for v in allowed:
                        if n <= v:
                            return v
                    return 49

                # --- Helper: Process one chain (forward or reverse) ---
                def process_chain(chain_latents, chain_total, init_start_latent, direction="forward"):
                    """
                    chain_latents: latent slice for this chain, shape (b, c, T, h, w)
                    chain_total: desired total frames in the chain (including the initial stylized frame)
                    init_start_latent: the starting style latent for this chain.
                    direction: "forward" or "reverse" (for reverse, chain_latents is assumed already time-reversed)
                    Returns: a tensor of shape (b, c, chain_total, h, w)
                    """
                    # If no new frames are needed, just return the starting latent.
                    if chain_total == 1:
                        return init_start_latent.unsqueeze(2)  # add time dim

                    out_chain = []  # list to collect segments (each a tensor of shape (b, c, seg_frames, h, w))
                    current_start = init_start_latent  # current stylized latent to feed as start_image

                    # remaining frames to generate in this chain (including current_start)
                    remaining = chain_total  
                    offset = 0  # offset into chain_latents (control frames)
                    # new frames needed (excluding the starting frame)
                    new_frames_needed = chain_total - 1

                    while new_frames_needed > 0:
                        # For this segment, the desired total (including the start frame) is:
                        desired = min(new_frames_needed + 1, 49)
                        # Adjust to the next allowed value (pad if needed)
                        allowed = get_next_allowed_value(desired)
                        pad = allowed - desired  # number of extra frames to be generated (and later trimmed)

                        # Number of new frames we aim to get from this segment:
                        seg_new_frames = desired - 1  # note: pipeline call will produce 'allowed' frames (including start)
                        
                        # --- Extract corresponding control-video latents for this segment ---
                        # Determine how many control frames are available in chain_latents starting at offset.
                        available = chain_latents.shape[2] - offset
                        if available < seg_new_frames:
                            # If not enough, take what's available and then pad by repeating the last frame.
                            seg_control = chain_latents[:, :, offset:offset+available]
                            pad_tensor = seg_control[:, :, -1:].repeat(1, 1, seg_new_frames - available)
                            seg_control = torch.cat([seg_control, pad_tensor], dim=2)
                        else:
                            seg_control = chain_latents[:, :, offset:offset+seg_new_frames]
                        # If padding is needed to meet the allowed length, pad seg_control further.
                        if pad > 0:
                            pad_tensor = seg_control[:, :, -1:].repeat(1, 1, pad)
                            seg_control = torch.cat([seg_control, pad_tensor], dim=2)
                        # Now, seg_control has shape (b, c, allowed - 1, h, w)

                        # Call the pipeline for this segment.
                        seg_out = pipeline(
                            prompt,
                            video_length = allowed,
                            negative_prompt = negative_prompt,
                            height = height,
                            width = width,
                            generator = generator,
                            guidance_scale = cfg,
                            num_inference_steps = steps,
                            start_image = current_start,
                            # We pass the segment’s control video latents:
                            control_video = seg_control,
                            comfyui_progressbar = True,
                        ).frames  # shape: (b, c, allowed, h, w)

                        # Trim the output to remove padded frames.
                        # For the first segment, we want to keep 'desired' frames.
                        # For subsequent segments, the first frame is duplicate, so we drop it.
                        if offset == 0:
                            seg_trim = seg_out[:, :, :desired, :, :]
                        else:
                            seg_trim = seg_out[:, :, 1:desired, :, :]
                        # Update the current_start with the last frame of this segment.
                        current_start = seg_trim[:, :, -1, :, :]
                        # Append the segment (if not the first segment, we already dropped duplicate) 
                        out_chain.append(seg_trim)
                        # Advance the offset by the number of new frames produced (which is desired-1)
                        offset += (desired - 1)
                        new_frames_needed -= (desired - 1)
                    # Concatenate all segments along the time dimension.
                    # For the first segment, we kept the full segment (including starting frame).
                    chain_tensor = torch.cat(out_chain, dim=2)
                    return chain_tensor

                # --- Define the chains based on start_index ---
                # The final desired video has total 'video_length' frames.
                # The provided stylized frame is at position 'start_index'.
                # Forward chain: from start_index to end.
                # Reverse chain: from 0 to start_index.
                forward_total = video_length - start_index  # at least 1
                reverse_total = start_index + 1             # at least 1

                # Extract the corresponding slices from input_video latents.
                # For forward chain: frames from start_index onward.
                if forward_total > 0:
                    forward_latents = input_video[:, :, start_index:, :, :]
                else:
                    forward_latents = None

                # For reverse chain: frames before start_index.
                if reverse_total > 0:
                    reverse_latents = input_video[:, :, :start_index, :, :]
                    # Reverse the time dimension so that processing goes "backwards".
                    reverse_latents = reverse_latents.flip(dims=[2])
                else:
                    reverse_latents = None

                # Process forward chain (if more than just the stylized frame is needed).
                if forward_total > 1:
                    forward_chain = process_chain(forward_latents, forward_total, start_image_latent, direction="forward")
                else:
                    forward_chain = start_image_latent.unsqueeze(2)
                    
                # Process reverse chain (if available).
                if reverse_total > 1:
                    reverse_chain = process_chain(reverse_latents, reverse_total, start_image_latent, direction="reverse")
                    # After processing, flip reverse chain back into proper order.
                    reverse_chain = reverse_chain.flip(dims=[2])
                else:
                    reverse_chain = start_image_latent.unsqueeze(2)

                # --- Merge the two chains ---
                # Avoid duplicating the stylized frame (which appears as the last frame of reverse_chain and first frame of forward_chain).
                # If both chains exist, remove the first frame of forward_chain.
                if reverse_chain is not None and forward_chain is not None:
                    # Exclude the first frame of forward_chain.
                    merged = torch.cat([reverse_chain, forward_chain[:, :, 1:, :, :]], dim=2)
                elif forward_chain is not None:
                    merged = forward_chain
                else:
                    merged = reverse_chain

                # Rearrange the merged tensor to have shape (total_frames, height, width, channels)
                sample = merged  # shape (b, c, t, h, w)
            else:
                # Inpaint branch (unchanged)
                sample = pipeline(
                    prompt, 
                    video_length = video_length,
                    negative_prompt = negative_prompt,
                    height      = height,
                    width       = width,
                    generator   = generator,
                    guidance_scale = cfg,
                    num_inference_steps = steps,
                    video        = input_video,
                    mask_video   = input_video_mask,
                    clip_image   = clip_image, 
                    strength = float(denoise_strength),
                    comfyui_progressbar = True,
                ).frames

            # Rearrange frames from (b, c, t, h, w) to (b*t, h, w, c)
            videos = rearrange(sample, "b c t h w -> (b t) h w c")

            # --- Unmerge LoRA if necessary ---
            if not easyanimate_model.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(easyanimate_model.get("loras", []), easyanimate_model.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)
        return (videos,)


class EasyAnimateV5_V2VSampler(EasyAnimateV2VSampler):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "easyanimate_model": (
                    "EASYANIMATESMODEL", 
                ),
                "prompt": (
                    "STRING_PROMPT",
                ),
                "negative_prompt": (
                    "STRING_PROMPT",
                ),
                "video_length": (
                    "INT", {"default": 49, "min": 1, "max": 49, "step": 4}
                ),
                "base_resolution": (
                    [ 
                        512,
                        768,
                        960,
                        1024,
                    ], {"default": 768}
                ),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "steps": (
                    "INT", {"default": 25, "min": 1, "max": 200, "step": 1}
                ),
                "cfg": (
                    "FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.01}
                ),
                "denoise_strength": (
                    "FLOAT", {"default": 0.70, "min": 0.05, "max": 1.00, "step": 0.01}
                ),
                "scheduler": (
                    [ 
                        "Euler",
                        "Euler A",
                        "DPM++",
                        "PNDM",
                        "DDIM",
                        "Flow",
                    ],
                    {
                        "default": 'Flow'
                    }
                ),
                "teacache_threshold": ("FLOAT", {"default": 0.10, "min": 0.00, "max": 1.00, "step": 0.005}),
                "enable_teacache":([False, True],  {"default": True,}),
            },
            "optional":{
                "validation_video": ("IMAGE",),
                "control_video": ("IMAGE",),
                "camera_conditions": ("STRING", {"forceInput": True}),
                "ref_image": ("IMAGE",),
                "start_image": ("IMAGE",),
                "start_index": (
                    "INT", {"default": 0, "min": 0, "max": 300, "step": 1}
                ),
                "end_image": ("IMAGE",),
            },
        }

class CreateTrajectoryBasedOnKJNodes:
    # Modified from https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/curve_nodes.py
    # Modify to meet the trajectory control requirements of EasyAnimate.
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "createtrajectory"
    CATEGORY = "EasyAnimateWrapper"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coordinates": ("STRING", {"forceInput": True}),
                "masks": ("MASK", {"forceInput": True}),
            },
    } 

    def createtrajectory(self, coordinates, masks):
        # Define the number of images in the batch
        if len(coordinates) < 10:
            coords_list = []
            for coords in coordinates:
                coords = json.loads(coords.replace("'", '"'))
                coords_list.append(coords)
        else:
            coords = json.loads(coordinates.replace("'", '"'))
            coords_list = [coords]

        _, frame_height, frame_width = masks.size()
        heatmap = gen_gaussian_heatmap()

        circle_size = int(50 * ((frame_height * frame_width) / (1280 * 720)) ** (1/2))

        images_list = []
        for coords in coords_list:
            _images_list = []
            for i in range(len(coords)):
                _image = np.zeros((frame_height, frame_width, 3))
                center_coordinate = [coords[i][key] for key in coords[i]]

                y1 = max(center_coordinate[1] - circle_size, 0)
                y2 = min(center_coordinate[1] + circle_size, np.shape(_image)[0] - 1)
                x1 = max(center_coordinate[0] - circle_size, 0)
                x2 = min(center_coordinate[0] + circle_size, np.shape(_image)[1] - 1)
                
                if x2 - x1 > 3 and y2 - y1 > 3:
                    need_map = cv2.resize(heatmap, (x2 - x1, y2 - y1))[:, :, None]
                    _image[y1:y2, x1:x2] = np.maximum(need_map.copy(), _image[y1:y2, x1:x2])
                
                _image = np.expand_dims(_image, 0) / 255
                _images_list.append(_image)
            images_list.append(np.concatenate(_images_list, axis=0))
            
        out_images = torch.from_numpy(np.max(np.array(images_list), axis=0))
        return (out_images, )

class ImageMaximumNode:
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "imagemaximum"
    CATEGORY = "EasyAnimateWrapper"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_1": ("IMAGE",),
                "video_2": ("IMAGE",),
            },
    } 

    def imagemaximum(self, video_1, video_2):
        length_1, h_1, w_1, c_1 = video_1.size()
        length_2, h_2, w_2, c_2 = video_2.size()
        
        if h_1 != h_2 or w_1 != w_2:
            video_1, video_2 = video_1.permute([0, 3, 1, 2]), video_2.permute([0, 3, 1, 2])
            video_2 = F.interpolate(video_2, video_1.size()[-2:])
            video_1, video_2 = video_1.permute([0, 2, 3, 1]), video_2.permute([0, 2, 3, 1])

        if length_1 > length_2:
            outputs = torch.maximum(video_1[:length_2], video_2)
        else:
            outputs = torch.maximum(video_1, video_2[:length_1])
        return (outputs, )

class CameraBasicFromChaoJie:
    # Copied from https://github.com/chaojie/ComfyUI-CameraCtrl-Wrapper/blob/main/nodes.py
    # Since ComfyUI-CameraCtrl-Wrapper requires a specific version of diffusers, which is not suitable for us. 
    # The code has been copied into the current repository.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pose":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "speed":("FLOAT",{"default":1.0}),
                "video_length":("INT",{"default":16}),
            },
        }

    RETURN_TYPES = ("CameraPose",)
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self,camera_pose,speed,video_length):
        camera_dict = {
                "motion":[camera_pose],
                "mode": "Basic Camera Poses",  # "First A then B", "Both A and B", "Custom"
                "speed": speed,
                "complex": None
                } 
        motion_list = camera_dict['motion']
        mode = camera_dict['mode']
        speed = camera_dict['speed'] 
        angle = np.array(CAMERA[motion_list[0]]["angle"])
        T = np.array(CAMERA[motion_list[0]]["T"])
        RT = get_camera_motion(angle, T, speed, video_length)
        return (RT,)

class CameraCombineFromChaoJie:
    # Copied from https://github.com/chaojie/ComfyUI-CameraCtrl-Wrapper/blob/main/nodes.py
    # Since ComfyUI-CameraCtrl-Wrapper requires a specific version of diffusers, which is not suitable for us. 
    # The code has been copied into the current repository.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pose1":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "camera_pose2":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "camera_pose3":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "camera_pose4":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "speed":("FLOAT",{"default":1.0}),
                "video_length":("INT",{"default":16}),
            },
        }

    RETURN_TYPES = ("CameraPose",)
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self,camera_pose1,camera_pose2,camera_pose3,camera_pose4,speed,video_length):
        angle = np.array(CAMERA[camera_pose1]["angle"]) + np.array(CAMERA[camera_pose2]["angle"]) + np.array(CAMERA[camera_pose3]["angle"]) + np.array(CAMERA[camera_pose4]["angle"])
        T = np.array(CAMERA[camera_pose1]["T"]) + np.array(CAMERA[camera_pose2]["T"]) + np.array(CAMERA[camera_pose3]["T"]) + np.array(CAMERA[camera_pose4]["T"])
        RT = get_camera_motion(angle, T, speed, video_length)
        return (RT,)

class CameraJoinFromChaoJie:
    # Copied from https://github.com/chaojie/ComfyUI-CameraCtrl-Wrapper/blob/main/nodes.py
    # Since ComfyUI-CameraCtrl-Wrapper requires a specific version of diffusers, which is not suitable for us. 
    # The code has been copied into the current repository.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pose1":("CameraPose",),
                "camera_pose2":("CameraPose",),
            },
        }

    RETURN_TYPES = ("CameraPose",)
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self,camera_pose1,camera_pose2):
        RT = combine_camera_motion(camera_pose1, camera_pose2)
        return (RT,)

class CameraTrajectoryFromChaoJie:
    # Copied from https://github.com/chaojie/ComfyUI-CameraCtrl-Wrapper/blob/main/nodes.py
    # Since ComfyUI-CameraCtrl-Wrapper requires a specific version of diffusers, which is not suitable for us. 
    # The code has been copied into the current repository.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pose":("CameraPose",),
                "fx":("FLOAT",{"default":0.474812461, "min": 0, "max": 1, "step": 0.000000001}),
                "fy":("FLOAT",{"default":0.844111024, "min": 0, "max": 1, "step": 0.000000001}),
                "cx":("FLOAT",{"default":0.5, "min": 0, "max": 1, "step": 0.01}),
                "cy":("FLOAT",{"default":0.5, "min": 0, "max": 1, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING","INT",)
    RETURN_NAMES = ("camera_trajectory","video_length",)
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self,camera_pose,fx,fy,cx,cy):
        #print(camera_pose)
        camera_pose_list=camera_pose.tolist()
        trajs=[]
        for cp in camera_pose_list:
            traj=[fx,fy,cx,cy,0,0]
            traj.extend(cp[0])
            traj.extend(cp[1])
            traj.extend(cp[2])
            trajs.append(traj)
        return (json.dumps(trajs),len(trajs),)

NODE_CLASS_MAPPINGS = {
    "TextBox": TextBox,
    "EasyAnimate_TextBox": EasyAnimate_TextBox,
    "LoadEasyAnimateModel": LoadEasyAnimateModel,
    "LoadEasyAnimateLora": LoadEasyAnimateLora,
    "EasyAnimateI2VSampler": EasyAnimateI2VSampler,
    "EasyAnimateT2VSampler": EasyAnimateT2VSampler,
    "EasyAnimateV2VSampler": EasyAnimateV2VSampler,
    "EasyAnimateV5_I2VSampler": EasyAnimateV5_I2VSampler,
    "EasyAnimateV5_T2VSampler": EasyAnimateV5_T2VSampler,
    "EasyAnimateV5_V2VSampler": EasyAnimateV5_V2VSampler,
    "CreateTrajectoryBasedOnKJNodes": CreateTrajectoryBasedOnKJNodes,
    "CameraBasicFromChaoJie": CameraBasicFromChaoJie,
    "CameraTrajectoryFromChaoJie": CameraTrajectoryFromChaoJie,
    "CameraJoinFromChaoJie": CameraJoinFromChaoJie,
    "CameraCombineFromChaoJie": CameraCombineFromChaoJie,
    "ImageMaximumNode": ImageMaximumNode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "TextBox": "TextBox",
    "EasyAnimate_TextBox": "EasyAnimate_TextBox",
    "LoadEasyAnimateModel": "Load EasyAnimate Model",
    "LoadEasyAnimateLora": "Load EasyAnimate Lora",
    "EasyAnimateI2VSampler": "EasyAnimate Sampler for Image to Video",
    "EasyAnimateT2VSampler": "EasyAnimate Sampler for Text to Video",
    "EasyAnimateV2VSampler": "EasyAnimate Sampler for Video to Video",
    "EasyAnimateV5_I2VSampler": "EasyAnimateV5 Sampler for Image to Video",
    "EasyAnimateV5_T2VSampler": "EasyAnimateV5 Sampler for Text to Video",
    "EasyAnimateV5_V2VSampler": "EasyAnimateV5 Sampler for Video to Video",
    "CreateTrajectoryBasedOnKJNodes": "Create Trajectory Based On KJNodes",
    "CameraBasicFromChaoJie": "CameraBasicFromChaoJie",
    "CameraTrajectoryFromChaoJie": "CameraTrajectoryFromChaoJie",
    "CameraJoinFromChaoJie": "CameraJoinFromChaoJie",
    "CameraCombineFromChaoJie": "CameraCombineFromChaoJie",
    "ImageMaximumNode": "ImageMaximumNode",
} 
