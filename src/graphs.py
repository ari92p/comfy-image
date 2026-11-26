"""ComfyUI API graphs for the two model families. Pure functions: they return a dict that ComfyUI's /prompt endpoint accepts.

krea2        Krea 2 Turbo: new images, stacked LoRAs, up to three style reference images
qwen21       Qwen-Image 2.1: editing, identity from a character image, style from an image, any mix of up to four images

File names are the ones ComfyUI lists; use comfy.find_name() to resolve them first.
"""

import random

QWEN = {"unet": "", "clip": "", "vae": "", "steps": 25, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "resolution": 1024, "cache": "auto"}
KREA = {"unet": "", "clip": "", "vae": "", "steps": 8, "sampler": "euler", "scheduler": "simple"}


def _seed(seed):
    return random.randrange(2**32) if seed is None else seed


def build_qwen(prompt, images=(), width=None, height=None, seed=None, negative="", prefix="image", **overrides):
    """Qwen-Image 2.1. images: files already uploaded to ComfyUI, referred to in the prompt as <image1>, <image2>, ...

    Without width/height an edit keeps the size of image 1 and a text-to-image job is `resolution` x `resolution`."""
    o = {**QWEN, **overrides}
    g = {
        "unet": ({"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": o["unet"]}}
                 if o["unet"].endswith(".gguf") else
                 {"class_type": "UNETLoader", "inputs": {"unet_name": o["unet"], "weight_dtype": "default"}}),
        "cache": {"class_type": "QwenImage21Cache", "inputs": {"model": ["unet", 0], "device": o["cache"], "dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": o["clip"], "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": o["vae"]}},
        "encode": {"class_type": "TextEncodeQwenImage21",
                   "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt,
                              "negative_prompt": negative, "resolution": o["resolution"]}},
    }
    for i, name in enumerate(images, 1):
        g[f"load{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g["encode"]["inputs"][f"images.image_{i}"] = [f"load{i}", 0]
    latent = ["encode", 2]
    if width and height:
        # Qwen's latent is 64 channels at 1/16; ComfyUI widens this empty 128-channel one
        g["empty"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
        latent = ["empty", 0]
    g["sample"] = {"class_type": "KSampler",
                   "inputs": {"model": ["cache", 0], "positive": ["encode", 0], "negative": ["encode", 1], "latent_image": latent,
                              "seed": _seed(seed), "steps": o["steps"], "cfg": o["cfg"], "sampler_name": o["sampler"],
                              "scheduler": o["scheduler"], "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}}
    return g


def build_krea(prompt, loras=(), style_images=(), width=1024, height=1024, seed=None, prefix="image",
               style_lora="", **overrides):
    """Krea 2 Turbo. loras: (file, strength) pairs, stacked in order. style_images: up to three uploaded images whose
    look Krea copies; that adds `style_lora` (the style reference LoRA) and the reference encoder."""
    o = {**KREA, **overrides}
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": o["unet"], "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": o["clip"], "type": "krea2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": o["vae"]}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
    }
    model = ["unet", 0]
    loras = list(loras)
    if style_images:
        loras.append((style_lora, 1.0))
    for i, (name, strength) in enumerate(loras, 1):
        g[f"lora{i}"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": name, "strength_model": strength}}
        model = [f"lora{i}", 0]
    if style_images:
        g["shift"] = {"class_type": "ModelSamplingFlux",
                      "inputs": {"model": model, "max_shift": 1.15, "base_shift": 0.5, "width": width, "height": height}}
        model = ["shift", 0]
        g["encode"] = {"class_type": "TextEncodeQwenImageEditPlus",
                       "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt}}
        for i, name in enumerate(style_images[:3], 1):
            g[f"load{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            g["encode"]["inputs"][f"image{i}"] = [f"load{i}", 0]
        g["positive"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
                         "inputs": {"conditioning": ["encode", 0], "reference_latents_method": "index_timestep_zero"}}
    else:
        g["positive"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}}
    g["negative"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["positive", 0]}}
    g["sample"] = {"class_type": "KSampler",
                   "inputs": {"model": model, "positive": ["positive", 0], "negative": ["negative", 0],
                              "latent_image": ["latent", 0], "seed": _seed(seed), "steps": o["steps"], "cfg": 1.0,
                              "sampler_name": o["sampler"], "scheduler": o["scheduler"], "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}}
    return g
