"""Everything this tool pins: the ComfyUI version, the custom node, PyTorch, and the model files with their exact sizes and checksums.

These are the versions and files the workflows were built and tested with. Nothing here is downloaded unless you say yes
(`--setup`, or the question the guided mode asks). Model files come straight from their authors' Hugging Face repositories.
"""

COMFYUI = {"repo": "https://github.com/Comfy-Org/ComfyUI.git", "ref": "v0.41.0", "commit": "98cc0bf84b7cb1383654c5b402e7612d875d6d22"}
GGUF_NODE = {"repo": "https://github.com/leejet/ComfyUI-GGUF.git", "commit": "373048b8403a7820620065210a691263d4da0a61"}
PYTHON = "3.13"
TORCH = ["torch==2.14.0", "torchvision==0.29.0", "torchaudio==2.11.0"]
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu130"   # NVIDIA on Windows and Linux; everywhere else the default index
LAUNCH_FLAGS = ["--disable-auto-launch", "--preview-method", "none"]
HF = "https://huggingface.co"   # IMAGE_HF_ENDPOINT overrides it (mirrors, tests)

# name -> (title, terms in one line, link). The tool prints these before it downloads anything.
LICENSES = {
    "krea2": ("Krea 2 Community License", "permissive use; commercial licensing is offered separately by Krea",
              "https://www.krea.ai/krea-2-licensing"),
    "qwen": ("Qwen Research License", "NON-COMMERCIAL use of the model only; commercial use needs a separate license from Qwen",
             "https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE"),
}

FILES = {
    "krea2-unet": dict(repo="Comfy-Org/Krea-2", rev="588feaf955b08b9ac4723eeb751752ab045dba71", path="diffusion_models/krea2_turbo_fp8_scaled.safetensors",
        dest="diffusion_models/krea2_turbo_fp8_scaled.safetensors", size=13141730784, sha256="eb4dd8c612cfd10f64f25b057e6e6bbcb5737c94a7372177e456dbf7579502f1", license="krea2"),
    "krea2-clip": dict(repo="Comfy-Org/Krea-2", rev="588feaf955b08b9ac4723eeb751752ab045dba71", path="text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
        dest="text_encoders/qwen3vl_4b_fp8_scaled.safetensors", size=5242467968, sha256="54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094", license="krea2"),
    "qwen-image-vae": dict(repo="Comfy-Org/Krea-2", rev="588feaf955b08b9ac4723eeb751752ab045dba71", path="vae/qwen_image_vae.safetensors",
        dest="vae/qwen_image_vae.safetensors", size=253806246, sha256="a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f", license="krea2"),
    "krea2-style": dict(repo="Comfy-Org/Krea-2", rev="588feaf955b08b9ac4723eeb751752ab045dba71", path="loras/krea2_style_reference.safetensors",
        dest="loras/krea2/krea2_style_reference.safetensors", size=457111760, sha256="f50df5a9e62e4be8aa926a63dd5bb1a64770c4004f763c1208007ae13daa82b8", license="krea2"),
    "qwen21-unet": dict(repo="unsloth/Qwen-Image-2.1-GGUF", rev="2c31ccd392b367a6637841a143813320a02dff55", path="qwen-image-2.1-Q6_K.gguf",
        dest="diffusion_models/qwen-image-2.1-Q6_K.gguf", size=6271551200, sha256="2ead1ee665e8ebc5fec517d287a167f5c6a7e985acbf3b1b93143437a04616ac", license="qwen"),
    "qwen21-clip": dict(repo="Comfy-Org/Qwen-Image-2.1", rev="aacf31a6d48e707f352ecba714cee0680da3479b", path="text_encoders/qwen3vl_8b_int8_convrot.safetensors",
        dest="text_encoders/qwen3vl_8b_int8_convrot.safetensors", size=9350798360, sha256="8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f", license="qwen"),
    "qwen21-vae": dict(repo="Comfy-Org/Qwen-Image-2.1", rev="aacf31a6d48e707f352ecba714cee0680da3479b", path="vae/qwen_image_2.1_vae_bf16.safetensors",
        dest="vae/qwen_image_2.1_vae_bf16.safetensors", size=675509688, sha256="bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9", license="qwen"),
}

# the files each model family loads by default, and what the optional features add
FAMILIES = {
    "krea2": ["krea2-unet", "krea2-clip", "qwen-image-vae"],
    "qwen21": ["qwen21-unet", "qwen21-clip", "qwen21-vae"],
}
EXTRAS = {"krea2-style": "style images with Krea 2 (--style)"}
# which catalog file fixes a missing (family, kind); kinds as used by pipeline.Engine
FIXES = {("krea2", "unet"): ["krea2-unet"], ("krea2", "clip"): ["krea2-clip"], ("krea2", "vae"): ["qwen-image-vae"],
         ("qwen21", "unet"): ["qwen21-unet"], ("qwen21", "clip"): ["qwen21-clip"], ("qwen21", "vae"): ["qwen21-vae"]}
ALIASES = {"krea": "krea2", "krea2": "krea2", "qwen": "qwen21", "qwen21": "qwen21"}
