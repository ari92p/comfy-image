# easy-image

AI images on your own computer: new images from text, edits of existing images, a character from a reference picture in a new scene,
a style copied from a picture, and LoRAs. No account, no API key. It brings its own ComfyUI and, after asking, the model files.

## Quick start

```bash
git clone https://github.com/ari92p/easy-image && cd easy-image
src/setup.sh     # once (Windows: double-click src\setup.cmd)
./easy-image         # Windows: easy-image.cmd
```

`src/setup.sh` sets everything up in this folder, nothing global: it fetches `uv`, `uv` fetches Python, then it installs ComfyUI and downloads
the models. It shows what it will do (about 35 GB of model files and a few GB of PyTorch, with the licenses) and asks **one question**. After that:

```bash
./easy-image                       # guided: asks what you want and prints the one-line command
./easy-image "a fox in the snow"   # or give the prompt directly (it also reads stdin)
```

The result is written to `out/`. You need a graphics card (an NVIDIA card with 16 GB is the tested setup). Run `src/setup.sh` again any time:
it only adds what is missing.

The examples below write `easy-image`: that is `./easy-image` (Windows: `easy-image.cmd`) in this folder. For a command that works from anywhere: `ln -s "$PWD/easy-image" ~/.local/bin/easy-image`.

## Run it as command

```bash
easy-image "a lighthouse at dawn" --aspect 16:9
easy-image "a fox in the snow" --lora watercolor:0.7                      # a LoRA from the loras folder
easy-image --edit photo.png "make the jacket red"                         # change an image
easy-image --character me.png --save-character me "me as an astronaut"    # keep a person's look, saved as "me"
easy-image --character me --style painting.png "me reading in a library"  # saved characters work by name
easy-image --style painting.png "a quiet harbour at night"                # copy a picture's look
```

| Option | What it does |
| --- | --- |
| `"text"`, `--prompt`, `--prompt-file FILE` or piped in | what the image shows; with `--edit`, what to change |
| `--edit IMAGE` | change this image |
| `--character IMAGE\|NAME`, `--style IMAGE\|NAME` | keep a person's look / copy a picture's look (up to 3 each); `--save-character NAME`, `--save-style NAME` keep them, `--list-characters`, `--list-styles` show them |
| `--lora NAME[:STRENGTH]` | a LoRA (`--list-loras`); repeat to stack |
| `--model FILE`, `--qwen-model FILE`, `--clip FILE`, `--vae FILE` | use other model files; what you leave out is detected |
| `--aspect 16:9`, `--size 1.5`, `--quality fast\|normal\|high` | shape, resolution, steps |
| `--seed N`, `--count N` | same seed and settings give the same image; `--count` makes several |
| `--setup [krea\|qwen\|krea-style]` | what `src/setup.sh` runs: everything by default, or only the models you name (`--dry-run` shows the plan) |
| `--configure` | advanced: use a ComfyUI or a models folder you already have, and remember it |
| `-y`, `--accept-licenses`, `--offline` | say yes in advance; accept the licenses; never download |
| `--models-dir DIR`, `--comfy URL` | use a models folder or a ComfyUI you already have |
| `--check`, `--list-models` | which models are ready; which files can be downloaded |
| `-v` | show everything the setup and ComfyUI print |

More options (`--ref`, `--two-step`, `--fidelity`, `--eyes`, `--keep-running`) are in `--help`.
`--configure` keeps your choice in `config.json`; saved pictures in `characters/` and `styles/` (none of it is part of the repository).

## Which model does what

- **Krea 2** generates new images (plain sentences, LoRAs; `--style` uses its style reference). It is the default for new images.
- **Qwen-Image 2.1** edits images and mixes generating with references: `--edit`, `--character` and `--ref` always use it. A character in a LoRA style
  is painted in two steps: Krea 2 renders the look, Qwen paints the character into it (with a second pass that puts the original's face back).

These two are the baseline the tool is built and tested with. **You are free to use other models:** put the files into the models folder (`checkpoints/`, `diffusion_models/`,
`loras/`, ...) and name them with `--model`, `--qwen-model`, `--clip` or `--vae`; a part of the file name is enough, and files that look like a family's are found by themselves.
The tool does not choose for you where you name something, and in guided mode you pick from the files that are there. A LoRA only works with the model it was trained for;
Qwen modes ignore LoRAs and say so. Always give a character's *original* picture, not a generated one, and don't describe the character in the prompt. For edits say only what changes.

## Run it as service

```bash
easy-image --serve                      # http://127.0.0.1:8192
easy-image --serve --host 0.0.0.0       # reachable from other computers (no login, use a firewall)
```

A small HTTP service for other programs: `POST /generate` (JSON in, `image/png` out), saved characters and styles by name, `/health`, `/engines`, `/loras`, `/cancel`.
The endpoints are listed at the top of [src/server.py](src/server.py). On start it downloads the models of the setup, then answers one job at a time,
stops its ComfyUI after 5 minutes without requests to free the graphics card, and cancels and restarts it when a job hangs.
Options: `--port`, `--models-dir`, `--offline`. `src/start_server.cmd` (Windows) and `src/start_server.sh` (Mac/Linux) start it.

## Models, downloads and files

Everything stays in the repository folder (`.venvs/`, `comfyui/`, `out/`, `characters/`, `styles/`, none of it goes into git), nothing is installed globally, and nothing is downloaded before you say yes. `comfyui/` holds ComfyUI **v0.41.0**, the
ComfyUI-GGUF node and a private Python 3.13 environment with PyTorch 2.14; it runs on 127.0.0.1 port 8199 while needed. `comfyui/models/` (or your `--models-dir`) holds the
model files; a file you already have, under any folder name, is recognised and not downloaded again. Delete `comfyui/` to start over.

| Model | Files (checked by size and SHA-256, from a pinned revision) | License |
| --- | --- | --- |
| Krea 2, 18.6 GB ([Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2)) | `krea2_turbo_fp8_scaled`, `qwen3vl_4b_fp8_scaled`, `qwen_image_vae`; for `--style` also `krea2_style_reference` | [Krea 2 Community License](https://www.krea.ai/krea-2-licensing): permissive use, commercial licensing separate |
| Qwen-Image 2.1, 16.3 GB ([unsloth GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF), [Comfy-Org](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)) | `qwen-image-2.1-Q6_K.gguf`, `qwen3vl_8b_int8_convrot`, `qwen_image_2.1_vae_bf16` | Qwen Research License: **non-commercial** use of the model |

The tool prints these licenses before it downloads and asks you to accept them. Models and LoRAs you add yourself carry their own license (often "no commercial use");
this tool includes and recommends none. ComfyUI is GPL-3.0, ComfyUI-GGUF Apache-2.0, this repository MIT.

## Optional: Ollama

You can ignore this. [Ollama](https://ollama.com) is a separate program that runs small AI models on your computer; this tool never installs it. If it happens to be installed and
running, the tool uses it for two small jobs: it looks at your character picture once and writes down the face and hair so the character stays closer to the original, and it
turns a short edit wish like "mach den Mantel rot" into a clear English instruction. If Ollama is not there, everything works the same, just without those two helpers.
Switch it off with `--no-ollama`.

Don't make images of real people without their consent, don't pass generated images off as photographs of real events, and respect the licenses above.
