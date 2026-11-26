"""The image logic: picks the model family, finds the model files ComfyUI has, and runs the right chain of graphs.

Krea 2 generates images. Qwen-Image 2.1 edits images and combines generating with references.
Modes (chosen from what you give it):
  krea2      a new image from a prompt; krea2 can take style images and LoRAs
  qwen       Qwen-Image 2.1 paints a new scene with a character and/or a style image as references
  krea2+qwen two steps: Krea 2 renders the look (LoRAs, model file, style images), Qwen paints the character into it
  edit       Qwen-Image 2.1 changes an existing image (optionally keeping a character's identity)
"""

import dataclasses
import json
import random
import time

import catalog
import graphs
from comfy import Cancelled, ComfyError, find_name, norm, search_names
from ollama import to_png

# about one megapixel, multiples of 16
ASPECTS = {
    "1:1": (1024, 1024), "4:5": (896, 1120), "3:4": (896, 1184), "2:3": (832, 1248), "9:16": (768, 1344),
    "5:4": (1120, 896), "4:3": (1184, 896), "3:2": (1248, 832), "16:9": (1344, 768), "21:9": (1536, 640),
}
# sampling steps per quality level
STEPS = {"krea2": {"fast": 6, "normal": 8, "high": 12}, "qwen21": {"fast": 18, "normal": 25, "high": 40}}
FAMILY = {"auto": "krea2", "krea": "krea2", "krea2": "krea2", "qwen": "qwen21", "qwen21": "qwen21"}
TITLES = {"krea2": "Krea 2 Turbo", "qwen21": "Qwen-Image 2.1"}

# Default files per family, as they are named in ComfyUI's model folders. `unet` lists candidates in order of preference.
FILES = {
    "krea2": {"unet": ["krea2_turbo_fp8_scaled.safetensors"], "clip": "qwen3vl_4b_fp8_scaled.safetensors",
              "vae": "qwen_image_vae.safetensors", "style_lora": "krea2_style_reference.safetensors"},
    "qwen21": {"clip": "qwen3vl_8b_int8_convrot.safetensors", "vae": "qwen_image_2.1_vae_bf16.safetensors"},
}
FOLDER = {"unet": "models/diffusion_models", "clip": "models/text_encoders", "vae": "models/vae", "lora": "models/loras"}
NODES = {
    "krea2": ["UNETLoader", "CLIPLoader", "VAELoader", "EmptyLatentImage", "CLIPTextEncode", "ConditioningZeroOut",
              "LoraLoaderModelOnly", "KSampler", "VAEDecode", "SaveImage"],
    "qwen21": ["QwenImage21Cache", "TextEncodeQwenImage21", "EmptyFlux2LatentImage", "CLIPLoader", "VAELoader", "LoadImage",
               "KSampler", "VAEDecode", "SaveImage"],
}
STYLE_NODES = ["ModelSamplingFlux", "TextEncodeQwenImageEditPlus", "FluxKontextMultiReferenceLatentMethod", "LoadImage"]


def find_loose(text, installed):
    """Like find_name, but also accepts a name without extension or a unique part of the name. None if nothing or several match."""
    exact = find_name(text, installed)
    if exact:
        return exact
    hits = search_names(text, installed)
    return hits[0] if len(hits) == 1 else None


class Missing(Exception):
    """A node or model file this job needs is not in ComfyUI. The message says what and where to put it; `ids` names the
    files from catalog.py that would fix it (the tool can download those when it manages ComfyUI)."""

    def __init__(self, message, ids=()):
        super().__init__(message)
        self.ids = list(dict.fromkeys(ids))


@dataclasses.dataclass
class Job:
    prompt: str
    engine: str = "auto"                 # auto | krea | qwen
    model: str = ""                      # the Krea 2 diffusion model file to use instead of the detected one
    qwen_model: str = ""                 # Qwen file; default: the first Qwen-Image 2.1 file ComfyUI has
    clip: str = ""                       # text encoder / VAE of the chosen family, if you do not want the detected ones
    vae: str = ""
    loras: list = dataclasses.field(default_factory=list)       # (name as ComfyUI lists it, strength)
    characters: list = dataclasses.field(default_factory=list)  # image bytes, at most 3
    styles: list = dataclasses.field(default_factory=list)      # image bytes, at most 3
    refs: list = dataclasses.field(default_factory=list)        # extra reference images, at most 2
    edit: bytes = None                   # the image to change
    aspect: str = "1:1"
    size: str = "1"                      # 1 | 1.5 | 2 (times about one megapixel)
    quality: str = "normal"              # fast | normal | high
    seed: int = None
    fidelity: str = "high"               # character + style: re-align the face with the original afterwards
    eyes: str = "off"                    # extra pass that evens out the eyes
    two_step: bool = False               # character + one style image: let the family render the style first


def sniff_ext(data):
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "png"


# ---------- character facts (from Ollama's JSON description) ----------

def char_facts(descr):
    try:
        d = json.loads(descr)
        return d if isinstance(d, dict) else {}
    except (ValueError, TypeError):
        return {}


def char_text(descr):
    f = char_facts(descr)
    if not f:
        return descr or ""
    # clothing is not part of who the character is: the request decides it, so it never appears in the identity text
    return "; ".join(f"{k}: {v}" for k, v in f.items() if v and v != "not visible" and k not in ("outfit", "accessories"))


def ears_pointed(descr):
    ears = (char_facts(descr).get("ears") or "").lower()
    return "pointed" in ears or "elf" in ears


def ear_clause(descr, n):
    """What to say about ears. Only when the description says something about them; without a description we stay silent."""
    ears = (char_facts(descr).get("ears") or "").lower()
    if not ears:
        return ""
    if ears_pointed(descr):
        return f"The character keeps the pointed ears shown in <image{n}>."
    return (f"The character has normal round human ears exactly as in <image{n}>; never give them pointed or elf ears, "
            f"even if another reference image shows pointed ears.")


class Engine:
    def __init__(self, comfy, ollama=None):
        self.comfy = comfy
        self.ollama = ollama          # None = no helpers
        self.on_progress = None       # callable(step, steps)

    def cancel(self):
        self.comfy.cancel.set()

    # ---------- what ComfyUI has ----------

    def _unets(self):
        return self.comfy.choices("UNETLoader", "unet_name")

    def _ggufs(self):
        return self.comfy.choices("UnetLoaderGGUF", "unet_name")

    def loras(self):
        return self.comfy.choices("LoraLoaderModelOnly", "lora_name")

    def resolve_lora(self, text):
        """A LoRA typed loosely (bare name, with or without .safetensors, or part of the name) -> the name ComfyUI uses."""
        hits = search_names(text, self.loras())
        if not hits:
            raise Missing(f"no LoRA matches '{text}' (see --list-loras)")
        if len(hits) > 1:
            raise Missing(f"'{text}' matches several LoRAs: {', '.join(hits[:8])}. Type more of the name.")
        return hits[0]

    @staticmethod
    def lora_family(name):
        """A LoRA only runs with the model it was trained for. Folders named after a family (krea2/, qwen21/) tell which."""
        parts = norm(name).lower().split("/")[:-1]
        for p in parts:
            if p in ("krea2", "qwen21"):
                return p
        return None

    # What a file name must look like to belong to a family. Only used to find candidates; anything can be chosen by name.
    MATCH = {
        "krea2": {"unet": lambda b: any(k in b for k in ("krea2", "krea-2", "krea_2")),
                  "clip": lambda b: "qwen3vl_4b" in b, "vae": lambda b: "qwen_image_vae" in b},
        "qwen21": {"unet": lambda b: "qwen-image-2.1" in b or "qwen_image_2.1" in b,
                   "clip": lambda b: "qwen3vl_8b" in b, "vae": lambda b: "qwen_image_2.1_vae" in b},
    }

    def _installed(self, kind):
        return {"unet": lambda: self._unets() + self._ggufs(), "clip": lambda: self.comfy.choices("CLIPLoader", "clip_name"),
                "vae": lambda: self.comfy.choices("VAELoader", "vae_name")}[kind]()

    def candidates(self, family, kind="unet"):
        """The installed files that look like they belong to this family, best guess first. Never filters what you may pick yourself."""
        installed = self._installed(kind)
        want = FILES.get(family, {}).get({"unet": "unet", "clip": "clip", "vae": "vae"}[kind])
        wants = [want] if isinstance(want, str) else list(want or [])
        if family == "krea2" and kind == "unet" and catalog.KREA_GGUF:
            wants.insert(0, catalog.FILES["krea2-unet-gguf"]["dest"].rsplit("/", 1)[-1])   # Mac: the GGUF file first
        first = [find_name(w, installed) for w in wants]
        hits = [n for n in installed if self.MATCH[family][kind](norm(n).lower().rsplit("/", 1)[-1])]

        def key(n):
            low = norm(n).lower().rsplit("/", 1)[-1]
            return ("uc" in low.replace("_", "-").split("-"), catalog.TIER_TAGS[catalog.TIER] not in low, low) if family == "qwen21" else (False, False, low)
        return [n for n in dict.fromkeys([f for f in first if f] + sorted(hits, key=key))]

    def qwen_files(self):
        return self.candidates("qwen21", "unet")

    def pick(self, family, model="", clip="", vae=""):
        """The files to load for this family -> {unet, clip, vae, ...}. What you name is used as it is; anything you leave
        out is detected among the installed files. Raises Missing with a helpful message."""
        miss, out, ids = [], {}, []

        def choose(kind, given, label, folder):
            installed = self._installed(kind)
            if given:
                hit = find_loose(given, installed)
                if not hit:
                    miss.append(f"{label} '{given}' (ComfyUI/{folder})")
                return hit
            cands = self.candidates(family, kind)
            if not cands:
                miss.append(f"a {label} for {TITLES[family]} (ComfyUI/{folder})")
                ids.extend(catalog.FIXES.get((family, kind), []))
                return None
            return cands[0]

        out["unet"] = choose("unet", model, "model file" if family != "qwen21" else "Qwen-Image 2.1 model file", FOLDER["unet"])
        out["clip"] = choose("clip", clip, "text encoder", FOLDER["clip"])
        out["vae"] = choose("vae", vae, "VAE", FOLDER["vae"])
        if (out["unet"] or "").endswith(".gguf") and not self.comfy.node("UnetLoaderGGUF"):
            miss.append("the ComfyUI-GGUF custom node (needed for .gguf models)")
        if family == "krea2":
            out["style_lora"] = find_name(FILES["krea2"]["style_lora"], self.loras())   # optional: only style images need it
        for n in NODES[family]:
            if not self.comfy.node(n):
                miss.append(f"ComfyUI node {n} (update ComfyUI)")
        if miss:
            raise Missing(f"{TITLES[family]} cannot run, missing: " + "; ".join(miss), ids)
        return out

    def check(self):
        """-> {family: {"ok": bool, "files": {...}, "problems": str, "styles": bool}} for every family."""
        res = {}
        for fam in ("krea2", "qwen21"):
            try:
                files = self.pick(fam)
                entry = {"ok": True, "files": files, "problems": ""}
            except Missing as e:
                entry = {"ok": False, "files": {}, "problems": str(e)}
            if fam == "krea2":
                entry["styles"] = bool(entry["ok"] and entry["files"].get("style_lora")
                                       and all(self.comfy.node(n) for n in STYLE_NODES))
            res[fam] = entry
        return res

    # ---------- planning ----------

    @staticmethod
    def plan_mode(job, family):
        if job.edit:
            return "edit"
        if family == "qwen21":
            return "qwen"
        people = bool(job.characters) or bool(job.refs)
        styled = bool(job.loras) or bool(job.model) or len(job.styles) > 1 or job.two_step
        if not people:
            return family
        if not styled and job.engine in ("auto", ""):
            return "qwen"
        return family + "+qwen"

    def family_for(self, job):
        """The family for this job. With `auto` it is Krea 2 when that is ready, and Qwen only when Krea 2 cannot generate:
        the tool works with what you have instead of prescribing a model."""
        eng = job.engine or "auto"
        if eng != "auto":
            return FAMILY[eng]
        if job.characters or job.refs:
            return "krea2"   # only the base for a styled two-step job; plain character jobs go straight to Qwen (see plan_mode)
        for fam in ("krea2", "qwen21"):
            try:
                self.pick(fam, job.model if fam == "krea2" else "")
                return fam
            except Missing:
                continue
        return "krea2"   # nothing is ready: the later pick() says what is missing

    # ---------- generating ----------

    def _render(self, graph):
        data, _ = self.comfy.run(graph, self.on_progress)
        return data[0]

    def _upload(self, data):
        data = to_png(data)
        return self.comfy.upload(data, f"image_{random.getrandbits(48):012x}.{sniff_ext(data)}")

    def _base_render(self, family, text, job, w, h, seed, q, files, status):
        loras = [(n, s) for n, s in job.loras if (self.lora_family(n) in (None, family))]
        if len(loras) != len(job.loras):
            status("note: some LoRAs belong to another model family and are left out")
        steps = STEPS[family][q]
        styles = job.styles[:3]
        if styles and not (files.get("style_lora") and all(self.comfy.node(n) for n in STYLE_NODES)):
            raise Missing(f"style images with Krea 2 need '{FILES['krea2']['style_lora']}' (ComfyUI/{FOLDER['lora']}) "
                          f"and the nodes {', '.join(STYLE_NODES)}")
        names = [self._upload(d) for d in styles]
        return self._render(graphs.build_krea(text, loras, names, w, h, seed=seed, steps=steps, unet=files["unet"],
                                              clip=files["clip"], vae=files["vae"], style_lora=files.get("style_lora") or ""))

    def _qwen(self, text, names, qfiles, steps, w=None, h=None, seed=None, resolution=1024):
        return self._render(graphs.build_qwen(text, names, w, h, seed=seed, steps=steps, resolution=resolution,
                                              unet=qfiles["unet"], clip=qfiles["clip"], vae=qfiles["vae"]))

    def _eyes_pass(self, data, qfiles, steps):
        """Qwen draws eyes unevenly now and then; one narrow edit evens them out."""
        text = ("Edit <image1>: perfect the eyes of the character. Both eyes the same size and shape, round matching irises and "
                "pupils, both eyes looking in exactly the same direction, natural eyelids and eyelashes, sharp catchlights. "
                "Change nothing else: same face, hair, body exactly as it is (nothing added, nothing removed), pose, lighting, "
                "background and art style.")
        return self._qwen(text, [self._upload(data)], qfiles, steps, seed=random.randrange(2**32), resolution=0)

    def describe(self, data, role):
        return self.ollama.describe(data, role) if self.ollama else ""

    def plan(self, job):
        """Decides what a job will do and which files it loads, without running anything. Raises Missing (with the catalog files
        that would fix it) when something is not there, so callers can fix that before any image is made."""
        family = self.family_for(job)
        job.characters, job.styles, job.refs = job.characters[:3], job.styles[:3], job.refs[:2]
        w, h = ASPECTS.get(job.aspect if job.aspect in ASPECTS else "1:1")
        scale = {"1": 1.0, "1.5": 1.5, "2": 2.0}.get(str(job.size or "1"), 1.0)
        w, h = (int(round(v * scale / 16)) * 16 for v in (w, h))
        q = job.quality if job.quality in ("fast", "normal", "high") else "normal"
        mode = self.plan_mode(job, family)
        if not job.prompt.strip():
            raise ValueError("the prompt is empty")
        qwen_path = mode in ("edit", "qwen") or mode.endswith("+qwen")
        base_render = mode not in ("edit", "qwen")
        problems, ids, qfiles, files = [], [], None, None
        for name, wanted in (("qwen", qwen_path), ("base", base_render)):
            if not wanted:
                continue
            try:
                if name == "qwen":
                    qfiles = self.pick("qwen21", job.qwen_model, job.clip if family == "qwen21" else "", job.vae if family == "qwen21" else "")
                else:
                    files = self.pick(family, job.model, job.clip, job.vae)
            except Missing as e:
                problems.append(str(e))
                ids.extend(e.ids)
        if files is not None:   # the optional files a feature needs
            if family == "krea2" and job.styles:
                if not files.get("style_lora"):
                    problems.append(f"style images with Krea 2 need '{FILES['krea2']['style_lora']}' (ComfyUI/{FOLDER['lora']})")
                    ids.append("krea2-style")
                elif not all(self.comfy.node(n) for n in STYLE_NODES):
                    problems.append(f"style images with Krea 2 need the ComfyUI nodes {', '.join(STYLE_NODES)} (update ComfyUI)")
        if problems:
            raise Missing("; ".join(problems), ids)
        return {"family": family, "mode": mode, "w": w, "h": h, "quality": q, "files": files, "qfiles": qfiles}

    def generate(self, job, status=print):
        """-> (png bytes, info). status(text) reports progress in words."""
        t0 = time.time()
        p = self.plan(job)   # every missing file is reported here, before anything is uploaded or rendered
        family, mode, w, h, q, files, qfiles = p["family"], p["mode"], p["w"], p["h"], p["quality"], p["files"], p["qfiles"]
        seed = job.seed if job.seed is not None else random.randrange(2**32)
        info = {"mode": mode, "family": family, "seed": seed, "width": w, "height": h, "quality": q}
        text = job.prompt.strip()
        info["files"] = {"base": files, "qwen": qfiles}
        if mode in ("edit", "qwen") and job.loras:
            status("note: LoRAs only work when Krea 2 draws the image; they are ignored in this mode")

        # Everything that needs Ollama happens first, then its model is unloaded, then ComfyUI renders: a vision model that
        # stays in VRAM next to Qwen makes it dozens of times slower.
        qwen_path = mode == "edit" or mode == "qwen" or mode.endswith("+qwen")
        natural, descs, style_desc = text, [], ""
        if qwen_path:
            if mode == "edit" and self.ollama:
                natural = self.ollama.edit_instruction(text)
            for c in (job.characters[:1] if mode == "edit" else job.characters):
                if self.ollama:
                    status("Looking at the character ...")
                descs.append(self.describe(c, "character"))
            if mode == "qwen" and job.styles:
                style_desc = self.describe(job.styles[0], "style")
        if self.ollama:
            self.ollama.unload()

        if mode == "edit":
            info["prompt_used"] = natural
            status("Qwen edits the image ...")
            names = [self._upload(job.edit)]
            instruction = f"Edit <image1>: {natural} Make this change clearly visible. Keep everything else unchanged."
            for c, desc in zip(job.characters[:1], descs):
                names.append(self._upload(c))
                look = char_text(desc)
                instruction += (f" The character must keep the exact identity shown in <image2>{' (' + look + ')' if look else ''}; "
                                f"<image2> is only an identity reference.")
            # random seed on purpose: editing with the seed that made the image only copies it
            data = self._qwen(instruction, names, qfiles, STEPS["qwen21"][q], seed=random.randrange(2**32), resolution=0)
            if job.characters and job.eyes == "on":
                status("Refining the eyes ...")
                data, info["eyes_pass"] = self._eyes_pass(data, qfiles, STEPS["qwen21"][q]), True
            return data, self._done(info, t0)

        if mode in (family,) and family != "qwen21":
            status(f"{TITLES[family]} is rendering ...")
            info["prompt_used"] = text
            return self._base_render(family, text, job, w, h, seed, q, files, status), self._done(info, t0)

        # qwen, or family+qwen
        style_img = job.styles[0] if job.styles else None
        if mode.endswith("+qwen"):
            status(f"{TITLES[family]} is rendering the style ...")
            style_img, style_desc = self._base_render(family, text, job, w, h, seed, q, files, status), ""

        status("Qwen puts character and style together ...")
        names, parts, n = [], [text], 0
        for c, desc in zip(job.characters, descs):
            n += 1
            names.append(self._upload(c))
            look = char_text(desc)
            parts.append(f"The character from <image{n}> must keep the exact same identity: face, facial features and "
                         f"markings, eyes, hair, ears, skin tone and body proportions"
                         f"{' (' + look + ')' if look else ''}. {ear_clause(desc, n)} <image{n}> only defines who the "
                         f"character is: do not copy its background, pose, camera angle or lighting; the scene is new, as described. "
                         f"Clothing follows the description above; only if it says nothing about clothing, keep the outfit of <image{n}>.")
        if style_img:
            n += 1
            names.append(self._upload(style_img))
            parts.append(f"Render the whole image in the visual style of <image{n}>{' (' + style_desc[:1200] + ')' if style_desc else ''}: "
                         f"its brushwork, rendering, shading, lighting, colour palette and line quality, and nothing else. Take nothing "
                         f"of the person in <image{n}>: not their face, hair, ears, body shape, proportions, bust size, age or clothing. "
                         + ("The character's figure and body proportions stay exactly as in <image1>." if job.characters else ""))
        for r in job.refs:
            n += 1
            names.append(self._upload(r))
            parts.append(f"Use <image{n}> as an additional visual reference.")
        if job.characters:
            who = " and ".join(f"<image{i}>" for i in range(1, len(job.characters) + 1))
            parts.append(f"The only people in the image are the character(s) from {who}, unless the description asks for others.")
        info["prompt_used"] = " ".join(parts)
        steps = STEPS["qwen21"][q]
        data = self._qwen(" ".join(parts), names[:4], qfiles, steps, w, h, seed=seed)

        if job.characters and style_img and job.fidelity == "high":
            # Strong styles (and faces inside style images) pull the face away from the character. A second edit pass
            # re-aligns face, hair and ears with the original while keeping the new style.
            status("Matching the face to the character ...")
            c, desc = job.characters[0], descs[0]
            look = f" ({style_desc[:1200]})" if style_desc else ""
            text2 = ("Edit <image1>: give the character the exact face, hair and ears of the person in <image2>: same face "
                     "shape, eyes, eyebrows, nose, mouth, hairstyle, hair length and ear shape. "
                     f"{ear_clause(desc, 2)} Restore the body shape, figure, proportions and bust size of <image2>. Keep the art style and "
                     f"rendering of <image1>{look}, and keep its framing, pose, lighting and background. <image2> is only an "
                     "identity reference: do not copy its background, pose, camera angle, lighting, realism or game-graphics look.")
            data = self._qwen(text2, [self._upload(data), self._upload(c)], qfiles, steps, seed=random.randrange(2**32), resolution=0)
            info["fidelity_pass"] = True
        if job.characters and style_img and char_facts(descs[0]) and not ears_pointed(descs[0]):
            # Style images often carry elf ears or similar features that leak into the character; a short, narrow edit
            # pass is the one thing that reliably removes them again (a long identity prompt does not).
            status("Matching the ears ...")
            text3 = ("Edit <image1>: replace the character's pointed elf ears, if any, with normal rounded human ears, the same "
                     "ears as the person in <image2>. Change nothing else: same face, hair, body shape and proportions exactly as they "
                     "are (nothing added, nothing removed), pose, lighting, background and art style.")
            data = self._qwen(text3, [self._upload(data), self._upload(job.characters[0])], qfiles, steps,
                              seed=random.randrange(2**32), resolution=0)
            info["ears_pass"] = True
        if job.characters and job.eyes == "on":
            status("Refining the eyes ...")
            data, info["eyes_pass"] = self._eyes_pass(data, qfiles, steps), True
        return data, self._done(info, t0)

    @staticmethod
    def _done(info, t0):
        info["seconds"] = round(time.time() - t0, 1)
        return info
