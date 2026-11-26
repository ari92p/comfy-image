"""A fake ComfyUI for the offline tests: it answers the endpoints comfy-image uses, checks every graph it receives and
returns a tiny PNG. It is not ComfyUI: it cannot tell whether an image would look right, only whether the request is well formed.

If you point IMAGE_TEST_OBJECT_INFO at a real `GET /object_info` dump, every graph is also checked against that node
definition (class names, required inputs, allowed values); without it only the links between nodes are checked."""

import json
import os
import threading
import zlib
import struct
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def tiny_png():
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = b"\x00\xff\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


# what the fake "has installed": names use backslashes like ComfyUI on Windows; none of these are real people's or private files
INSTALLED = {
    "UNETLoader": ("unet_name", ["krea2_turbo_fp8_scaled.safetensors", "qwen-image-2.1-UC-int8_convrot.safetensors",
                                 "sample_krea2_finetune.safetensors"]),
    "UnetLoaderGGUF": ("unet_name", ["qwen-image-2.1-UC-Q4_K_M.gguf", "qwen-image-2.1-Q6_K.gguf", "qwen-image-2.1-UC-Q6_K.gguf"]),
    "CLIPLoader": ("clip_name", ["qwen3vl_8b_int8_convrot.safetensors", "qwen3vl_4b_fp8_scaled.safetensors", "qwen3vl_4b_bf16.safetensors"]),
    "VAELoader": ("vae_name", ["qwen_image_2.1_vae_bf16.safetensors", "qwen_image_vae.safetensors"]),
    "LoraLoaderModelOnly": ("lora_name", ["krea2\\krea2_softwatercolor.safetensors", "krea2\\krea2_darkbrush.safetensors",
                                          "krea2\\krea2_style_reference.safetensors"]),
}
LOADER_FIELDS = {(n, f) for n, (f, _) in INSTALLED.items()}

# nodes the fake knows when no real dump is given (name -> required input names that are not links we create ourselves)
MINI_NODES = ["UNETLoader", "UnetLoaderGGUF", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly",
              "QwenImage21Cache", "TextEncodeQwenImage21", "EmptyFlux2LatentImage", "EmptyLatentImage", "CLIPTextEncode",
              "ConditioningZeroOut", "ModelSamplingFlux", "TextEncodeQwenImageEditPlus", "FluxKontextMultiReferenceLatentMethod",
              "LoadImage", "KSampler", "VAEDecode", "SaveImage"]


class FakeComfy:
    def __init__(self, missing_nodes=(), drop_files=(), port=0):
        self.graphs, self.uploads, self.errors, self.deleted = [], [], [], []
        self.guard = None   # callable -> list of problems, checked on every /prompt 
        self.missing_nodes, self.drop_files = set(missing_nodes), set(drop_files)
        path = os.environ.get("IMAGE_TEST_OBJECT_INFO", "")
        self.info = json.load(open(path)) if path and os.path.exists(path) else None
        fake = self
        self.slow = 0.0

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                p = self.path
                if p == "/system_stats":
                    return self.send(200, {"system": {}})
                if p.startswith("/object_info/"):
                    return self.send(200, fake.node_info(p.rsplit("/", 1)[1]))
                if p.startswith("/history/"):
                    pid = p.rsplit("/", 1)[1]
                    return self.send(200, {pid: {"status": {"completed": True, "status_str": "success", "messages": []},
                                                 "outputs": {"save": {"images": [{"filename": "x.png", "subfolder": "", "type": "output"}]}}}})
                if p.startswith("/view"):
                    return self.send(200, tiny_png(), "image/png")
                self.send(404, {})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n)
                if self.path == "/upload/image":
                    name = f"up{len(fake.uploads)}.png"
                    fake.uploads.append((name, len(body)))
                    return self.send(200, {"name": name})
                if self.path == "/prompt":
                    graph = json.loads(body)["prompt"]
                    errs = fake.validate(graph) + (fake.guard() if fake.guard else [])
                    if errs:
                        fake.errors.append(errs)
                        return self.send(400, {"error": "invalid", "node_errors": errs})
                    fake.graphs.append(graph)
                    return self.send(200, {"prompt_id": f"p{len(fake.graphs)}"})
                if self.path in ("/queue", "/interrupt"):
                    fake.deleted.append((self.path, json.loads(body)))
                    return self.send(200, {})
                self.send(404, {})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def files(self, node):
        field, names = INSTALLED[node]
        return field, [n for n in names if n not in self.drop_files]

    def node_info(self, name):
        if name in self.missing_nodes:
            return {}
        if self.info is not None:
            if name not in self.info:
                return {}
            d = json.loads(json.dumps(self.info[name]))
        elif name in MINI_NODES:
            d = {"input": {"required": {}}}
        else:
            return {}
        if name in INSTALLED:
            field, names = self.files(name)
            d.setdefault("input", {}).setdefault("required", {})[field] = [names]
        return {name: d}

    def validate(self, graph):
        errs = []
        for nid, node in graph.items():
            ct, inputs = node["class_type"], node["inputs"]
            if ct in self.missing_nodes:
                errs.append(f"{nid}: node {ct} is not installed")
                continue
            if self.info is not None and ct not in self.info:
                errs.append(f"{nid}: unknown node {ct}")
                continue
            for k, v in inputs.items():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    if v[0] not in graph:
                        errs.append(f"{nid}.{k}: link to missing node {v[0]}")
            if ct in INSTALLED:
                field, names = self.files(ct)
                if inputs.get(field) not in names:
                    errs.append(f"{nid}.{field}: '{inputs.get(field)}' is not one of the installed files")
            if self.info is None:
                continue
            spec = self.info[ct]["input"]
            required, optional = spec.get("required", {}), spec.get("optional", {})
            for k, t in required.items():
                if t[0] == "COMFY_AUTOGROW_V3":
                    continue
                if k not in inputs:
                    errs.append(f"{nid}: {ct} is missing required input '{k}'")
            for k, v in inputs.items():
                if k.startswith("images.image_"):
                    continue
                t = required.get(k) or optional.get(k)
                if t is None:
                    errs.append(f"{nid}: {ct} has no input '{k}'")
                elif ct == "LoadImage" and k == "image":
                    if v not in [u[0] for u in self.uploads]:
                        errs.append(f"{nid}.image: '{v}' was never uploaded")
                elif isinstance(t[0], list) and (ct, k) not in LOADER_FIELDS and not isinstance(v, list) and v not in t[0]:
                    errs.append(f"{nid}.{k}: '{v}' is not an allowed value")
        return errs
