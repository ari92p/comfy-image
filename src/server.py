"""Image service: the same pipeline as `comfy-image`, behind a small HTTP API (standard library only).

Started by `comfy-image --serve`. There is no login: restrict who may reach it with a firewall rule. One job at a time.
The ComfyUI this tool manages is stopped after IMAGE_IDLE_SECONDS (default 300) without a request, which frees the graphics card, and
started again by the next request; a job that runs longer than IMAGE_HARD_SECONDS (default 1200) is cancelled and ComfyUI restarted.

    GET    /health      {state: idle|busy, text, progress, comfy, ollama}
    GET    /engines     which model families are ready in ComfyUI, with the files they use, or what is missing
    GET    /loras       the LoRA names ComfyUI lists (use these names in /generate)
    GET    /models      the Qwen files ComfyUI lists
    GET    /characters, /styles   the pictures saved with --save-character / --save-style (use them by name in /generate)
    POST   /generate    json -> image/png, details in the X-Image-Info header (json)
    POST   /cancel      stops the job that is running now (only this service's own job, never other ComfyUI work)

    /generate body (everything but "prompt" is optional; images are base64 strings, a data: prefix is fine):
    {"prompt": "...", "engine": "auto|krea|qwen", "model": "file", "qwen_model": "file", "clip": "file", "vae": "file",
     "loras": [["name", 0.8], ...] or [{"name": "...", "strength": 0.8}], "characters": [b64, ...], "styles": [b64, ...],
     "refs": [b64, ...], "character_names": ["me"], "style_names": ["ink"], "edit": b64,
     "aspect": "1:1", "size": "1", "quality": "normal", "seed": 123,
     "fidelity": "high", "eyes": "off", "two_step": false}
"""

import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import library
from pipeline import Engine, Job, Missing
from comfy import Cancelled, ComfyError

STATE = {"busy": False, "text": "", "step": 0, "steps": 0, "since": 0.0, "last": time.time()}
IDLE_SECONDS = int(os.environ.get("IMAGE_IDLE_SECONDS", "300"))
HARD_SECONDS = int(os.environ.get("IMAGE_HARD_SECONDS", "1200"))
LOCK = threading.Lock()
MAX_BODY = 200 * 1024 * 1024


def _b64(s):
    if "," in s[:64] and s.startswith("data:"):
        s = s.split(",", 1)[1]
    return base64.b64decode(s)


def job_from(body, eng):
    if not isinstance(body, dict) or not str(body.get("prompt", "")).strip():
        raise ValueError("\"prompt\" is required")
    loras = []
    for item in body.get("loras") or []:
        name, strength = (item["name"], item.get("strength", 1.0)) if isinstance(item, dict) else (item[0], item[1] if len(item) > 1 else 1.0)
        loras.append((eng.resolve_lora(name), float(strength)))
    chars = [_b64(x) for x in body.get("characters") or []]
    styles = [_b64(x) for x in body.get("styles") or []]
    for kind, key, target in (("character", "character_names", chars), ("style", "style_names", styles)):
        for n in body.get(key) or []:
            data = library.resolve(kind, str(n))
            if data is None:
                raise ValueError(f"no saved {kind} called '{n}' (saved: {', '.join(library.names(kind)) or 'none'})")
            target.append(data)
    return Job(
        prompt=str(body["prompt"]), engine=body.get("engine") or "auto", model=body.get("model") or "",
        qwen_model=body.get("qwen_model") or "", clip=body.get("clip") or "", vae=body.get("vae") or "", loras=loras,
        characters=chars, styles=styles,
        refs=[_b64(x) for x in body.get("refs") or []], edit=_b64(body["edit"]) if body.get("edit") else None,
        aspect=body.get("aspect") or "1:1", size=str(body.get("size") or "1"), quality=body.get("quality") or "normal",
        seed=int(body["seed"]) if body.get("seed") not in (None, "") else None,
        fidelity=body.get("fidelity") or "high", eyes=body.get("eyes") or "off", two_step=bool(body.get("two_step")))


def serve(eng, host, port, log=print, managed=None, idle=None, hard=None):
    idle = IDLE_SECONDS if idle is None else idle
    hard = HARD_SECONDS if hard is None else hard
    def status(text):
        STATE["text"] = text
        log(time.strftime("%H:%M:%S"), text)

    def progress(step, steps):
        STATE["step"], STATE["steps"] = step, steps
    eng.on_progress = progress

    def watch():
        """Frees the graphics card after a quiet time and ends a job that hangs. Only touches the ComfyUI this tool started."""
        while True:
            time.sleep(min(20, max(0.2, idle / 4)))
            now = time.time()
            if STATE["busy"] and hard and now - STATE["since"] > hard:
                log("a job runs longer than", hard, "s: cancelling it and restarting ComfyUI")
                eng.cancel()
                if managed:
                    managed.stop()
            elif not STATE["busy"] and managed and managed.proc and idle and now - STATE["last"] > idle and LOCK.acquire(blocking=False):
                try:
                    if not STATE["busy"] and now - STATE["last"] > idle:
                        log("no requests for", idle, "s: stopping ComfyUI to free the graphics card")
                        managed.stop()
                finally:
                    LOCK.release()
    threading.Thread(target=watch, daemon=True).start()

    def wake():
        if managed and not STATE["busy"]:
            managed.start(out=lambda *a: log(*a))

    class Handler(BaseHTTPRequestHandler):
        server_version = "image-service"

        def log_message(self, fmt, *args):
            log(time.strftime("%H:%M:%S"), self.address_string(), fmt % args)

        def reply(self, code, payload, ctype="application/json", headers=None):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            try:
                if self.path == "/health":
                    prog = round(STATE["step"] / STATE["steps"], 3) if STATE["steps"] else 0
                    self.reply(200, {"state": "busy" if STATE["busy"] else "idle", "text": STATE["text"], "progress": prog,
                                     "comfy": eng.comfy.alive(), "ollama": bool(eng.ollama and eng.ollama.up())})
                elif self.path == "/engines":
                    wake()
                    eng.comfy.refresh()
                    self.reply(200, eng.check())
                elif self.path == "/loras":
                    wake()
                    eng.comfy.refresh()
                    self.reply(200, eng.loras())
                elif self.path in ("/characters", "/styles"):
                    self.reply(200, library.names("character" if self.path == "/characters" else "style"))
                elif self.path == "/models":
                    wake()
                    eng.comfy.refresh()
                    self.reply(200, {"qwen": eng.qwen_files()})
                else:
                    self.reply(404, {"error": "not found"})
            except (OSError, ComfyError) as e:
                self.reply(502, {"error": f"ComfyUI is not reachable: {e}"})
            except SystemExit:
                self.reply(502, {"error": "ComfyUI could not be started (see ./comfyui/comfyui.log)"})

        def do_POST(self):
            if self.path == "/cancel":
                if STATE["busy"]:
                    eng.cancel()
                return self.reply(200, {"cancelled": STATE["busy"]})
            if self.path != "/generate":
                return self.reply(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return self.reply(400, {"error": "missing or oversized body"})
            raw = self.rfile.read(n)
            if not LOCK.acquire(blocking=False):
                return self.reply(503, {"error": "busy: one job at a time, try again shortly"})
            try:
                STATE.update(busy=True, text="starting", step=0, steps=0, since=time.time())
                eng.comfy.cancel.clear()
                if managed:
                    managed.start(out=lambda *a: log(*a))   # a no-op while it runs; starts it again after an idle stop or a crash
                eng.comfy.refresh()
                job = job_from(json.loads(raw), eng)
                data, info = eng.generate(job, status)
                self.reply(200, data, "image/png", {"X-Image-Info": json.dumps(info, ensure_ascii=True)})
            except (ValueError, KeyError, TypeError, Missing) as e:
                self.reply(400, {"error": str(e)})
            except Cancelled:
                self.reply(499, {"error": "cancelled"})
            except (OSError, ComfyError, TimeoutError) as e:
                self.reply(502, {"error": str(e)})
            except SystemExit:   # ComfyUI would not start again
                self.reply(502, {"error": "ComfyUI could not be started (see ./comfyui/comfyui.log)"})
            except Exception as e:   # never let one bad request end the service
                log("job failed:", repr(e))
                self.reply(500, {"error": repr(e)})
            finally:
                eng.comfy.cancel.clear()
                STATE.update(busy=False, text="", step=0, steps=0, last=time.time())
                LOCK.release()

    httpd = ThreadingHTTPServer((host, port), Handler)
    log(f"Image service on http://{host}:{port}  (Ctrl+C stops it)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
