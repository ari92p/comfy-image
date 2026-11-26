"""Everything that makes this tool self-contained: the pinned ComfyUI in ./comfyui, its private Python environment, the model files
(downloaded only after you say yes, verified by size and SHA-256), and starting/stopping that ComfyUI for you.

Nothing here runs unless asked for: `--setup`, a question in the guided mode, or the first image that needs it (which asks first)."""

import atexit
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

import catalog
from comfy import Comfy
import ui
from ui import IS_WIN, ask, die, human, say, shell_join, yes

ROOT = pathlib.Path(__file__).resolve().parent.parent   # the repository folder (this file lives in src/)
COMFY_DIR = pathlib.Path(os.environ.get("IMAGE_COMFY_DIR") or ROOT / "comfyui")
CACHE = pathlib.Path(os.environ.get("IMAGE_CACHE_DIR") or ROOT / ".cache")
OFFLINE = False   # --offline: never download
MANAGED_PORT = int(os.environ.get("IMAGE_COMFY_PORT", "8199"))   # not 8188, so it never collides with a ComfyUI you run yourself


# ---------- where the models live ----------

def models_dir(arg=None):
    """./comfyui/models by default; `--models-dir` (or IMAGE_MODELS_DIR) points at a folder you already have."""
    d = arg or os.environ.get("IMAGE_MODELS_DIR")
    return pathlib.Path(d).expanduser().resolve() if d else (COMFY_DIR / "models").resolve()


def present(f, mdir):
    """The file on disk, wherever it sits under the models folder (same name and exact size), or None."""
    name = pathlib.Path(f["dest"]).name
    if not mdir.exists():
        return None
    for p in mdir.rglob(name):
        try:
            if p.is_file() and p.stat().st_size == f["size"]:
                return p
        except OSError:
            continue
    return None


def missing_of(ids, mdir):
    return [i for i in dict.fromkeys(ids) if not present(catalog.FILES[i], mdir)]


# ---------- licenses ----------

def _accepted():
    try:
        return set(json.loads((CACHE / "licenses-accepted.json").read_text()))
    except (OSError, ValueError):
        return set()


def _remember(keys):
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "licenses-accepted.json").write_text(json.dumps(sorted(_accepted() | set(keys))))


# ---------- downloading ----------

def _url(f):
    return f"{os.environ.get('IMAGE_HF_ENDPOINT', catalog.HF).rstrip('/')}/{f['repo']}/resolve/{f['rev']}/{f['path']}"


def download(f, mdir, out=say):
    """Downloads one catalog file into the models folder. Resumes an interrupted download, checks size and SHA-256, and only then
    moves it to its final name, so a broken file never looks like a model."""
    dest = mdir / f["dest"]
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = f["size"]
    last_err = None
    for attempt in range(6):
        done, h = 0, hashlib.sha256()
        if part.exists():
            with open(part, "rb") as fh:   # re-hash what is already there, then continue after it
                while chunk := fh.read(8 << 20):
                    h.update(chunk)
                    done += len(chunk)
            if done > size:
                part.unlink()
                done, h = 0, hashlib.sha256()
        headers = {"User-Agent": "easy-image"}
        if os.environ.get("HF_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
        if done:
            headers["Range"] = f"bytes={done}-"
        try:
            if done == size:
                resp = None
            else:
                resp = urllib.request.urlopen(urllib.request.Request(_url(f), headers=headers), timeout=60)
                if done and resp.status != 206:   # the server ignored the range: start over
                    part.unlink()
                    done, h = 0, hashlib.sha256()
            if resp is not None:
                with resp, open(part, "ab") as fh:
                    t0, shown, got0 = time.time(), 0.0, done
                    while chunk := resp.read(1 << 20):
                        fh.write(chunk)
                        h.update(chunk)
                        done += len(chunk)
                        if time.time() - shown > 1.0:
                            shown = time.time()
                            rate = (done - got0) / max(shown - t0, 0.001)
                            print(f"\r   {human(done)} of {human(size)}  ({done * 100 // size}%)  {human(rate)}/s   ", end="", flush=True)
                    print(f"\r   {human(done)} of {human(size)}  (100%)                    ", flush=True)
            if done != size:
                raise OSError(f"got {done} of {size} bytes")
            if h.hexdigest() != f["sha256"]:
                part.unlink()
                raise ValueError("checksum does not match; the file was removed")
            os.replace(part, dest)
            return dest
        except ValueError:
            raise
        except (OSError, urllib.error.URLError) as e:
            last_err = e
            out(f"\n   connection problem ({e}); trying again from where it stopped ...")
            time.sleep(2 + attempt * 3)
    raise OSError(f"download of {pathlib.Path(f['dest']).name} failed: {last_err}")


def show_plan(todo, mdir):
    """Prints the files that would be downloaded, their size and their licenses."""
    total = sum(catalog.FILES[i]["size"] for i in todo)
    say("\nThese model files are missing and can be downloaded (from the authors' Hugging Face pages, checked by SHA-256):")
    for i in todo:
        f = catalog.FILES[i]
        say(f"   {pathlib.Path(f['dest']).name:48} {human(f['size']):>9}   {catalog.LICENSES[f['license']][0]}")
    say(f"   total {human(total)} into {mdir}")
    say("\nLicenses of these files (read them, they differ):")
    for k in sorted({catalog.FILES[i]["license"] for i in todo}):
        title, terms, link = catalog.LICENSES[k]
        say(f"   {title}: {terms}\n   {link}")


def ensure_files(ids, mdir, assume_yes=False, accept_licenses=False, interactive=True, show=True):
    """Makes sure the catalog files exist in the models folder. Tells you what, how big and under which license; asks before
    downloading. Returns the list of files fetched."""
    todo = missing_of(ids, mdir)
    if not todo:
        return []
    if OFFLINE:
        die("model files are missing and --offline is set, so nothing is downloaded: " + ", ".join(pathlib.Path(catalog.FILES[i]["dest"]).name for i in todo))
    total = sum(catalog.FILES[i]["size"] for i in todo)
    if show:
        show_plan(todo, mdir)
    probe = mdir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < total * 1.05:
        die(f"not enough disk space: {human(total)} needed, {human(free)} free on {probe}")
    lic = sorted({catalog.FILES[i]["license"] for i in todo})
    if not interactive or not sys.stdin.isatty():
        if not assume_yes:
            die("model files are missing. Run with -y to download them (and --accept-licenses to accept the licenses above).")
    elif not assume_yes and not yes("\nDownload them now?"):
        die("cancelled. Without these files this image cannot be made.")
    if not set(lic) <= _accepted():
        if accept_licenses:
            pass
        elif interactive and sys.stdin.isatty():
            if not yes("Do you accept these licenses?", "n"):
                die("cancelled: the licenses were not accepted.")
        else:
            die("add --accept-licenses to confirm you accept the licenses above.")
        _remember(lic)
    got = []
    for n, i in enumerate(todo, 1):
        f = catalog.FILES[i]
        say(f"\n[{n}/{len(todo)}] {pathlib.Path(f['dest']).name}")
        try:
            download(f, mdir)
        except (OSError, ValueError) as e:
            die(str(e))
        got.append(i)
    say("\nAll model files are in place.")
    return got


# ---------- installing ComfyUI ----------

def _run(cmd, dry=False, **kw):
    """Runs a setup command without its chatter (everything with -v); when it fails, shows what it printed."""
    if dry or ui.VERBOSE:
        say("  $ " + shell_join(cmd))
    if dry:
        return
    if ui.VERBOSE:
        r = subprocess.run([str(c) for c in cmd], **kw)
        if r.returncode:
            die(f"command failed: {shell_join(cmd)}")
        return
    r = subprocess.run([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                       errors="replace", **kw)
    if r.returncode:
        die(f"setup failed ({shell_join(cmd)}):\n" + "\n".join(r.stdout.strip().splitlines()[-25:]))


def find_uv():
    """`uv` from the PATH, or the one src/setup.sh / setup.cmd put into .venvs/uv."""
    found = shutil.which("uv")
    if found:
        return found
    own = ROOT / ".venvs" / "uv" / "bin" / ("uv.exe" if IS_WIN else "uv")
    return str(own) if own.exists() else None


def archive_url(repo, commit):
    """The zip of one exact commit (no git needed): https://github.com/<owner>/<name>/archive/<commit>.zip"""
    return repo.removesuffix(".git") + f"/archive/{commit}.zip"


def fetch_source(repo, commit, dest, dry=False):
    """Downloads the pinned commit as a zip and unpacks it into `dest` (the top folder of the zip is dropped)."""
    url = archive_url(repo, commit)
    if dry or ui.VERBOSE:
        say(f"  download {url}  ->  {dest}")
    if dry:
        return
    import tempfile
    import zipfile
    dest = pathlib.Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        zpath = pathlib.Path(tmp) / "src.zip"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "easy-image"}), timeout=60) as r, open(zpath, "wb") as fh:
                shutil.copyfileobj(r, fh)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(pathlib.Path(tmp) / "x")
        except (OSError, urllib.error.URLError, zipfile.BadZipFile) as e:
            die(f"could not download {url}: {e}")
        tops = list((pathlib.Path(tmp) / "x").iterdir())
        if len(tops) != 1 or not tops[0].is_dir():
            die(f"unexpected content in {url}")
        # merge into an existing folder (it may already hold downloaded models); never delete anything there
        shutil.copytree(tops[0], dest, dirs_exist_ok=True)


def nvidia_gpu():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        return r.stdout.strip().splitlines()[0]
    except Exception:
        return None


def comfy_python():
    return COMFY_DIR / ".venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")


def installed():
    return (COMFY_DIR / "main.py").exists() and comfy_python().exists() and (COMFY_DIR / ".image-ready").exists()


def describe_install():
    c, g = catalog.COMFYUI, catalog.GGUF_NODE
    return [f"ComfyUI {c['ref']} (commit {c['commit'][:8]}) from {c['repo']}",
            f"the ComfyUI-GGUF node (commit {g['commit'][:8]}) from {g['repo']}",
            f"a private Python {catalog.PYTHON} environment in {COMFY_DIR / '.venv'} with {', '.join(catalog.TORCH)} and ComfyUI's requirements (several GB)"]


def install_comfyui(assume_yes=False, dry=False, interactive=True, show=True):
    """Downloads the pinned ComfyUI and the GGUF node into ./comfyui and builds a private environment. Downloads libraries, no models."""
    if show:
        say("\nSetting up the ComfyUI this tool uses (the exact version the workflows were built with), nothing global:")
        for line in describe_install():
            say("  - " + line)
        say("  Model files are not part of this step.")
    uv = find_uv()
    if not dry:
        if not uv:
            die("`uv` is needed once to build the Python environment (it also fetches the right Python version).\n"
                "  Run src/setup.sh (Windows: src\\setup.cmd) once: it fetches uv into this folder.\n"
                "Then run this command again.")
        if not assume_yes:
            if not (interactive and sys.stdin.isatty()):
                die("ComfyUI is not set up yet. Run again with -y to set it up (several GB of libraries), or use --comfy URL for a ComfyUI you already have.")
            if not yes("Go ahead?"):
                die("cancelled")
    uv = uv or "uv"
    gpu = nvidia_gpu()
    if dry or ui.VERBOSE:
        say(f"  Graphics card: {gpu or 'no NVIDIA card found (Apple GPU or CPU: works, but slowly)'}")
    else:
        say("Setting up (first time only, a few minutes) ...")
    c, g = catalog.COMFYUI, catalog.GGUF_NODE
    if not (COMFY_DIR / "main.py").exists():
        fetch_source(c["repo"], c["commit"], COMFY_DIR, dry)
    node = COMFY_DIR / "custom_nodes" / "ComfyUI-GGUF"
    if not (node / "requirements.txt").exists():
        fetch_source(g["repo"], g["commit"], node, dry)
    if not (COMFY_DIR / ".venv").exists():
        _run([uv, "venv", "--python", catalog.PYTHON, COMFY_DIR / ".venv"], dry)
    py = comfy_python()
    torch_cmd = [uv, "pip", "install", "--python", py, *catalog.TORCH]
    if gpu and sys.platform != "darwin":
        torch_cmd += ["--index-url", catalog.TORCH_CUDA_INDEX]
    _run(torch_cmd, dry)
    _run([uv, "pip", "install", "--python", py, "-r", COMFY_DIR / "requirements.txt"], dry)
    _run([uv, "pip", "install", "--python", py, "-r", node / "requirements.txt"], dry)
    if dry:
        return
    (COMFY_DIR / ".image-ready").write_text("ok\n")
    say("\nComfyUI is set up in ./comfyui.")


# ---------- running it ----------

def _extra_paths(mdir):
    """A config that makes ComfyUI read models from your own folder (same layout as ComfyUI's models folder)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "extra_model_paths.yaml"
    base = str(mdir).replace("\\", "/")
    p.write_text("easy_image:\n    base_path: " + base + "\n    diffusion_models: diffusion_models\n    text_encoders: text_encoders\n"
                 "    vae: vae\n    checkpoints: checkpoints\n    loras: loras\n", encoding="utf-8")
    return p


class Managed:
    """The ComfyUI in ./comfyui, started on demand and stopped again when the tool ends (unless it was already running)."""

    def __init__(self, mdir, port=MANAGED_PORT, keep=False):
        self.mdir, self.port, self.keep = mdir, port, keep
        self.url = f"http://127.0.0.1:{port}"
        self.proc = None
        self.comfy = Comfy(self.url)

    def command(self):
        cmd = [str(comfy_python()), "main.py", "--listen", "127.0.0.1", "--port", str(self.port), *catalog.LAUNCH_FLAGS,
               "--output-directory", str(CACHE / "comfy-output")]
        if self.mdir.resolve() != (COMFY_DIR / "models").resolve():
            cmd += ["--extra-model-paths-config", str(_extra_paths(self.mdir))]
        return cmd

    def start(self, out=say):
        if self.comfy.alive(2):
            out(f"ComfyUI is already running on port {self.port}; using it.")
            return self.comfy
        CACHE.mkdir(parents=True, exist_ok=True)
        out("Starting ComfyUI ...")
        log = open(COMFY_DIR / "comfyui.log", "ab")
        self.proc = subprocess.Popen(self.command(), cwd=COMFY_DIR, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))
        if not self.keep:
            atexit.register(self.stop)
        for _ in range(180):
            if self.comfy.alive(2):
                self.comfy.refresh()
                return self.comfy
            if self.proc.poll() is not None:
                tail = (COMFY_DIR / "comfyui.log").read_text(errors="replace")[-1500:]
                die(f"ComfyUI stopped right after starting (exit {self.proc.returncode}). The end of ./comfyui/comfyui.log:\n{tail}")
            time.sleep(2)
        self.stop()
        die("ComfyUI did not answer within 6 minutes. See ./comfyui/comfyui.log")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
