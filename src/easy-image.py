#!/usr/bin/env python3
"""easy-image: AI images on your own computer: new images, edits, characters, style references and LoRAs.

  easy-image                                   sets everything up the first time (one question), then guided mode
  easy-image "a fox in the snow"               one command, no questions
  easy-image --edit photo.png "make the jacket red"
  easy-image --character me.png "me as an astronaut on Mars"

`easy-image` is ./easy-image (Windows: easy-image.cmd) in the repository folder. Run with --help for all options.
This file needs nothing but a plain Python 3.9+ (standard library only); src/setup.sh makes one for you.
"""
import argparse
import datetime
import json
import os
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent   # the repository folder (this file lives in src/)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import catalog   # noqa: E402
import library   # noqa: E402
import setup_env   # noqa: E402
import ui   # noqa: E402
from comfy import Cancelled, Comfy, ComfyError   # noqa: E402
from ollama import Ollama   # noqa: E402
from pipeline import ASPECTS, FILES, TITLES, Engine, Job, Missing   # noqa: E402
from ui import ask, clean_path, die, human, launcher, rel, say, shell_join, yes   # noqa: E402

OUT = pathlib.Path(os.environ.get("IMAGE_OUT_DIR") or ROOT / "out")
CONFIG = pathlib.Path(os.environ.get("IMAGE_CONFIG") or ROOT / "config.json")   # only exists if you ran --configure
CACHE = setup_env.CACHE
DEFAULT_OLLAMA = os.environ.get("IMAGE_OLLAMA_URL", "http://127.0.0.1:11434")
ENGINES = ["auto", "krea", "qwen"]


def read_image(path):
    p = clean_path(path)
    if not os.path.isfile(p):
        die(f"image not found: {p}")
    return pathlib.Path(p).read_bytes()


def load_refs(kind, items):
    """Character / style pictures: a file you name, or one you saved earlier under a name."""
    out = []
    for t in items or []:
        data = library.resolve(kind, clean_path(t)) or library.resolve(kind, t)
        if data is None:
            have = library.names(kind)
            die(f"{kind} not found: '{t}'" + (f". Saved {kind}s: {', '.join(have)}" if have else f". Give a file, or save one with --save-{kind} NAME"))
        out.append(data)
    return out


def load_config():
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(cfg):
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def apply_config(a):
    """Your saved choices fill in what the command line leaves out (command line first, then the environment, then config.json)."""
    cfg = load_config()
    if not a.comfy and not os.environ.get("IMAGE_COMFY_URL") and cfg.get("comfy"):
        a.comfy = cfg["comfy"]
    if not a.models_dir and cfg.get("models_dir"):
        a.models_dir = cfg["models_dir"]
    return cfg


# ---------- which ComfyUI ----------

class Session:
    """The ComfyUI this run uses: one you point to with --comfy, or (default) the one this tool sets up and starts in ./comfyui."""

    def __init__(self, a):
        self.a = a
        url = a.comfy or os.environ.get("IMAGE_COMFY_URL")
        self.external = url
        self.mdir = setup_env.models_dir(a.models_dir)
        self.managed = None if url else setup_env.Managed(self.mdir, keep=a.keep_running)

    def open(self, interactive=True):
        if self.external:
            comfy = Comfy(self.external)
            if not comfy.alive():
                die(f"ComfyUI is not reachable at {self.external}.\n"
                    f"  Start it, or leave out --comfy to let this tool set up and run its own ComfyUI.")
            return comfy
        if not setup_env.installed():
            setup_env.install_comfyui(assume_yes=self.a.yes, interactive=interactive)
        return self.managed.start()

    def close(self):
        if self.managed and not self.a.keep_running:
            self.managed.stop()

    def prepare(self, eng, job):
        """Plans the job; when files are missing and this tool manages ComfyUI, offers to download exactly those, then plans again."""
        for attempt in (1, 2):
            try:
                return eng.plan(job)
            except Missing as e:
                if attempt == 1 and e.ids and not self.external:
                    setup_env.ensure_files(e.ids, self.mdir, self.a.yes, self.a.accept_licenses)
                    eng.comfy.refresh()
                    continue
                hint = ("\n  (This ComfyUI is not managed by this tool; put the files there, or leave out --comfy.)" if self.external else "")
                die(str(e) + hint)


def make_ollama(a, quiet=False):
    if a.no_ollama:
        return None
    o = Ollama(a.ollama, a.chat_model, CACHE / "describe")
    if o.up():
        return o
    if not quiet and ui.VERBOSE:
        say(f"(Ollama not found at {a.ollama}: continuing without character/style descriptions and without translating edit wishes)")
    return None


def family_download(fam, mdir):
    """Bytes still to download for a family's default files (0 when everything is there)."""
    ids = catalog.FAMILIES.get(fam, [])
    return sum(catalog.FILES[i]["size"] for i in setup_env.missing_of(ids, mdir))


def print_check(eng, mdir=None):
    res = eng.check()
    for fam, r in res.items():
        todo = family_download(fam, mdir) if mdir else 0
        say(f"  {'ready    ' if r['ok'] else 'NOT ready'}  {TITLES[fam]}" + (f"   (download: {human(todo)})" if todo and not r["ok"] else ""))
        if r["ok"]:
            shown = ", ".join(f"{k}={v}" for k, v in r["files"].items() if v)
            say(f"             {shown}")
            alts = eng.candidates(fam, "unet")
            if len(alts) > 1:
                say(f"             {len(alts)} model files found: {', '.join(alts[:6])}{' ...' if len(alts) > 6 else ''}")
        else:
            say(f"             {r['problems']}")
    return res


# ---------- headless ----------

def parse_lora(text):
    m = re.match(r"^(.*):(-?\d+(?:\.\d+)?)$", text)
    return (m.group(1), float(m.group(2))) if m else (text, 1.0)


def get_prompt(a):
    given = [x for x in (a.prompt, " ".join(a.words) if a.words else None, a.prompt_file, a.stdin_text) if x]
    if len(given) > 1:
        die("give the prompt once: as an argument, with --prompt, with --prompt-file, or on stdin")
    if a.prompt is not None:
        prompt = a.prompt
    elif a.words:
        prompt = " ".join(a.words)
    elif a.prompt_file:
        prompt = pathlib.Path(clean_path(a.prompt_file)).read_text(encoding="utf-8")
    elif a.stdin_text is not None:
        prompt = a.stdin_text
    else:
        die(f"give a prompt, e.g.  {launcher()} \"a fox in the snow\"")
    if not prompt.strip():
        die("the prompt is empty")
    return prompt.strip()


def save_refs(a):
    """--save-character NAME / --save-style NAME: keep the single picture you gave under that name."""
    for kind, name, items in (("character", a.save_character, a.character), ("style", a.save_style, a.style)):
        if not name:
            continue
        if len(items or []) != 1 or not os.path.isfile(clean_path(items[0])):
            die(f"--save-{kind} needs exactly one --{kind} FILE")
        p = pathlib.Path(clean_path(items[0]))
        try:
            saved = library.save(kind, name, p.read_bytes(), p.suffix.lower())
        except ValueError as e:
            die(str(e))
        say(f"Saved {kind} '{saved.stem}'. Use it any time with --{kind} {saved.stem}")


def build_job(a, eng):
    prompt = get_prompt(a)
    loras = []
    for item in a.lora or []:
        name, strength = parse_lora(item)
        try:
            loras.append((eng.resolve_lora(name), strength))
        except Missing as e:
            die(str(e))
    save_refs(a)
    job = Job(prompt=prompt, engine=a.engine, model=a.model or "", qwen_model=a.qwen_model or "", clip=a.clip or "", vae=a.vae or "", loras=loras,
              characters=load_refs("character", a.character), styles=load_refs("style", a.style),
              refs=[read_image(p) for p in a.ref or []], edit=read_image(a.edit) if a.edit else None,
              aspect=a.aspect, size=a.size, quality=a.quality, seed=a.seed, fidelity=a.fidelity, eyes=a.eyes,
              two_step=a.two_step)
    if job.edit and (job.styles or job.refs or job.loras):
        say("note: styles, extra references and LoRAs are not used when editing an image")
    if len(job.characters) > 3 or len(job.styles) > 3 or len(job.refs) > 2:
        say("note: at most 3 characters, 3 style images and 2 extra references are used")
    return job


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "image"


def save(data, info, job, out, n, count):
    if out:
        out = pathlib.Path(clean_path(str(out)))
        if count > 1:
            out = out.with_name(f"{out.stem}-{n + 1}{out.suffix or '.png'}")
        if not out.suffix:
            out = out.with_suffix(".png")
    else:
        out = OUT / f"{slug(job.prompt)}-{datetime.datetime.now():%Y%m%d-%H%M%S}{f'-{n + 1}' if count > 1 else ''}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    meta = {"prompt": job.prompt, "info": info, "loras": job.loras, "engine": job.engine}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def run_job(eng, job, count, out):
    """Generates `count` images (a fixed seed counts up per image), saves them, returns the paths."""
    base_seed, paths = job.seed, []
    if sys.stdout.isatty():
        eng.on_progress = lambda s, m: print(f"\r   step {s}/{m}   ", end="", flush=True) if m else None
    for n in range(count):
        job.seed = None if base_seed is None else base_seed + n
        prefix = f"[{n + 1}/{count}] " if count > 1 else ""
        t0 = time.time()
        try:
            data, info = eng.generate(job, lambda t: say(f"\n{prefix}{t}" if sys.stdout.isatty() else f"{prefix}{t}"))
        except Missing as e:
            die(str(e))
        except Cancelled:
            die("cancelled")
        except (ComfyError, TimeoutError, OSError) as e:
            die(str(e))
        except KeyboardInterrupt:
            die("stopped (the running ComfyUI job was interrupted; other jobs in its queue are untouched)")
        p = save(data, info, job, out, n, count)
        say(f"\nSaved: {p}   ({info['mode']}, seed {info['seed']}, {time.time() - t0:.0f} s)")
        paths.append(p)
    return paths


def headless(a):
    sess = Session(a)
    try:
        comfy = sess.open(interactive=False)
        eng = Engine(comfy, make_ollama(a))
        job = build_job(a, eng)
        sess.prepare(eng, job)
        run_job(eng, job, a.count, a.out)
    finally:
        sess.close()


# ---------- first-start setup ----------

# What `--setup` and the first start set up when you name nothing: both models and the Krea 2 style reference (35 GB).
DEFAULT_FAMILIES = ["krea2", "qwen21"]
DEFAULT_IDS = [i for f in DEFAULT_FAMILIES for i in catalog.FAMILIES[f]] + ["krea2-style"]

CHOICES = [("krea2", "Krea 2", "generates new images; LoRAs and style images"),
           ("qwen21", "Qwen-Image 2.1", "edits images; characters and style references")]


def configure(a, first):
    """Asks where ComfyUI comes from, where the models live and which models you want; remembers it in config.json; sets it up."""
    cfg = load_config()
    say("\nInitial configuration" if first else "\nConfiguration")
    s = ask("ComfyUI: press Enter and this tool sets up and runs its own (recommended), or type the address of one you already run "
            "(e.g. http://HOST:8188):", cfg.get("comfy", ""))
    comfy = "" if not s else (s if "://" in s else "http://" + s)
    mdir, families = "", []
    if not comfy:
        default = str(setup_env.models_dir(None))
        m = ask(f"Folder for the model files. Enter = {default}, or the path of a models folder you already have (it is used as it is):",
                cfg.get("models_dir", ""))
        mdir = clean_path(m) if m else ""
        md = setup_env.models_dir(mdir or None)
        say("\nWhich models do you want? Nothing is downloaded before you confirm; you can add more later.")
        for i, (fam, title, note) in enumerate(CHOICES, 1):
            todo = family_download(fam, md)
            say(f"   {i}. {title}: {note}  ({'already in that folder' if not todo else human(todo) + ' to download'})")
        say("   (Other models and LoRAs: just put the files into the folder; you can name them with --model.)")
        pick_ = ask("Numbers separated by spaces, `all`, or `none` (default: 1 2):", "1 2").lower()
        nums = range(1, len(CHOICES) + 1) if pick_ == "all" else [int(t) for t in pick_.split() if t.isdigit() and 1 <= int(t) <= len(CHOICES)]
        families = [CHOICES[i - 1][0] for i in nums]
    cfg.update(comfy=comfy, models_dir=mdir, families=families)
    save_config(cfg)
    a.comfy, a.models_dir = comfy or None, mdir or None
    if not comfy:
        if not setup_env.installed():
            setup_env.install_comfyui(assume_yes=a.yes)
        ids = [i for f in families for i in catalog.FAMILIES[f]]
        if ids:
            setup_env.ensure_files(ids, setup_env.models_dir(mdir or None), a.yes, a.accept_licenses)
    say(f"\nSaved. Now try:  {launcher()} \"a fox in the snow\"")
    return cfg


# ---------- guided mode ----------

def pick(title, options, default=1):
    """options: [(label, note)]. Prints a numbered list, returns the 0-based index."""
    say(title)
    for i, (label, note) in enumerate(options, 1):
        say(f"   {i}. {label}" + (f"\n      {note}" if note else ""))
    s = ask("   Number", str(default))
    return int(s) - 1 if s.isdigit() and 1 <= int(s) <= len(options) else default - 1


def ask_images(title, limit, optional=True, kind=None):
    """Asks for pictures until the user presses Enter: a file (drag it into the window) or, for characters and styles, a saved name."""
    say(title)
    saved = library.names(kind) if kind else []
    say("   Drag the file into this window or type its path." + (f" Saved: {', '.join(saved)} (type a name)." if saved else "")
        + " Press Enter when you have no (more) images.")
    items = []
    while len(items) < limit:
        s = ask(f"   Image {len(items) + 1}:")
        if not s:
            if items or optional:
                break
            continue
        p = clean_path(s)
        if os.path.isfile(p):
            items.append(p)
        elif kind and library.find(kind, s):
            items.append(library.clean_name(s))
        else:
            say("   I can't find that file or saved name, try again.")
    return items


def ask_loras(eng, family):
    """Lists the LoRAs ComfyUI has (those in a folder named after the family first) and lets the user pick by number."""
    names = eng.loras()
    if not names:
        return []
    internal = {FILES["krea2"]["style_lora"]}   # used automatically, not a style to pick
    names = [n for n in names if n.replace("\\", "/").rsplit("/", 1)[-1] not in internal]
    mine = [n for n in names if eng.lora_family(n) == family] or [n for n in names if eng.lora_family(n) in (None, family)]
    if not mine:
        return []
    say(f"\nLoRAs for {TITLES[family]} (a LoRA only works with the model it was trained for):")
    for i, n in enumerate(mine, 1):
        say(f"   {i}. {n}")
    say("   Type numbers with an optional strength, separated by spaces, e.g. `3` or `3:0.8 7:0.5`. Enter = none.")
    out = []
    for tok in ask("   LoRAs:").split():
        num, _, strength = tok.partition(":")
        if num.isdigit() and 1 <= int(num) <= len(mine):
            try:
                out.append((mine[int(num) - 1], float(strength) if strength else 1.0))
            except ValueError:
                say(f"   skipped '{tok}'")
    return out


def guided(a, ap):
    say("\nI'll ask a few questions and make your image. At the end I print the one-line command that does the same without questions.")
    sess = Session(a)
    try:
        comfy = sess.open(interactive=True)
        _guided_rest(a, ap, sess, comfy)
    finally:
        sess.close()


def _guided_rest(a, ap, sess, comfy):
    ollama = make_ollama(a, quiet=True)
    eng = Engine(comfy, ollama)
    managed = not sess.external

    # 2) what is here
    if ui.VERBOSE:
        say("\nOllama helpers: " + ("on" if ollama else "off"))
        say("Model families and what ComfyUI has for them:")
        res = print_check(eng, sess.mdir if managed else None)
    else:
        res = eng.check()
        ready = [TITLES[f] for f, r in res.items() if r["ok"]]
        later = [f"{TITLES[f]} ({human(family_download(f, sess.mdir))})" for f, r in res.items() if not r["ok"] and managed and family_download(f, sess.mdir)]
        say("\nReady: " + (", ".join(ready) or "nothing yet") + (f".  Downloads when you pick them: {', '.join(later)}" if later else ""))
    if not managed and not any(r["ok"] for r in res.values()):
        say("\nNothing is ready in this ComfyUI yet. Put model files in its folders (`--list-models` lists the ones this tool knows).")
        return

    # 3) what to make
    kind = pick("\nWhat do you want to make?", [
        ("A new image from text", ""),
        ("Change an existing image (edit)", "Qwen-Image 2.1 edits images"),
        ("A character from a reference picture in a new scene", "keeps the face and look; Qwen-Image 2.1 mixes generating with a reference"),
        ("An image in the style of a reference picture", "Krea 2 style reference; with a character: Qwen-Image 2.1")])
    argv = []

    # 4) pictures
    edit, chars, styles = None, [], []
    if kind == 1:
        edit = ask_images("\nWhich image should be changed?", 1, optional=False)[0]
        chars = ask_images("\nOptional: a character picture whose face must stay the same.", 1, kind="character")
    elif kind == 2:
        chars = ask_images("\nThe character reference (a clear picture of the person or figure, up to 3):", 3, optional=False, kind="character")
    elif kind == 3:
        chars = ask_images("\nOptional: a character reference (up to 3). Skip for none.", 3, kind="character")
        styles = ask_images("\nThe style reference picture(s) (up to 3):", 3, optional=False, kind="style")
    if edit:
        argv += ["--edit", rel(edit)]
    for p in chars:
        argv += ["--character", rel(p) if os.path.isfile(p) else p]
    for p in styles:
        argv += ["--style", rel(p) if os.path.isfile(p) else p]

    # 5) model and LoRAs
    def label(f):
        todo = family_download(f, sess.mdir) if managed else 0
        return TITLES[f] + (f"  (will ask to download {human(todo)})" if todo and not res[f]["ok"] else "")
    family = None
    if kind == 0 or (kind == 3 and not chars):
        family = "krea2"
    elif kind == 2 and yes("\nPaint the character in a LoRA style or another look? (Krea 2 renders the look first, Qwen adds the character)", "n"):
        family = "krea2"

    def choose_file(fam, flag, title):
        """When ComfyUI has several files for a family you decide which one; Enter takes the first. Another name can be typed."""
        cands = eng.candidates(fam, "unet")
        if len(cands) < 2:
            return
        say(f"\n{title}")
        for i, c in enumerate(cands, 1):
            say(f"   {i}. {c}")
        s = ask("   Number, or type another file name (Enter = 1)", "1")
        chosen = cands[int(s) - 1] if s.isdigit() and 1 <= int(s) <= len(cands) else (s if s != "1" else cands[0])
        if chosen != cands[0]:
            argv.extend([flag, chosen])
    if kind in (1, 2) or chars:
        choose_file("qwen21", "--qwen-model", "Which Qwen-Image 2.1 file?")
    if family:
        choose_file(family, "--model", f"Which {TITLES[family]} model file?")
        for n, s_ in ask_loras(eng, family):
            argv += ["--lora", f"{n}:{s_:g}"]

    # 6) text and format
    say("")
    prompt = ""
    while not prompt:
        prompt = ask("What should the picture show?" if kind != 1
                     else "What should change in the image?")
    argv += ["--prompt", prompt]
    if not edit:
        s = ask("\nShape? " + ", ".join(ASPECTS) + " (Enter = 1:1)", "1:1")
        if s in ASPECTS and s != "1:1":
            argv += ["--aspect", s]
    q = ask("Quality: fast, normal or high (Enter = normal)", "normal").lower()
    if q in ("fast", "high"):
        argv += ["--quality", q]
    n = ask("How many images? (Enter = 1)", "1")
    if n.isdigit() and 2 <= int(n) <= 16:
        argv += ["--count", n]
    if sess.external:
        argv += ["--comfy", sess.external]
    if a.models_dir:
        argv += ["--models-dir", a.models_dir]
    if a.no_ollama:
        argv += ["--no-ollama"]

    say("\nNext time, this one command does everything above without questions:")
    say(f"\n   {shell_join([launcher()] + argv)}\n")
    # the run below is made from exactly that command, so what you see is what the command does
    a2 = ap.parse_args(argv)
    a2.yes, a2.accept_licenses, a2.stdin_text = a.yes, a.accept_licenses, None
    job = build_job(a2, eng)
    sess.prepare(eng, job)
    run_job(eng, job, a2.count, a2.out)
    say("If it isn't what you wanted: change the prompt, try another --seed, or another model. See README.md.")


# ---------- setup and listing ----------

SETUP_NAMES = {"krea": ["krea2"], "krea2": ["krea2"], "qwen": ["qwen21"], "qwen21": ["qwen21"], "krea-style": ["krea2-style"]}


def setup_command(a):
    """--setup [krea] [qwen] [krea-style] [all]: set up ComfyUI, then download what you name."""
    names = [n.lower() for n in a.setup]
    unknown = [n for n in names if n not in SETUP_NAMES and n != "all"]
    if unknown:
        die(f"unknown: {', '.join(unknown)}. Choose from: {', '.join(list(SETUP_NAMES) + ['all'])}")
    ids = list(DEFAULT_IDS) if not names else []   # no name: the whole recommended set
    for n in (list(SETUP_NAMES) if "all" in names else names):
        for t in SETUP_NAMES[n]:
            ids += catalog.FAMILIES.get(t, [t])
    mdir = setup_env.models_dir(a.models_dir)
    if a.comfy or os.environ.get("IMAGE_COMFY_URL"):
        die("--setup sets up the ComfyUI this tool manages; leave out --comfy.")
    if a.dry_run:
        setup_env.install_comfyui(assume_yes=True, dry=True)
        todo = setup_env.missing_of(ids, mdir)
        say(f"\nModel files that would be downloaded into {mdir} (after asking):")
        for i in todo:
            f = catalog.FILES[i]
            say(f"   {pathlib.Path(f['dest']).name}  {human(f['size'])}  {f['repo']}@{f['rev'][:8]}")
        say("   (nothing)" if not todo else f"   total {human(sum(catalog.FILES[i]['size'] for i in todo))}")
        return
    todo = [] if setup_env.OFFLINE else setup_env.missing_of(ids, mdir)
    need_comfy = not setup_env.installed()
    if need_comfy or todo:
        say("\nThis sets up everything in this folder (nothing is installed globally):")
        if need_comfy:
            for line in setup_env.describe_install():
                say("  - " + line)
        if todo:
            setup_env.show_plan(todo, mdir)
        asked = False
        if not a.yes:   # one question for the whole setup, including the licenses
            if not sys.stdin.isatty():
                die("this needs a terminal for the question, or -y (and --accept-licenses) to say yes in advance.")
            if not yes("\nSet up everything now" + (" and accept these licenses?" if todo else "?"), "n"):
                die("cancelled")
            asked = True
        a.yes = True
        a.accept_licenses = a.accept_licenses or (asked and bool(todo))
    if need_comfy:
        setup_env.install_comfyui(assume_yes=True, show=False)
    else:
        say("ComfyUI is already set up in ./comfyui.")
    if ids:
        setup_env.ensure_files(ids, mdir, a.yes, a.accept_licenses, show=False)
    say(f"\nReady. Try:  {launcher()} \"a fox in the snow\"")


def list_models(a):
    mdir = setup_env.models_dir(a.models_dir)
    say(f"Model files this tool knows (models folder: {mdir}). `--setup NAME` downloads them after asking.\n")
    for fam, ids in list(catalog.FAMILIES.items()) + [("extras", list(catalog.EXTRAS))]:
        say(TITLES.get(fam, "Optional extras"))
        for i in ids:
            f = catalog.FILES[i]
            here = "present" if setup_env.present(f, mdir) else "missing"
            say(f"   {here:8} {pathlib.Path(f['dest']).name:46} {human(f['size']):>9}  {catalog.LICENSES[f['license']][0]}")
            if fam == "extras":
                say(f"            {catalog.EXTRAS[i]}")
    say("Licenses and links are in README.md. Other models and LoRAs: just put them in the models folder; the tool finds them.")


# ---------- main ----------

def make_parser():
    ap = argparse.ArgumentParser(
        prog="easy-image",
        description="AI images on your own computer: new images, edits, characters, styles, LoRAs. It sets up and runs its own ComfyUI. "
                    "No arguments = guided mode.",
        epilog="Examples:  easy-image --prompt \"a fox in the snow\"\n"
               "          easy-image --edit photo.png --prompt \"make the jacket red\"\n"
               "          easy-image --character me.png --prompt \"me as an astronaut\"",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*", metavar="PROMPT", help="what the image shows, or (with --edit) what to change; or use --prompt, "
                                                              "--prompt-file, or pipe it in")
    ap.add_argument("--configure", action="store_true", help="choose ComfyUI, the models folder and the models again")
    ap.add_argument("--prompt", help="what the image shows, or (with --edit) what to change")
    ap.add_argument("--prompt-file", help="read the prompt from a text file")
    ap.add_argument("--edit", metavar="IMAGE", help="change this image (done by Qwen-Image 2.1)")
    ap.add_argument("--character", action="append", metavar="IMAGE", help="keep this person/figure's look (repeat for up to 3)")
    ap.add_argument("--style", action="append", metavar="IMAGE", help="copy the look of this image (repeat for up to 3)")
    ap.add_argument("--ref", action="append", metavar="IMAGE", help="an extra reference image (up to 2, Qwen)")
    ap.add_argument("--save-character", metavar="NAME", help="keep the --character picture under this name (then --character NAME works)")
    ap.add_argument("--save-style", metavar="NAME", help="keep the --style picture under this name (then --style NAME works)")
    ap.add_argument("--lora", action="append", metavar="NAME[:STRENGTH]",
                    help="a LoRA from ComfyUI's loras folder (see --list-loras); repeat to stack; strength default 1.0")
    ap.add_argument("--engine", default="auto", choices=ENGINES, help="auto = Krea 2 for new images (Qwen if there is no Krea 2); Qwen when a character or an edit is involved")
    ap.add_argument("--model", help="the Krea 2 model file to use: any file in ComfyUI, "
                                    "a part of its name is enough; without it the first matching file is used (see --check)")
    ap.add_argument("--qwen-model", help="the Qwen-Image 2.1 file to use (default: the first one ComfyUI has)")
    ap.add_argument("--clip", help="text encoder file, instead of the detected one")
    ap.add_argument("--vae", help="VAE file, instead of the detected one")
    ap.add_argument("--two-step", action="store_true",
                    help="with a character and one style image: let the model family render the style first, then Qwen adds the character")
    ap.add_argument("--aspect", default="1:1", choices=list(ASPECTS), help="shape of a new image (an edit keeps the original's size)")
    ap.add_argument("--size", default="1", choices=["1", "1.5", "2"], help="times about one megapixel (default 1)")
    ap.add_argument("--quality", default="normal", choices=["fast", "normal", "high"])
    ap.add_argument("--seed", type=int, help="same seed + same settings = same image; with --count each image adds 1")
    ap.add_argument("--count", type=int, default=1, help="how many images")
    ap.add_argument("--fidelity", default="high", choices=["high", "off"],
                    help="character + style: second pass that re-aligns face, hair and ears with the original")
    ap.add_argument("--eyes", default="off", choices=["on", "off"], help="extra pass that evens out the eyes")
    ap.add_argument("-o", "--out", help="output file (default: out/<prompt>-<time>.png)")
    ap.add_argument("--comfy", help="use a ComfyUI you already run (e.g. http://127.0.0.1:8188, or IMAGE_COMFY_URL) instead of the one this tool "
                                    "sets up and starts in ./comfyui; then you manage its models yourself")
    ap.add_argument("--models-dir", help="the models folder to use and fill (default ./comfyui/models); point it at a folder you already have "
                                         "to share models (or IMAGE_MODELS_DIR)")
    ap.add_argument("--ollama", default=DEFAULT_OLLAMA, help="address of Ollama for the optional helpers (or IMAGE_OLLAMA_URL)")
    ap.add_argument("--chat-model", help="Ollama model that looks at images (default qwen3-vl:8b, or IMAGE_CHAT_MODEL)")
    ap.add_argument("--no-ollama", action="store_true", help="never use Ollama")
    ap.add_argument("--check", action="store_true", help="show which model families are ready")
    ap.add_argument("--list-loras", action="store_true", help="list the LoRAs ComfyUI has")
    ap.add_argument("--list-characters", action="store_true", help="show the characters you saved")
    ap.add_argument("--list-styles", action="store_true", help="show the styles you saved")
    ap.add_argument("--list-models", action="store_true", help="list the model files this tool can download, with sizes and licenses")
    ap.add_argument("--setup", nargs="*", metavar="NAME",
                    help="set up ComfyUI in ./comfyui, then download the named model files (krea, qwen, krea-style, all); "
                         "asks before downloading")
    ap.add_argument("--dry-run", action="store_true", help="with --setup: only print what would be done")
    ap.add_argument("-y", "--yes", action="store_true", help="answer yes to the questions about setting up and downloading")
    ap.add_argument("--accept-licenses", action="store_true",
                    help="confirm that you have read and accept the licenses of the model files that are downloaded")
    ap.add_argument("-v", "--verbose", action="store_true", help="show everything the setup and ComfyUI print")
    ap.add_argument("--offline", action="store_true", help="never download anything (model files must already be there)")
    ap.add_argument("--keep-running", action="store_true", help="leave the ComfyUI this tool started running when it ends (faster next time)")
    ap.add_argument("--guided", action="store_true", help="force the guided mode (it starts by itself when you give no arguments)")
    ap.add_argument("--serve", action="store_true", help="run the HTTP image service instead of generating once")
    ap.add_argument("--host", default="127.0.0.1", help="with --serve: address to listen on (0.0.0.0 = reachable from other computers)")
    ap.add_argument("--port", type=int, default=8192, help="with --serve: port (default 8192)")
    return ap


def main():
    ap = make_parser()
    a = ap.parse_args()
    ui.VERBOSE = a.verbose
    setup_env.OFFLINE = a.offline
    a.stdin_text = None

    if a.list_models:
        return list_models(a)
    if a.list_characters or a.list_styles:
        for kind, wanted in (("character", a.list_characters), ("style", a.list_styles)):
            if wanted:
                say("\n".join(library.names(kind)) or f"No saved {kind}s yet. Save one with: --{kind} FILE --save-{kind} NAME")
        return
    cfg = apply_config(a)
    interactive = sys.stdin.isatty()
    explicit = bool(a.comfy or os.environ.get("IMAGE_COMFY_URL") or a.models_dir)
    ui.banner()
    if a.configure:
        if not interactive:
            die("--configure needs a terminal")
        configure(a, first=not CONFIG.exists())
        return
    if a.setup is not None:
        return setup_command(a)
    if a.check or a.list_loras or a.serve:
        if a.serve:
            ensure_families(a, cfg)
        sess = Session(a)
        try:
            comfy = sess.open(interactive=False)
            eng = Engine(comfy, None if a.list_loras or a.check else make_ollama(a))
            if a.check:
                say(("ComfyUI at " + sess.external if sess.external else f"managed ComfyUI, models in {sess.mdir}") + "\n")
                print_check(eng, None if sess.external else sess.mdir)
            elif a.list_loras:
                say("\n".join(eng.loras()) or "no LoRAs found in ComfyUI's loras folder")
            else:
                import server
                server.serve(eng, a.host, a.port, managed=sess.managed)
        finally:
            sess.close()
        return

    has_prompt = a.prompt is not None or bool(a.words) or bool(a.prompt_file)
    if not (has_prompt or a.edit or a.character or a.style) and not interactive and not a.guided:
        text = sys.stdin.read()   # the prompt can be piped in
        if text.strip():
            a.stdin_text = text
        else:
            ap.print_help()
            return
    if not explicit and not setup_env.installed() and interactive and not a.offline:   # first start: set up the recommended setup
        say("easy-image is not set up yet.")
        a.setup = []
        setup_command(a)
    if has_prompt or a.stdin_text is not None or a.edit or a.character or a.style:
        if not 1 <= a.count <= 64:
            die("--count must be between 1 and 64")
        return headless(a)
    guided(a, ap)


def ensure_families(a, cfg):
    """The service downloads the models of the recommended setup (or the ones you chose with --configure) when it starts (asking nothing, like -y)."""
    if a.comfy or os.environ.get("IMAGE_COMFY_URL"):
        return
    ids = [i for f in cfg["families"] for i in catalog.FAMILIES.get(f, [])] if "families" in cfg else list(DEFAULT_IDS)
    if ids:
        if not setup_env.installed():
            setup_env.install_comfyui(assume_yes=a.yes, interactive=False)
        setup_env.ensure_files(ids, setup_env.models_dir(a.models_dir), a.yes, a.accept_licenses, interactive=False)


if __name__ == "__main__":
    main()
