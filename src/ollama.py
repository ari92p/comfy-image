"""Optional helpers that use a vision/text model served by Ollama: they describe a character or a style from an image and
turn a short edit wish into a clear English instruction. Everything here is optional; without Ollama the image tool still
works, it just skips these helpers."""

import base64
import hashlib
import io
import json
import os
import pathlib
import urllib.request

CHAR_KEYS = ("gender", "apparent_age", "build", "face", "eyes", "hair", "ears", "skin", "marks", "outfit", "accessories")
CHAR_SCHEMA = {"type": "object", "required": list(CHAR_KEYS), "properties": {k: {"type": "string"} for k in CHAR_KEYS}}

CHARACTER_QUESTION = (
    "Analyse this character as a precise identity sheet, JSON only. Be exhaustive and exact: another artist who never sees "
    "the picture must be able to redraw this exact person. Use concrete wording with exact shades (for example 'ash blonde, "
    "darker roots, copper highlights at the ends' rather than 'blonde'), shapes, sizes, positions and left/right asymmetries. "
    "State only what is clearly visible; write 'not visible' for anything hidden. Each value is 2 to 4 full sentences "
    "(except gender and apparent_age). Fields: "
    "gender: woman, man or unclear. apparent_age: age impression as a range. "
    "build: body type, height impression, shoulders, waist, hips, limbs, posture, proportions. "
    "face: face shape, forehead, jaw and chin, cheekbones, cheeks, nose (bridge, tip, nostrils), lips (shape, fullness, colour), "
    "eyebrows (shape, thickness, colour, position), facial hair, makeup. "
    "eyes: iris colour with exact shade and any ring, gradient or glow, eye shape, size, tilt, eyelids, lashes, pupils, eye makeup. "
    "hair: exact colour incl. roots, highlights, gradients, length, texture and curl, volume, parting, bangs or fringe, "
    "hairstyle with every strand group, braids, ties and hair accessories. "
    "ears: 'round human ears' if normally shaped, 'pointed ears' only if clearly pointed (state length), 'not visible' if hidden. "
    "skin: exact skin tone and undertone, texture, freckles, moles, blush, shine, tan lines. "
    "marks: tattoos, scars, birthmarks, markings, piercings, jewellery, glasses, with their exact positions. "
    "outfit: every garment, colour, material, fit, pattern and detail, from top to bottom. "
    "accessories: held or worn extras such as weapons, props, bags, headwear, gloves, belts.")

STYLE_QUESTION = (
    "Analyse ONLY the visual style of this image so that an artist could reproduce the same look on a completely different "
    "subject. Reply in English with one dense paragraph of 6 to 10 sentences, no headings, no lists. Cover, in this order: "
    "1) medium and genre (photo, 3D render, digital painting, anime, watercolour, ...) and the era or school it resembles; "
    "2) rendering technique: brushwork or polygon/texture treatment, edge quality, level of detail and sharpness, how surfaces and "
    "materials (skin, cloth, metal, hair, foliage) are drawn; "
    "3) line work: presence, weight, colour, consistency; "
    "4) shading: soft or hard, cel or gradient, ambient occlusion, contrast, texture grain; "
    "5) lighting: source, direction, colour temperature, shadows, rim light, atmosphere, haze, glow; "
    "6) colour palette: dominant, secondary and accent colours with exact shades, saturation, contrast, colour grading; "
    "7) camera and composition language: lens feel, depth of field, framing habits; "
    "8) overall mood. "
    "Never mention the subject, characters, objects, places or story of the picture; describe only how it is made.")

EDIT_SYSTEM = (
    "You turn a user's image-edit request (any language, possibly slang or very short) into ONE precise English instruction for an "
    "image-editing model. Keep exactly what the user asked and add nothing else. Be concrete about what changes. For anatomy fixes state "
    "the correct anatomy explicitly (for example: each foot has exactly five toes of natural size and shape; each hand has exactly five "
    "fingers). For length or amount changes describe the result (for example: very short, sparse, neatly trimmed stubble-like hair). "
    "Never keep time spans or idioms such as 'three days': say how the result looks, so an image model can draw it. "
    "Name where in the picture the change happens (for example: the feet at the bottom of the image). "
    "Always answer in English, even when the request is already short or partly English. Translate faithfully and do not swap a word for a "
    "different meaning. "
    "Reply with the instruction only: one or two sentences, no quotes, no explanation.")


def to_png(data):
    """PNG bytes of any image Pillow can read; without Pillow the bytes are returned as they are."""
    try:
        from PIL import Image
    except ImportError:
        return data
    img = Image.open(io.BytesIO(data))
    out = io.BytesIO()
    img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB").save(out, "PNG")
    return out.getvalue()


def json_from(text):
    try:
        d = json.loads(text[text.index("{"): text.rindex("}") + 1])
        return d if isinstance(d, dict) else None
    except ValueError:
        return None


class Ollama:
    def __init__(self, url, chat_model=None, cache_dir=None):
        self.url = url.rstrip("/")
        self.chat_model = chat_model or os.environ.get("IMAGE_CHAT_MODEL", "qwen3-vl:8b")
        self.cache = pathlib.Path(cache_dir) if cache_dir else None

    def up(self):
        try:
            urllib.request.urlopen(self.url + "/api/tags", timeout=3).read()
            return True
        except OSError:
            return False

    def chat(self, content, image=None, temperature=0.0, timeout=180, system=None, schema=None):
        msgs = [{"role": "system", "content": system}] if system else []
        msg = {"role": "user", "content": content}
        if image:
            msg["images"] = [base64.b64encode(to_png(image)).decode()]
        payload = {"model": self.chat_model, "messages": msgs + [msg], "stream": False, "think": False,
                   "options": {"temperature": temperature}}
        if schema:
            payload["format"] = schema
        req = urllib.request.Request(self.url + "/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["message"]["content"].strip()

    def describe(self, image, role):
        """A character's visible facts as a JSON string, or one paragraph about an image's art style. '' when it fails.
        Results are kept in the cache folder, keyed by the image, so each image is analysed once."""
        key = None
        if self.cache:
            key = self.cache / f"{hashlib.sha256(image).hexdigest()[:24]}-{role}.txt"
            if key.exists():
                return key.read_text(encoding="utf-8")
        out = ""
        if role == "character":
            for schema in (CHAR_SCHEMA, None):   # structured output first, plain JSON answer as a fallback
                try:
                    raw = self.chat(CHARACTER_QUESTION + ("" if schema else " Reply with the JSON object only."),
                                    image, 0.0, 420, None, schema)
                    d = json_from(raw)
                    if d:
                        out = json.dumps(d, ensure_ascii=False)
                        break
                except Exception as e:
                    print("  (describe failed:", e, ")", flush=True)
        else:
            try:
                out = self.chat(STYLE_QUESTION, image, 0.2, 300).replace("\n", " ").strip()
            except Exception as e:
                print("  (describe failed:", e, ")", flush=True)
        if out and key:
            key.parent.mkdir(parents=True, exist_ok=True)
            key.write_text(out, encoding="utf-8")
        return out

    def edit_instruction(self, text):
        """Edits are typed quickly, often in another language or slang; the editor needs a clear English command."""
        for temp in (0.3, 0.6):   # at very low temperature the model sometimes answers with nothing
            try:
                out = self.chat(text, None, temp, 90, EDIT_SYSTEM)
            except Exception as e:
                print("  (edit instruction failed:", e, ")", flush=True)
                return text
            out = out.replace("\n", " ").strip().strip('"')
            if 8 < len(out) < 600:
                return out
        return text

    def unload(self):
        """Frees the GPU memory Ollama holds before ComfyUI renders. Only models that are really loaded are unloaded (asking
        Ollama to unload an idle model would load it first) and the result is checked: a 7 GB model left in VRAM makes
        Qwen about 40 times slower because it spills into shared memory."""
        import time

        def loaded():
            with urllib.request.urlopen(self.url + "/api/ps", timeout=10) as r:
                return [m["name"] for m in json.load(r).get("models", [])]

        try:
            for _ in range(5):
                names = loaded()
                if not names:
                    return
                for n in names:
                    req = urllib.request.Request(self.url + "/api/generate", data=json.dumps({"model": n, "keep_alive": 0}).encode(),
                                                 headers={"Content-Type": "application/json"})
                    urllib.request.urlopen(req, timeout=30).read()
                time.sleep(1.5)
            print("  (warning: Ollama models still loaded:", names, ")", flush=True)
        except Exception as e:
            print("  (unload failed:", e, ")", flush=True)
