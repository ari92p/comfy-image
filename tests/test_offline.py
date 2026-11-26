"""Offline tests: run with `python3 -m unittest discover -s tests` from the repo folder. No ComfyUI, GPU or network needed.

They check that the right graphs are built and sent in the right order for every kind of job, that file names are matched
the way ComfyUI spells them, and that the command line, the guided mode and the HTTP service work end to end against a
fake ComfyUI. They do NOT check how an image looks: for that see "What has been tested" in the README."""

import base64
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from fake_comfy import FakeComfy, FakeOllama, tiny_png   # noqa: E402
from comfy import Comfy, find_name, search_names   # noqa: E402
from ollama import Ollama   # noqa: E402
from pipeline import Engine, Job, Missing   # noqa: E402

PNG = tiny_png()


def kinds(graph):
    return sorted(n["class_type"] for n in graph.values())


def node(graph, ct):
    return [n for n in graph.values() if n["class_type"] == ct]


class Base(unittest.TestCase):
    def setUp(self):
        self.fake = FakeComfy()
        self.eng = Engine(Comfy(self.fake.url), None)
        self.say = []

    def tearDown(self):
        self.fake.stop()

    def gen(self, **kw):
        job = Job(**{"prompt": "a test prompt", **kw})
        data, info = self.eng.generate(job, self.say.append)
        self.assertEqual(data, PNG)
        return info


class NameMatching(unittest.TestCase):
    def test_find_name(self):
        inst = ["krea2\\a.safetensors", "other\\b.safetensors", "plain.safetensors"]
        self.assertEqual(find_name("krea2/a.safetensors", inst), "krea2\\a.safetensors")
        self.assertEqual(find_name("a.safetensors", inst), "krea2\\a.safetensors")
        self.assertEqual(find_name("plain.safetensors", inst), "plain.safetensors")
        self.assertIsNone(find_name("zzz.safetensors", inst))

    def test_ambiguous_bare_name_is_not_guessed(self):
        self.assertIsNone(find_name("x.safetensors", ["a\\x.safetensors", "b\\x.safetensors"]))

    def test_search(self):
        inst = ["krea2\\krea2_softwatercolor.safetensors", "krea2\\krea2_darkbrush.safetensors"]
        self.assertEqual(search_names("watercolor", inst), [inst[0]])
        self.assertEqual(search_names("krea2_darkbrush", inst), [inst[1]])


class Readiness(Base):
    def test_all_ready(self):
        res = self.eng.check()
        self.assertTrue(all(r["ok"] for r in res.values()), {k: v["problems"] for k, v in res.items()})
        self.assertTrue(res["krea2"]["styles"])
        self.assertEqual(res["qwen21"]["files"]["unet"], "qwen-image-2.1-Q6_K.gguf")   # regular Q6 before UC, Q6 before Q4

    def test_other_files_of_a_family_are_detected_not_dictated(self):
        # the default Krea 2 model and text encoder are gone; other files that look like Krea 2 are used instead
        self.fake.drop_files.update({"krea2_turbo_fp8_scaled.safetensors", "qwen3vl_4b_fp8_scaled.safetensors"})
        self.eng.comfy.refresh()
        r = self.eng.check()["krea2"]
        self.assertTrue(r["ok"], r["problems"])
        self.assertEqual((r["files"]["unet"], r["files"]["clip"]), ("sample_krea2_finetune.safetensors", "qwen3vl_4b_bf16.safetensors"))
        self.gen()
        self.assertEqual(node(self.fake.graphs[0], "UNETLoader")[0]["inputs"]["unet_name"], "sample_krea2_finetune.safetensors")

    def test_auto_uses_what_is_ready(self):
        self.fake.drop_files.update({"krea2_turbo_fp8_scaled.safetensors", "sample_krea2_finetune.safetensors"})
        self.eng.comfy.refresh()
        self.assertEqual(self.gen()["family"], "qwen21")     # there is no Krea 2 here: Qwen draws it instead

    def test_you_can_name_clip_and_vae(self):
        self.gen(clip="qwen3vl_4b_bf16", vae="qwen_image_vae")
        self.assertEqual(node(self.fake.graphs[0], "CLIPLoader")[0]["inputs"]["clip_name"], "qwen3vl_4b_bf16.safetensors")

    def test_missing_files_are_named(self):
        self.fake.drop_files.update({"qwen3vl_4b_fp8_scaled.safetensors", "qwen3vl_4b_bf16.safetensors"})
        self.eng.comfy.refresh()
        res = self.eng.check()
        self.assertFalse(res["krea2"]["ok"])
        self.assertIn("text encoder", res["krea2"]["problems"])
        self.assertIn("models/text_encoders", res["krea2"]["problems"])
        self.assertTrue(res["qwen21"]["ok"])

    def test_missing_node(self):
        self.fake.missing_nodes.add("QwenImage21Cache")
        self.eng.comfy.refresh()
        self.assertIn("QwenImage21Cache", self.eng.check()["qwen21"]["problems"])

    def test_gguf_node_needed_for_gguf(self):
        self.fake.missing_nodes.add("UnetLoaderGGUF")
        self.fake.drop_files.add("qwen-image-2.1-UC-int8_convrot.safetensors")
        self.eng.comfy.refresh()
        self.assertFalse(self.eng.check()["qwen21"]["ok"])


class Jobs(Base):
    def test_new_image_krea(self):
        info = self.gen(seed=7)
        g = self.fake.graphs[0]
        self.assertEqual(info["mode"], "krea2")
        self.assertEqual(node(g, "KSampler")[0]["inputs"]["steps"], 8)
        self.assertEqual(node(g, "KSampler")[0]["inputs"]["seed"], 7)
        self.assertEqual(node(g, "KSampler")[0]["inputs"]["cfg"], 1.0)
        self.assertEqual((node(g, "EmptyLatentImage")[0]["inputs"]["width"], node(g, "EmptyLatentImage")[0]["inputs"]["height"]), (1024, 1024))
        self.assertEqual(len(self.fake.graphs), 1)

    def test_aspect_size_quality(self):
        self.gen(aspect="16:9", size="1.5", quality="high")
        i = node(self.fake.graphs[0], "EmptyLatentImage")[0]["inputs"]
        self.assertEqual((i["width"], i["height"]), (2016, 1152))
        self.assertEqual(node(self.fake.graphs[0], "KSampler")[0]["inputs"]["steps"], 12)

    def test_loras_stack_and_use_comfyui_spelling(self):
        loras = [(self.eng.resolve_lora("watercolor"), 0.6), (self.eng.resolve_lora("krea2/krea2_darkbrush"), 0.5)]
        self.gen(loras=loras)
        chain = [(n["inputs"]["lora_name"], n["inputs"]["strength_model"]) for n in node(self.fake.graphs[0], "LoraLoaderModelOnly")]
        self.assertEqual(chain, [("krea2\\krea2_softwatercolor.safetensors", 0.6), ("krea2\\krea2_darkbrush.safetensors", 0.5)])

    def test_ambiguous_lora_is_refused(self):
        with self.assertRaises(Missing):
            self.eng.resolve_lora("krea2")
        with self.assertRaises(Missing):
            self.eng.resolve_lora("does-not-exist")

    def test_model_names_are_matched_loosely(self):
        self.gen(engine="krea", model="sample_krea2_finetune")
        self.assertEqual(node(self.fake.graphs[0], "UNETLoader")[0]["inputs"]["unet_name"], "sample_krea2_finetune.safetensors")

    def test_krea_style_images(self):
        self.gen(styles=[PNG, PNG])
        g = self.fake.graphs[0]
        self.assertEqual(len(node(g, "LoadImage")), 2)
        self.assertEqual([n["inputs"]["lora_name"] for n in node(g, "LoraLoaderModelOnly")], ["krea2\\krea2_style_reference.safetensors"])
        self.assertTrue(node(g, "FluxKontextMultiReferenceLatentMethod"))
        self.assertEqual(len(self.fake.uploads), 2)

    def test_qwen_text_to_image(self):
        info = self.gen(engine="qwen", aspect="3:2")
        g = self.fake.graphs[0]
        self.assertEqual(info["mode"], "qwen")
        self.assertEqual(node(g, "UnetLoaderGGUF")[0]["inputs"]["unet_name"], "qwen-image-2.1-Q6_K.gguf")
        self.assertEqual(node(g, "KSampler")[0]["inputs"]["steps"], 25)
        e = node(g, "EmptyFlux2LatentImage")[0]["inputs"]
        self.assertEqual((e["width"], e["height"]), (1248, 832))

    def test_edit_keeps_size_and_never_reuses_the_seed(self):
        info = self.gen(edit=PNG, seed=5, prompt="make the jacket red")
        g = self.fake.graphs[0]
        self.assertEqual(info["mode"], "edit")
        self.assertEqual(len(self.fake.graphs), 1)
        self.assertFalse(node(g, "EmptyFlux2LatentImage"))
        self.assertEqual(node(g, "TextEncodeQwenImage21")[0]["inputs"]["resolution"], 0)
        self.assertIn("Edit <image1>: make the jacket red", node(g, "TextEncodeQwenImage21")[0]["inputs"]["prompt"])
        self.assertNotEqual(node(g, "KSampler")[0]["inputs"]["seed"], 5)

    def test_character_goes_to_qwen(self):
        info = self.gen(characters=[PNG], prompt="walking on the moon")
        g = self.fake.graphs[0]
        self.assertEqual(info["mode"], "qwen")
        self.assertEqual(len(self.fake.graphs), 1)
        p = node(g, "TextEncodeQwenImage21")[0]["inputs"]
        self.assertIn("images.image_1", p)
        self.assertIn("exact same identity", p["prompt"])

    def test_character_and_one_style_is_qwen_with_fidelity_pass(self):
        info = self.gen(characters=[PNG], styles=[PNG])
        self.assertEqual(info["mode"], "qwen")
        self.assertEqual(len(self.fake.graphs), 2)   # main render + face re-alignment; no ears pass without a description
        self.assertTrue(info["fidelity_pass"])
        self.assertNotIn("ears_pass", info)
        first = node(self.fake.graphs[0], "TextEncodeQwenImage21")[0]["inputs"]
        self.assertIn("images.image_2", first)
        self.assertIn("visual style of <image2>", first["prompt"])
        second = node(self.fake.graphs[1], "TextEncodeQwenImage21")[0]["inputs"]
        self.assertEqual(second["resolution"], 0)

    def test_fidelity_off(self):
        info = self.gen(characters=[PNG], styles=[PNG], fidelity="off")
        self.assertEqual(len(self.fake.graphs), 1)
        self.assertNotIn("fidelity_pass", info)

    def test_character_in_a_lora_style_is_two_steps(self):
        info = self.gen(characters=[PNG], loras=[("krea2\\krea2_softwatercolor.safetensors", 0.8)])
        self.assertEqual(info["mode"], "krea2+qwen")
        self.assertIn("LoraLoaderModelOnly", kinds(self.fake.graphs[0]))   # Krea first ...
        self.assertIn("QwenImage21Cache", kinds(self.fake.graphs[1]))       # ... then Qwen paints the character in
        self.assertEqual(len(node(self.fake.graphs[1], "LoadImage")), 2)    # character + the Krea render as the style

    def test_two_step_flag(self):
        info = self.gen(characters=[PNG], styles=[PNG], two_step=True)
        self.assertEqual(info["mode"], "krea2+qwen")

    def test_loras_are_ignored_in_qwen_only_modes(self):
        self.gen(edit=PNG, loras=[("krea2\\krea2_softwatercolor.safetensors", 1.0)])
        self.assertTrue(any("ignored" in s for s in self.say))
        self.assertNotIn("LoraLoaderModelOnly", kinds(self.fake.graphs[0]))

    def test_missing_model_fails_before_anything_runs(self):
        self.fake.drop_files.update({"krea2_turbo_fp8_scaled.safetensors", "sample_krea2_finetune.safetensors"})
        self.eng.comfy.refresh()
        with self.assertRaises(Missing) as c:
            self.gen(engine="krea")
        self.assertIn("models/diffusion_models", str(c.exception))
        self.assertEqual(self.fake.graphs, [])
        self.assertEqual(self.fake.uploads, [])

    def test_empty_prompt(self):
        with self.assertRaises(ValueError):
            self.gen(prompt="  ")

    def test_graphs_never_clear_the_queue(self):
        self.gen()
        self.assertEqual(self.fake.deleted, [])


class WithOllama(unittest.TestCase):
    def setUp(self):
        self.fake, self.ollama = FakeComfy(), FakeOllama()
        self.tmp = tempfile.TemporaryDirectory()
        self.oll = Ollama(self.ollama.url, cache_dir=self.tmp.name)
        self.eng = Engine(Comfy(self.fake.url), self.oll)
        # ComfyUI refuses (400) every render that arrives while Ollama still holds a model in memory
        self.fake.guard = lambda: [f"Ollama still holds {sorted(self.ollama.loaded)} while ComfyUI renders"] if self.ollama.loaded else []

    def tearDown(self):
        self.fake.stop()
        self.ollama.stop()
        self.tmp.cleanup()

    def test_description_goes_into_the_prompt_and_is_cached(self):
        self.eng.generate(Job(prompt="in a forest", characters=[PNG]), lambda t: None)
        p = node(self.fake.graphs[0], "TextEncodeQwenImage21")[0]["inputs"]["prompt"]
        self.assertIn("hair: short brown hair", p)
        self.assertNotIn("grey coat", p)   # clothing is not identity
        self.assertIn("normal round human ears", p)
        n = len(self.ollama.chats)
        self.eng.generate(Job(prompt="again", characters=[PNG]), lambda t: None)
        self.assertEqual(len(self.ollama.chats), n)   # the second run reads the cache

    def test_ears_pass_runs_with_style_when_ears_are_round(self):
        info = self.eng.generate(Job(prompt="x", characters=[PNG], styles=[PNG]), lambda t: None)[1]
        self.assertTrue(info["ears_pass"] and info["fidelity_pass"])
        self.assertEqual(len(self.fake.graphs), 3)

    def test_pointed_ears_skip_the_ears_pass(self):
        self.ollama.ears = "pointed ears, long"
        info = self.eng.generate(Job(prompt="x", characters=[PNG], styles=[PNG]), lambda t: None)[1]
        self.assertNotIn("ears_pass", info)

    def test_ollama_is_unloaded_before_every_render(self):
        # new character + new style + edit wish: three kinds of Ollama work, none may still be loaded when ComfyUI renders
        self.eng.generate(Job(prompt="x", characters=[PNG], styles=[PNG]), lambda t: None)
        self.eng.generate(Job(prompt="fix the sleeve", edit=PNG, characters=[PNG]), lambda t: None)
        self.eng.generate(Job(prompt="x", characters=[PNG], loras=[("krea2\\krea2_softwatercolor.safetensors", 1.0)]), lambda t: None)
        self.assertEqual(self.fake.errors, [])
        self.assertEqual(self.ollama.loaded, set())

    def test_edit_wish_is_translated(self):
        self.eng.generate(Job(prompt="mach die jacke rot", edit=PNG), lambda t: None)
        p = node(self.fake.graphs[0], "TextEncodeQwenImage21")[0]["inputs"]["prompt"]
        self.assertIn("clearly red", p)


class CommandLine(unittest.TestCase):
    def setUp(self):
        self.fake = FakeComfy()
        self.tmp = tempfile.TemporaryDirectory()
        self.env = {**os.environ, "IMAGE_COMFY_URL": self.fake.url, "IMAGE_OLLAMA_URL": "http://127.0.0.1:9",
                    "IMAGE_OUT_DIR": self.tmp.name + "/out", "IMAGE_CACHE_DIR": self.tmp.name + "/cache",
                    "IMAGE_CONFIG": self.tmp.name + "/config.json", "IMAGE_LIBRARY_DIR": self.tmp.name + "/library"}

    def tearDown(self):
        self.fake.stop()
        self.tmp.cleanup()

    def run_cli(self, *args, stdin=None, env=None):
        return subprocess.run([sys.executable, str(ROOT / "src" / "easy-image.py"), *args], capture_output=True, text=True, input=stdin,
                              env=env or self.env, cwd=self.tmp.name, timeout=120)

    def test_headless(self):
        out = pathlib.Path(self.tmp.name) / "fox.png"
        r = self.run_cli("--prompt", "a fox", "--seed", "3", "-o", str(out), "--no-ollama")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(out.read_bytes(), PNG)
        self.assertEqual(json.loads(out.with_suffix(".json").read_text())["info"]["seed"], 3)

    def test_count_numbers_files_and_seeds(self):
        out = pathlib.Path(self.tmp.name) / "set.png"
        r = self.run_cli("--prompt", "a fox", "--seed", "10", "--count", "2", "-o", str(out), "--no-ollama")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((pathlib.Path(self.tmp.name) / "set-1.png").exists() and (pathlib.Path(self.tmp.name) / "set-2.png").exists())
        self.assertEqual([node(g, "KSampler")[0]["inputs"]["seed"] for g in self.fake.graphs], [10, 11])

    def test_check_and_list_loras(self):
        r = self.run_cli("--check")
        self.assertIn("ready", r.stdout)
        self.assertNotIn("NOT ready", r.stdout)
        self.assertIn("krea2\\krea2_softwatercolor.safetensors", self.run_cli("--list-loras").stdout)

    def test_unreachable_comfyui_is_explained(self):
        self.env["IMAGE_COMFY_URL"] = "http://127.0.0.1:9"
        r = self.run_cli("--prompt", "x", "--no-ollama")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not reachable", r.stderr)

    def test_missing_image_file(self):
        r = self.run_cli("--edit", "nope.png", "--prompt", "x", "--no-ollama")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("image not found", r.stderr)

    def test_setup_dry_run_only_prints(self):
        env = {k: v for k, v in self.env.items() if k != "IMAGE_COMFY_URL"}
        env["IMAGE_COMFY_DIR"] = self.tmp.name + "/comfyui"
        env["IMAGE_MODELS_DIR"] = self.tmp.name + "/models"   # not the real folder, which may hold the models
        had_comfy = (ROOT / "comfyui").exists()
        r = self.run_cli("--setup", "krea", "--dry-run", env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        for expect in ("ComfyUI/archive/98cc0bf84b7c", "373048b8", "uv venv --python 3.13", "uv venv --python 3.13", "torch==2.14.0", "krea2_turbo_fp8_scaled.safetensors"):
            self.assertIn(expect, r.stdout)
        self.assertNotIn("qwen-image-2.1-Q6_K", r.stdout)
        self.assertFalse((pathlib.Path(self.tmp.name) / "comfyui").exists())
        self.assertEqual((ROOT / "comfyui").exists(), had_comfy)   # a dry run changes nothing

    def test_quality_tiers_pick_the_matching_qwen_file(self):
        import catalog as c
        try:
            for tier, want in (("max", "qwen21-unet-f16"), ("best", "qwen21-unet-q8"), ("balanced", "qwen21-unet"), ("smallest", "qwen21-unet-q4")):
                c.apply_tier(tier)
                self.assertEqual(c.FAMILIES["qwen21"][0], want)
                self.assertEqual(c.FIXES[("qwen21", "unet")], [want])
                self.assertIn(c.TIER_TAGS[tier], c.FILES[want]["dest"].lower())
        finally:
            c.apply_tier("balanced")
        self.assertEqual(c.apply_tier("nonsense"), None)
        self.assertEqual(c.TIER, "balanced")

    def test_tiers_offered_fit_the_memory(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("easy_image_cli_t", ROOT / "src" / "easy-image.py")
        image = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(image)
        keys, titles, rec = image.tier_options(24.0)
        self.assertEqual((keys, rec, titles["best"]), (["best", "balanced", "smallest"], "balanced", "Best quality"))
        keys, titles, rec = image.tier_options(48.0)
        self.assertEqual((keys[0], rec, titles["max"]), ("max", "max", "Best quality"))
        self.assertEqual(image.tier_options(8.0)[2], "smallest")

    def test_setup_env_name_means_comfyui_only(self):
        r = self.run_cli("--setup", "env", "--dry-run", env={**{k: v for k, v in self.env.items() if k != "IMAGE_COMFY_URL"},
                                                            "IMAGE_COMFY_DIR": self.tmp.name + "/comfyui"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ComfyUI/archive", r.stdout)
        self.assertIn("(nothing)", r.stdout)

    def test_setup_refuses_an_external_comfyui(self):
        r = self.run_cli("--setup", "--dry-run")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("leave out --comfy", r.stderr)

    def test_list_models(self):
        env = {k: v for k, v in self.env.items() if k != "IMAGE_COMFY_URL"}
        env["IMAGE_MODELS_DIR"] = self.tmp.name + "/models"
        r = self.run_cli("--list-models", env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("missing  krea2_turbo_fp8_scaled.safetensors", r.stdout)
        self.assertIn("Qwen Research License", r.stdout)

    def test_the_prompt_can_be_an_argument_or_piped_in(self):
        out = pathlib.Path(self.tmp.name) / "a.png"
        r = self.run_cli("a fox in the snow", "--no-ollama", "-o", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("easy-image", r.stdout)   # the name is printed on start
        self.assertEqual(node(self.fake.graphs[-1], "CLIPTextEncode")[0]["inputs"]["text"], "a fox in the snow")
        r = self.run_cli("--no-ollama", "-o", str(out), stdin="a lighthouse at dawn\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(node(self.fake.graphs[-1], "CLIPTextEncode")[0]["inputs"]["text"], "a lighthouse at dawn")
        r = self.run_cli("one prompt", "--prompt", "and another", "--no-ollama")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("prompt once", r.stderr)

    def test_save_and_reuse_a_character_and_a_style(self):
        img = pathlib.Path(self.tmp.name) / "me.png"
        img.write_bytes(PNG)
        r = self.run_cli("--character", str(img), "--save-character", "Me Myself", "--style", str(img), "--save-style", "ink", "walking", "--no-ollama")
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn("Saved character 'me-myself'", r.stdout)
        self.assertEqual(self.run_cli("--list-characters").stdout.strip(), "me-myself")
        self.assertEqual(self.run_cli("--list-styles").stdout.strip(), "ink")
        before = len(self.fake.uploads)
        r = self.run_cli("--character", "me-myself", "--style", "ink", "on the moon", "--no-ollama")   # no file needed any more
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual(len(self.fake.uploads) - before, 2 + 2)   # character + style, then the face pass uploads the image and the character again
        r = self.run_cli("--character", "nobody", "x", "--no-ollama")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Saved characters: me-myself", r.stderr)

    def test_saving_needs_exactly_one_picture(self):
        r = self.run_cli("--save-character", "x", "hello", "--no-ollama")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("exactly one --character FILE", r.stderr)

    def test_choices_in_config_json_are_used(self):
        pathlib.Path(self.tmp.name, "config.json").write_text(json.dumps({"comfy": self.fake.url}))
        env = {k: v for k, v in self.env.items() if k != "IMAGE_COMFY_URL"}
        r = self.run_cli("--check", env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(self.fake.url, r.stdout)

    def test_a_first_start_without_a_terminal_does_not_ask(self):
        env = {k: v for k, v in self.env.items() if k != "IMAGE_COMFY_URL"}
        env["IMAGE_COMFY_DIR"] = self.tmp.name + "/none"
        r = self.run_cli("a fox", "--no-ollama", env=env, stdin="")
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((pathlib.Path(self.tmp.name) / "config.json").exists())

    def test_guided_prints_the_command_and_runs_it(self):
        img = pathlib.Path(self.tmp.name) / "me.png"
        img.write_bytes(PNG)
        # 1 = edit, image, no character, prompt, quality, count
        r = self.run_cli("--guided", stdin=f"2\n{img}\n\n\nmake it red\nhigh\n1\n", )
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn("--edit", r.stdout)
        self.assertIn("--quality high", r.stdout)
        self.assertIn("Saved:", r.stdout)
        self.assertEqual(len(self.fake.graphs), 1)
        self.assertEqual(node(self.fake.graphs[0], "KSampler")[0]["inputs"]["steps"], 40)

    def test_guided_lets_you_pick_the_model_file(self):
        # new image, Krea 2 file 2 of 2, no LoRA, prompt, shape, quality, count
        r = self.run_cli("--guided", stdin="1\n2\n\na fox\n\n\n1\n")
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn("--model sample_krea2_finetune.safetensors", r.stdout)
        self.assertEqual(node(self.fake.graphs[0], "UNETLoader")[0]["inputs"]["unet_name"], "sample_krea2_finetune.safetensors")

    def test_guided_new_image_with_a_lora(self):
        # 1 = new image, Enter = first Krea file, LoRA 1 at 0.7, prompt, shape, quality, count
        r = self.run_cli("--guided", stdin="1\n\n1:0.7\na lighthouse\n16:9\nfast\n1\n")
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn("--lora", r.stdout)
        g = self.fake.graphs[0]
        self.assertEqual(node(g, "LoraLoaderModelOnly")[0]["inputs"]["strength_model"], 0.7)
        self.assertEqual(node(g, "EmptyLatentImage")[0]["inputs"]["width"], 1344)
        self.assertEqual(node(g, "KSampler")[0]["inputs"]["steps"], 6)


class CatalogAndDownloads(unittest.TestCase):
    def setUp(self):
        import catalog
        import setup_env
        self.catalog, self.setup_env = catalog, setup_env
        self.tmp = tempfile.TemporaryDirectory()
        self.mdir = pathlib.Path(self.tmp.name) / "models"
        self.payload = os.urandom(300_000) + b"end"
        import hashlib
        self.entry = dict(repo="org/repo", rev="abc123", path="sub/model.bin", dest="diffusion_models/model.bin", size=len(self.payload),
                          sha256=hashlib.sha256(self.payload).hexdigest(), license="krea2")
        self.saved = dict(catalog.FILES)
        catalog.FILES["test-file"] = self.entry
        self.srv = self.serve(self.payload)
        os.environ["IMAGE_HF_ENDPOINT"] = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.cache = pathlib.Path(self.tmp.name) / "cache"
        setup_env.CACHE = self.cache

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.catalog.FILES.clear()
        self.catalog.FILES.update(self.saved)
        os.environ.pop("IMAGE_HF_ENDPOINT", None)
        self.tmp.cleanup()

    def serve(self, payload):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        test = self
        test.requests = []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                test.requests.append((self.path, self.headers.get("Range")))
                if self.path != "/org/repo/resolve/abc123/sub/model.bin":
                    self.send_response(404)
                    self.end_headers()
                    return
                rng = self.headers.get("Range")
                start = int(rng.split("=")[1].rstrip("-")) if rng else 0
                body = test.payload_override if getattr(test, "payload_override", None) else payload
                self.send_response(206 if rng else 200)
                self.send_header("Content-Length", str(len(body) - start))
                self.end_headers()
                self.wfile.write(body[start:])
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    def test_catalog_is_consistent(self):
        c = self.saved
        for i, f in c.items():
            self.assertTrue(len(f["sha256"]) == 64 and f["size"] > 0 and f["rev"] and f["license"] in self.catalog.LICENSES, i)
        dests = [f["dest"] for f in c.values()]
        self.assertEqual(len(dests), len(set(dests)))
        for ids in list(self.catalog.FAMILIES.values()) + [list(self.catalog.EXTRAS)] + list(self.catalog.FIXES.values()):
            for i in ids:
                self.assertIn(i, c)
        self.assertEqual(len(self.catalog.COMFYUI["commit"]), 40)
        self.assertEqual(len(self.catalog.GGUF_NODE["commit"]), 40)

    def test_download_verifies_and_moves_into_place(self):
        p = self.setup_env.download(self.entry, self.mdir, lambda *a: None)
        self.assertEqual(p.read_bytes(), self.payload)
        self.assertFalse(list(self.mdir.rglob("*.part")))
        self.assertEqual(self.setup_env.missing_of(["test-file"], self.mdir), [])

    def test_download_resumes(self):
        part = self.mdir / "diffusion_models" / "model.bin.part"
        part.parent.mkdir(parents=True)
        part.write_bytes(self.payload[:100_000])
        self.setup_env.download(self.entry, self.mdir, lambda *a: None)
        self.assertEqual((self.mdir / "diffusion_models" / "model.bin").read_bytes(), self.payload)
        self.assertEqual(self.requests[-1][1], "bytes=100000-")

    def test_wrong_checksum_is_refused_and_removed(self):
        self.payload_override = b"x" * len(self.payload)
        with self.assertRaises(ValueError):
            self.setup_env.download(self.entry, self.mdir, lambda *a: None)
        self.assertEqual(list(self.mdir.rglob("model.bin*")), [])

    def test_present_finds_the_file_in_any_subfolder_only_if_the_size_matches(self):
        sub = self.mdir / "loras" / "other"
        sub.mkdir(parents=True)
        (sub / "model.bin").write_bytes(b"short")
        self.assertEqual(self.setup_env.missing_of(["test-file"], self.mdir), ["test-file"])
        (sub / "model.bin").write_bytes(self.payload)
        self.assertEqual(self.setup_env.missing_of(["test-file"], self.mdir), [])

    def test_downloading_needs_a_yes(self):
        with self.assertRaises(SystemExit):   # not interactive, no -y
            self.setup_env.ensure_files(["test-file"], self.mdir, assume_yes=False, interactive=False)
        self.assertEqual(self.requests, [])   # nothing was fetched
        got = self.setup_env.ensure_files(["test-file"], self.mdir, assume_yes=True, interactive=False)
        self.assertEqual(got, ["test-file"])
        self.assertEqual(self.setup_env.ensure_files(["test-file"], self.mdir, True, False), [])   # already there: nothing to do

    def test_setup_commands_are_quiet_unless_verbose(self):
        import contextlib
        import io
        import ui
        ok = [sys.executable, "-c", "print('lots of installer chatter')"]
        bad = [sys.executable, "-c", "import sys; print('boom detail'); sys.exit(3)"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.setup_env._run(ok)
        self.assertEqual(buf.getvalue(), "")
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.setup_env._run(bad)
        self.assertIn("boom detail", err.getvalue())
        ui.VERBOSE = True
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.setup_env._run(ok)
            self.assertIn("$ ", buf.getvalue())   # with -v the command itself is shown
        finally:
            ui.VERBOSE = False

    def test_source_archive_is_unpacked_without_git(self):
        import io
        import zipfile
        from unittest import mock
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("ComfyUI-abc123/main.py", "print('hi')\n")
            z.writestr("ComfyUI-abc123/requirements.txt", "numpy\n")
        dest = pathlib.Path(self.tmp.name) / "comfyui"
        (dest / "models").mkdir(parents=True)
        (dest / "models" / "keep.bin").write_bytes(b"x")   # something already there must survive
        urls = []

        def fake_open(req, timeout=None):
            urls.append(req.full_url)
            return io.BytesIO(buf.getvalue())
        with mock.patch.object(self.setup_env.urllib.request, "urlopen", fake_open):
            self.setup_env.fetch_source("https://github.com/Comfy-Org/ComfyUI.git", "abc123", dest)
        self.assertEqual(urls, ["https://github.com/Comfy-Org/ComfyUI/archive/abc123.zip"])
        self.assertTrue((dest / "main.py").exists())
        self.assertTrue((dest / "requirements.txt").exists())
        self.assertTrue((dest / "models" / "keep.bin").exists())

    def test_offline_never_downloads(self):
        self.setup_env.OFFLINE = True
        try:
            with self.assertRaises(SystemExit):
                self.setup_env.ensure_files(["test-file"], self.mdir, True, True, False)
        finally:
            self.setup_env.OFFLINE = False
        self.assertEqual(self.requests, [])

    def test_first_start_asks_remembers_and_sets_up(self):
        import argparse
        from unittest import mock
        import importlib.util
        spec = importlib.util.spec_from_file_location("easy_image_cli", ROOT / "src" / "easy-image.py")
        image = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(image)
        cfg_path = pathlib.Path(self.tmp.name) / "config.json"
        answers = iter(["", self.tmp.name + "/mymodels", "1 2"])   # own ComfyUI, a models folder, Krea 2 + Qwen
        calls = {}
        a = argparse.Namespace(yes=False, comfy=None, models_dir=None)
        with mock.patch.object(image, "CONFIG", cfg_path), mock.patch.object(image, "ask", lambda *x, **k: next(answers)), \
                mock.patch.object(self.setup_env, "installed", lambda: False), \
                mock.patch.object(self.setup_env, "install_comfyui", lambda **k: calls.setdefault("install", k)), \
                mock.patch.object(self.setup_env, "ensure_files", lambda ids, mdir, *x: calls.setdefault("files", (list(ids), str(mdir)))):
            image.configure(a, first=True)
        saved = json.loads(cfg_path.read_text())
        self.assertEqual(saved["families"], ["krea2", "qwen21"])
        self.assertEqual(saved["comfy"], "")
        self.assertIn("install", calls)
        self.assertEqual(calls["files"][0], ["krea2-unet", "krea2-clip", "qwen-image-vae", "qwen21-unet", "qwen21-clip", "qwen21-vae"])
        self.assertTrue(calls["files"][1].endswith("mymodels"))
        self.assertEqual(a.models_dir, self.tmp.name + "/mymodels")

    def test_setup_without_names_sets_up_everything_with_one_question(self):
        import argparse
        import importlib.util
        from unittest import mock
        spec = importlib.util.spec_from_file_location("easy_image_cli2", ROOT / "src" / "easy-image.py")
        image = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(image)
        asked, calls = [], {}
        a = argparse.Namespace(setup=[], yes=False, comfy=None, models_dir=self.tmp.name + "/m", dry_run=False)
        with mock.patch.object(image, "CONFIG", pathlib.Path(self.tmp.name) / "cfg.json"), \
                mock.patch.object(image, "go", lambda *x, **k: asked.append(x) or True), \
                mock.patch.object(image, "ask", lambda *x, **k: ""), \
                mock.patch.object(sys.stdin, "isatty", lambda: True), \
                mock.patch.object(image.setup_env, "installed", lambda: False), \
                mock.patch.object(image.setup_env, "install_comfyui", lambda **k: calls.setdefault("install", k)), \
                mock.patch.object(image.setup_env, "ensure_files", lambda ids, mdir, *x, **k: calls.setdefault("files", (list(ids), x, k))):
            image.setup_command(a)
        self.assertEqual(len(asked), 1)   # one question for ComfyUI (then it is installed); the models are picked by Enter
        self.assertTrue(calls["install"]["assume_yes"])
        ids, args, _ = calls["files"]
        self.assertEqual(ids, image.default_ids())
        for must in ("krea2-unet", "qwen21-unet", "krea2-style"):
            self.assertIn(must, ids)
        self.assertEqual(args, (True,))

    def test_setup_can_install_the_environment_without_models(self):
        import argparse
        import importlib.util
        from unittest import mock
        spec = importlib.util.spec_from_file_location("easy_image_cli4", ROOT / "src" / "easy-image.py")
        image = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(image)
        calls = {}
        answers = iter([True])   # install ComfyUI: yes; at the model selection the user types n
        a = argparse.Namespace(setup=[], yes=False, comfy=None, models_dir=self.tmp.name + "/m", dry_run=False)
        with mock.patch.object(image, "CONFIG", pathlib.Path(self.tmp.name) / "cfg.json"), \
                mock.patch.object(image, "go", lambda *x, **k: next(answers)), \
                mock.patch.object(image, "ask", lambda *x, **k: "n"), \
                mock.patch.object(sys.stdin, "isatty", lambda: True), \
                mock.patch.object(image.setup_env, "installed", lambda: False), \
                mock.patch.object(image.setup_env, "install_comfyui", lambda **k: calls.setdefault("install", k)), \
                mock.patch.object(image.setup_env, "ensure_files", lambda *x, **k: calls.setdefault("files", x)):
            image.setup_command(a)
        self.assertIn("install", calls)
        self.assertNotIn("files", calls)

    def test_setup_asks_for_the_models_folder_and_remembers_it(self):
        import argparse
        import importlib.util
        from unittest import mock
        spec = importlib.util.spec_from_file_location("easy_image_cli3", ROOT / "src" / "easy-image.py")
        image = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(image)
        mine = self.tmp.name + "/my models"
        replies = iter([mine, "", ""])   # the folder; Enter = download everything; quality: Enter
        calls, cfg_path = {}, pathlib.Path(self.tmp.name) / "config.json"
        a = argparse.Namespace(setup=[], yes=False, comfy=None, models_dir=None, dry_run=False)
        with mock.patch.object(image, "ask", lambda *x, **k: next(replies)), mock.patch.object(image, "go", lambda *x, **k: True), \
                mock.patch.object(image, "CONFIG", cfg_path), \
                mock.patch.object(sys.stdin, "isatty", lambda: True), \
                mock.patch.object(image.setup_env, "installed", lambda: True), \
                mock.patch.object(image.setup_env, "memory_gb", lambda: 8.0), \
                mock.patch.object(image.setup_env, "ensure_files", lambda ids, mdir, *x, **k: calls.setdefault("mdir", str(mdir))):
            image.setup_command(a)
        self.assertTrue(calls["mdir"].endswith("my models"))
        saved = json.loads(cfg_path.read_text())
        self.assertEqual(saved["models_dir"], mine)
        self.assertEqual(saved["tier"], "smallest")   # the mocked card has 8 GB: Enter takes the smallest tier
        self.assertEqual(self.setup_env.vram_gb.__name__, "vram_gb")

    def test_vram_is_read_from_nvidia_smi_text(self):
        from unittest import mock
        with mock.patch.object(self.setup_env, "nvidia_gpu", lambda: "NVIDIA GeForce RTX 5080, 16303 MiB"):
            self.assertAlmostEqual(self.setup_env.vram_gb(), 15.9, delta=0.1)
        with mock.patch.object(self.setup_env, "nvidia_gpu", lambda: None):
            self.assertIsNone(self.setup_env.vram_gb())

    def test_missing_files_name_the_catalog_files_that_fix_them(self):
        fake = FakeComfy(drop_files={"krea2_turbo_fp8_scaled.safetensors", "sample_krea2_finetune.safetensors",
                                     "qwen3vl_4b_fp8_scaled.safetensors", "qwen3vl_4b_bf16.safetensors"})
        try:
            eng = Engine(Comfy(fake.url), None)
            with self.assertRaises(Missing) as c:
                eng.plan(Job(prompt="x", engine="krea"))
            self.assertEqual(sorted(c.exception.ids), ["krea2-clip", "krea2-unet"])
            fake.drop_files.clear()
            fake.drop_files.add("krea2\\krea2_style_reference.safetensors")
            eng.comfy.refresh()
            with self.assertRaises(Missing) as c:
                eng.plan(Job(prompt="x", styles=[PNG]))
            self.assertEqual(c.exception.ids, ["krea2-style"])
        finally:
            fake.stop()

    def test_managed_comfyui_is_started_used_and_stopped_again(self):
        """A stand-in ComfyUI folder: its main.py serves the fake API on the port it is given."""
        comfy_dir = pathlib.Path(self.tmp.name) / "comfyui"
        (comfy_dir / ".venv" / "bin").mkdir(parents=True)
        (comfy_dir / ".venv" / "Scripts").mkdir(parents=True)
        for rel_ in ("bin/python", "Scripts/python.exe"):
            (comfy_dir / ".venv" / rel_).symlink_to(sys.executable)
        (comfy_dir / ".image-ready").write_text("ok")
        (comfy_dir / "main.py").write_text(
            "import sys, time\nsys.path.insert(0, %r)\nfrom fake_comfy import FakeComfy\n"
            "port = int(sys.argv[sys.argv.index('--port') + 1])\nFakeComfy(port=port)\ntime.sleep(600)\n" % str(HERE))
        import socket
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        env = {**os.environ, "IMAGE_COMFY_DIR": str(comfy_dir), "IMAGE_COMFY_PORT": str(port), "IMAGE_OUT_DIR": self.tmp.name + "/out",
               "IMAGE_CACHE_DIR": self.tmp.name + "/cache", "IMAGE_OLLAMA_URL": "http://127.0.0.1:9",
               "IMAGE_CONFIG": self.tmp.name + "/config.json", "IMAGE_LIBRARY_DIR": self.tmp.name + "/library"}
        env.pop("IMAGE_COMFY_URL", None)
        r = subprocess.run([sys.executable, str(ROOT / "src" / "easy-image.py"), "--prompt", "a fox", "--no-ollama", "-o", self.tmp.name + "/fox.png"],
                           capture_output=True, text=True, env=env, cwd=self.tmp.name, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual(pathlib.Path(self.tmp.name + "/fox.png").read_bytes(), PNG)
        self.assertFalse(Comfy(f"http://127.0.0.1:{port}").alive(1))   # stopped again when the tool ended

    def test_a_missing_install_asks_instead_of_installing(self):
        env = {**os.environ, "IMAGE_COMFY_DIR": self.tmp.name + "/none", "IMAGE_CACHE_DIR": self.tmp.name + "/cache", "IMAGE_OLLAMA_URL": "http://127.0.0.1:9",
               "IMAGE_CONFIG": self.tmp.name + "/config.json", "IMAGE_LIBRARY_DIR": self.tmp.name + "/library"}
        env.pop("IMAGE_COMFY_URL", None)
        r = subprocess.run([sys.executable, str(ROOT / "src" / "easy-image.py"), "--prompt", "x", "--no-ollama"], capture_output=True, text=True,
                           env=env, cwd=self.tmp.name, timeout=60, stdin=subprocess.DEVNULL)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("-y", r.stderr)
        self.assertFalse((pathlib.Path(self.tmp.name) / "none").exists())


class Service(unittest.TestCase):
    def setUp(self):
        import server
        self.fake = FakeComfy()
        self.eng = Engine(Comfy(self.fake.url), None)
        s = socket_free_port()
        self.port = s
        self.t = threading.Thread(target=server.serve, args=(self.eng, "127.0.0.1", s, lambda *a: None), daemon=True)
        self.t.start()
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{s}/health", timeout=1).read()
                break
            except OSError:
                threading.Event().wait(0.1)

    def tearDown(self):
        self.fake.stop()

    def call(self, path, body=None):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=30)
            return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read(), dict(e.headers)

    def test_health_engines_loras(self):
        self.assertEqual(json.loads(self.call("/health")[1])["state"], "idle")
        self.assertTrue(json.loads(self.call("/engines")[1])["krea2"]["ok"])
        self.assertIn("krea2\\krea2_darkbrush.safetensors", json.loads(self.call("/loras")[1]))

    def test_generate_returns_png_and_info(self):
        code, data, headers = self.call("/generate", {"prompt": "a fox", "seed": 4, "loras": [{"name": "darkbrush", "strength": 0.5}]})
        self.assertEqual(code, 200)
        self.assertEqual(data, PNG)
        self.assertEqual(json.loads(headers["X-Image-Info"])["seed"], 4)
        self.assertEqual(node(self.fake.graphs[0], "LoraLoaderModelOnly")[0]["inputs"]["strength_model"], 0.5)

    def test_generate_with_images_as_base64(self):
        b = "data:image/png;base64," + base64.b64encode(PNG).decode()
        code, data, _ = self.call("/generate", {"prompt": "make it red", "edit": b})
        self.assertEqual(code, 200)
        self.assertEqual(len(self.fake.uploads), 1)

    def test_saved_characters_by_name(self):
        import library
        tmp = tempfile.TemporaryDirectory()
        os.environ["IMAGE_LIBRARY_DIR"] = tmp.name
        try:
            library.save("character", "me", PNG)
            self.assertEqual(json.loads(self.call("/characters")[1]), ["me"])
            code, data, _ = self.call("/generate", {"prompt": "on the moon", "character_names": ["me"]})
            self.assertEqual(code, 200)
            self.assertEqual(len(self.fake.uploads), 1)
            code, data, _ = self.call("/generate", {"prompt": "x", "character_names": ["nobody"]})
            self.assertEqual(code, 400)
            self.assertIn("no saved character", json.loads(data)["error"])
        finally:
            os.environ.pop("IMAGE_LIBRARY_DIR", None)
            tmp.cleanup()

    def test_bad_requests(self):
        self.assertEqual(self.call("/generate", {"prompt": ""})[0], 400)
        self.assertEqual(self.call("/generate", {"prompt": "x", "loras": [["nothing-like-it", 1]]})[0], 400)
        self.assertEqual(self.call("/nope")[0], 404)

    def test_missing_model_is_a_400_with_the_reason(self):
        self.fake.drop_files.update({"krea2_turbo_fp8_scaled.safetensors", "sample_krea2_finetune.safetensors"})
        code, data, _ = self.call("/generate", {"prompt": "x", "engine": "krea"})
        self.assertEqual(code, 400)
        self.assertIn("models/diffusion_models", json.loads(data)["error"])


class ServiceWatch(unittest.TestCase):
    """Idle release and the hang watchdog, with a stand-in for the managed ComfyUI."""

    class FakeManaged:
        def __init__(self):
            self.proc, self.stops, self.starts = object(), 0, 0

        def stop(self):
            self.proc = None
            self.stops += 1

        def start(self, out=None):
            if self.proc is None:
                self.proc = object()
                self.starts += 1

    def setUp(self):
        import server
        self.server = server
        self.fake = FakeComfy()
        self.eng = Engine(Comfy(self.fake.url), None)
        self.managed = self.FakeManaged()
        self.port = socket_free_port()

    def tearDown(self):
        self.fake.stop()

    def start(self, **kw):
        threading.Thread(target=self.server.serve, args=(self.eng, "127.0.0.1", self.port, lambda *a: None),
                         kwargs=dict(managed=self.managed, **kw), daemon=True).start()
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1).read()
                return
            except OSError:
                threading.Event().wait(0.1)

    def post(self, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=30).status
        except urllib.error.HTTPError as e:
            return e.code

    def test_the_graphics_card_is_freed_when_quiet_and_comfyui_comes_back_on_request(self):
        self.start(idle=1, hard=0)
        self.server.STATE["last"] = __import__("time").time()
        self.assertEqual(self.post({"prompt": "x"}), 200)
        for _ in range(60):
            if self.managed.stops:
                break
            threading.Event().wait(0.1)
        self.assertEqual(self.managed.stops, 1)           # stopped after the quiet time
        self.assertEqual(self.post({"prompt": "y"}), 200)  # the next request starts it again
        self.assertEqual(self.managed.starts, 1)

    def test_a_hanging_job_is_cancelled_and_comfyui_restarted(self):
        from comfy import Cancelled
        def hang(job, status):
            for _ in range(200):
                if self.eng.comfy.cancel.is_set():
                    raise Cancelled()
                threading.Event().wait(0.05)
            return PNG, {}
        self.eng.generate = hang
        self.start(idle=0, hard=1)
        self.assertEqual(self.post({"prompt": "x"}), 499)
        self.assertEqual(self.managed.stops, 1)
        self.assertEqual(self.managed.proc, None)   # ComfyUI stays down until the next request starts it again


def socket_free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


if __name__ == "__main__":
    unittest.main()
