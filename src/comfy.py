"""A small ComfyUI client using only the standard library: look up what is installed, upload images, run a graph, fetch results."""

import base64
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


class ComfyError(Exception):
    pass


class Cancelled(BaseException):
    """Raised when the running job is cancelled. Not an Exception on purpose, so a broad `except Exception` cannot swallow it."""


def norm(name):
    return name.replace("\\", "/")


def find_name(wanted, installed):
    """Match a file name against the names ComfyUI lists (which use the folder separator of its OS and may include subfolders).

    Tries the exact name, then the same name with other separators, then the bare file name. Returns the name as ComfyUI
    spells it, or None."""
    if not wanted:
        return None
    if wanted in installed:
        return wanted
    w = norm(wanted).lower()
    for n in installed:
        if norm(n).lower() == w:
            return n
    base = w.rsplit("/", 1)[-1]
    hits = [n for n in installed if norm(n).lower().rsplit("/", 1)[-1] == base]
    return hits[0] if len(hits) == 1 else None


def search_names(text, installed):
    """Candidates for a loosely typed name: exact/bare-name match first, then a case-insensitive substring."""
    exact = find_name(text, installed)
    if exact:
        return [exact]
    t = norm(text).lower()
    t_noext = t[:-len(".safetensors")] if t.endswith(".safetensors") else t
    bare = [n for n in installed if norm(n).lower().rsplit("/", 1)[-1].rsplit(".", 1)[0] == t_noext]
    if bare:
        return bare
    return [n for n in installed if t in norm(n).lower()]


# ---------- websocket (progress only) ----------

def _ws_open(base, client_id):
    u = urllib.parse.urlparse(base)
    host, port = u.hostname, u.port or (443 if u.scheme == "https" else 80)
    if u.scheme == "https":
        raise OSError("progress over https is not supported")
    s = socket.create_connection((host, port), timeout=10)
    try:
        key = base64.b64encode(os.urandom(16)).decode()
        s.sendall((f"GET /ws?clientId={client_id} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                   f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = s.recv(1024)
            if not chunk:
                raise OSError("websocket closed")
            buf += chunk
        head, rest = buf.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise OSError("websocket refused")
        s.settimeout(2)
        return s, rest
    except BaseException:
        s.close()
        raise


def _ws_listen(base, client_id, sink, stop):
    """Forwards ComfyUI's sampling progress to sink(step, steps). Progress is a nicety, so every failure is swallowed."""
    try:
        s, buf = _ws_open(base, client_id)
    except OSError:
        return

    def need(n):
        nonlocal buf
        while len(buf) < n:
            if stop.is_set():
                raise EOFError
            try:
                chunk = s.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                raise EOFError
            buf += chunk
    try:
        while not stop.is_set():
            need(2)
            op, ln, pos = buf[0] & 0x0F, buf[1] & 0x7F, 2
            if ln == 126:
                need(4)
                ln, pos = int.from_bytes(buf[2:4], "big"), 4
            elif ln == 127:
                need(10)
                ln, pos = int.from_bytes(buf[2:10], "big"), 10
            need(pos + ln)
            payload, buf = buf[pos:pos + ln], buf[pos + ln:]
            if op == 1:
                msg = json.loads(payload)
                if msg.get("type") == "progress":
                    d = msg.get("data") or {}
                    sink(d.get("value") or 0, d.get("max") or 0)
            elif op == 8:
                break
    except (EOFError, OSError, ValueError):
        pass
    finally:
        try:
            s.close()
        except OSError:
            pass


class Comfy:
    def __init__(self, base):
        self.base = base.rstrip("/")
        self._nodes = {}
        self.cancel = threading.Event()   # set to stop the job that is running now

    # ---- plain requests ----

    def _open(self, path, data=None, headers=None, timeout=30):
        req = urllib.request.Request(self.base + path, data=data, headers=headers or {})
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:2000]
            raise ComfyError(f"ComfyUI answered {e.code} for {path}: {body}") from None

    def get_json(self, path, timeout=30):
        with self._open(path, timeout=timeout) as r:
            return json.load(r)

    def alive(self, timeout=4):
        try:
            self.get_json("/system_stats", timeout=timeout)
            return True
        except (OSError, ComfyError, ValueError):
            return False

    # ---- what is installed ----

    def node(self, name):
        """The node's definition, or None when this ComfyUI does not have it."""
        if name not in self._nodes:
            try:
                info = self.get_json(f"/object_info/{name}", timeout=20)
            except (OSError, ComfyError, ValueError):
                info = {}
            self._nodes[name] = info.get(name)
        return self._nodes[name]

    def choices(self, node, field):
        """File names a loader node offers (empty when the node or the field does not exist)."""
        n = self.node(node)
        try:
            return list(n["input"]["required"][field][0])
        except (TypeError, KeyError, IndexError):
            return []

    def refresh(self):
        self._nodes.clear()

    # ---- running ----

    def upload(self, data, filename):
        """Uploads image bytes to ComfyUI's input folder; returns the stored name."""
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
                f"filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode()
        body += data + (f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
                        f"--{boundary}--\r\n").encode()
        with self._open("/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout=180) as r:
            return json.load(r)["name"]

    def _stop(self, pid):
        """Stops only this prompt: removes it from the queue and interrupts it if it is the one running. Never clears the queue."""
        for path, body in (("/queue", {"delete": [pid]}), ("/interrupt", {"prompt_id": pid})):
            try:
                self._open(path, json.dumps(body).encode(), {"Content-Type": "application/json"}, timeout=10).read()
            except (OSError, ComfyError):
                pass

    def run(self, graph, on_progress=None, timeout=900):
        """Queues a graph, waits for it and returns ([image bytes], seconds)."""
        if self.cancel.is_set():
            raise Cancelled()
        client_id, stop = uuid.uuid4().hex, threading.Event()
        if on_progress:
            threading.Thread(target=_ws_listen, args=(self.base, client_id, on_progress, stop), daemon=True).start()
        try:
            return self._run(graph, timeout, client_id)
        finally:
            stop.set()

    def _run(self, graph, timeout, client_id):
        body = json.dumps({"prompt": graph, "client_id": client_id}).encode()
        with self._open("/prompt", body, {"Content-Type": "application/json"}) as r:
            pid = json.load(r)["prompt_id"]
        start = time.time()
        try:
            while time.time() - start < timeout:
                if self.cancel.is_set():
                    self._stop(pid)
                    raise Cancelled()
                try:
                    h = self.get_json(f"/history/{pid}", timeout=90).get(pid)
                except (OSError, ComfyError):
                    time.sleep(3)   # ComfyUI can stall while it swaps large models; keep waiting
                    continue
                if h and h.get("status", {}).get("completed") is not None:
                    if h["status"]["status_str"] != "success":
                        errors = [m[1] for m in h["status"]["messages"] if m[0] == "execution_error"]
                        raise ComfyError(f"ComfyUI job failed: {errors}")
                    files = [img for out in h["outputs"].values() for img in out.get("images", [])]
                    data = []
                    for f in files:
                        with self._open("/view?" + urllib.parse.urlencode(f), timeout=90) as r:
                            data.append(r.read())
                    return data, time.time() - start
                time.sleep(1)
        except KeyboardInterrupt:
            self._stop(pid)
            raise
        self._stop(pid)
        raise TimeoutError(f"ComfyUI job {pid} did not finish in {timeout}s")
