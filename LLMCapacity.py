"""Detect how many requests the LLM server can process in parallel."""
import json
import re
import shutil
import subprocess
import time
import concurrent.futures
from typing import Callable, Optional

import requests

import LLMBackend

_KEY_RE = re.compile(r"parallel|concurren|n_slots|total_slots", re.IGNORECASE)


def _base_url(api_url: str) -> str:
    """'http://host:1234/v1/chat/completions' -> 'http://host:1234'"""
    if LLMBackend.is_ollama_url(api_url):
        return LLMBackend.ollama_base_url(api_url)
    return re.split(r"/v1(/|$)", api_url.rstrip("/"))[0]


def _find_int(obj, depth: int = 0) -> Optional[int]:
    """Recursively search a JSON structure for an int under a key that looks like a parallelism setting."""
    if depth > 6:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _KEY_RE.search(str(k)) and isinstance(v, int) and not isinstance(v, bool) and v > 0:
                return v
        for v in obj.values():
            found = _find_int(v, depth + 1)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_int(v, depth + 1)
            if found:
                return found
    return None


def read_declared_limit(api_url: str, model: str, log: Callable[[str], None] = lambda m: None) -> Optional[tuple[int, str]]:
    """Asks the server (and the LM Studio CLI) whether it declares a parallel-request limit."""
    if LLMBackend.is_ollama_url(api_url):
        return None  # Ollama's OLLAMA_NUM_PARALLEL is not exposed through its API -> measure instead
    base = _base_url(api_url)
    probes = [(f"{base}/api/v0/models/{model}", "LM Studio model info"), (f"{base}/props", "llama.cpp /props")]
    for url, label in probes:
        try:
            r = requests.get(url, timeout=5)
            if r.ok:
                n = _find_int(r.json())
                if n:
                    return n, label
        except (requests.RequestException, ValueError):
            pass

    lms = shutil.which("lms")
    if lms:
        try:
            out = subprocess.run([lms, "ps", "--json"], capture_output=True, text=True, timeout=10)
            for entry in json.loads(out.stdout or "[]"):
                names = {str(entry.get(k)) for k in ("identifier", "modelKey", "path", "id")}
                if model in names:
                    n = _find_int(entry)
                    if n:
                        return n, "lms ps"
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return None


def _timed_request(api_url: str, model: str, timeout: float) -> float:
    payload = LLMBackend.build_chat_payload(
        api_url, model, "Count from 1 to 40 separated by commas. Output only the numbers.",
        max_tokens=64, temperature=0)
    start = time.time()
    r = requests.post(api_url, json=payload, timeout=timeout)
    r.raise_for_status()
    return time.time() - start


def measure_parallel_limit(api_url: str, model: str, max_probe: int = 8, slowdown: float = 1.7,
                           log: Callable[[str], None] = lambda m: None, timeout: float = 300) -> int:
    """
    Empirically measures concurrency: fires n simultaneous identical requests and compares the wall time to a
    single request. Requests beyond the server's parallel limit are queued, so the wall time jumps to roughly
    2x. Returns the highest n that still ran (almost) in parallel, capped at `max_probe`.
    """
    log("Warming up model (this loads it if necessary)...")
    _timed_request(api_url, model, timeout)  # warm-up / JIT load
    t1 = _timed_request(api_url, model, timeout)
    log(f"Single request: {t1:.2f}s")

    limit = 1
    for n in range(2, max_probe + 1):
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
            list(ex.map(lambda _: _timed_request(api_url, model, timeout), range(n)))
        total = time.time() - start
        log(f"{n} parallel requests: {total:.2f}s ({total / t1:.2f}x single)")
        if total > t1 * slowdown:
            break
        limit = n
    return limit


def detect_parallel_limit(api_url: str, model: str, max_probe: int = 8,
                          log: Callable[[str], None] = lambda m: None) -> tuple[int, str]:
    """Returns (limit, source). Raises requests exceptions if the server cannot be reached."""
    declared = read_declared_limit(api_url, model, log)
    if declared:
        log(f"Server declares {declared[0]} parallel request(s) ({declared[1]}).")
        return declared
    log("Server does not declare a limit - measuring it...")
    n = measure_parallel_limit(api_url, model, max_probe, log=log)
    return n, f"measured, tested up to {max_probe}"
