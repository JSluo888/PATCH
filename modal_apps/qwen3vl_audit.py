"""In-container Qwen3-VL 100-cell audit (localhost, no public edge).

Why this exists
---------------
``modal_apps/qwen3vl_vlm.py`` exposes vLLM through Modal's public
``@web_server`` edge. That edge issues a 303 "still-warming" redirect during
the ~6 min FP8 cold start, which makes the ``openai`` client hang/redirect-loop
on the very first cell — every external probe pays a fresh cold start and the
``scaledown_window`` recycles the warm container between probes. Net effect:
the audit never lands a single completion from outside.

This module sidesteps the edge entirely: a SINGLE GPU function

  1. starts ``vllm serve`` on ``127.0.0.1:8000`` inside the container,
  2. waits for ``/v1/models`` on localhost (no redirects, no edge),
  3. loops all 100 cells calling ``http://localhost:8000`` directly, and
  4. returns ``{cell_id: response_text}`` to the local entrypoint, which
     writes the per-cell YAML exactly like ``scripts/vlm_audit/run_audit.py``.

One cold start, one container, 100 localhost calls. Cost = one warm A100 for
the duration of the audit (~15-25 min on Marlin-FP8). Tear down is automatic
when the function returns (the function is not a persistent server).

Run (the user, one-shot)::

    modal run modal_apps/qwen3vl_audit.py --limit 100

The model id matches the FP8 checkpoint loaded by ``qwen3vl_vlm.py`` so the
``--served-model-name`` and response filenames line up.
"""

from __future__ import annotations

import os
import subprocess
import time
import urllib.request
from pathlib import Path

import modal

APP_NAME = "proteoranker-qwen3vl-audit"
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-32B-Instruct-FP8")
GPU_KIND = os.environ.get("QWEN3VL_GPU", "A100-80GB")
VLLM_PORT = 8000

# Local data staged into the image (read at runtime inside the function).
LOCAL_CELLS = Path(__file__).resolve().parents[1] / "vlm_100cells"
LOCAL_PROMPT = Path(__file__).resolve().parents[1] / "scripts/vlm_audit/cli_one_shot_prompt.md"
REMOTE_CELLS = "/data/cells"
REMOTE_PROMPT = "/data/prompt.md"

HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)

# IMPORTANT: keep this image spec byte-identical to qwen3vl_vlm.py so Modal
# reuses the already-built layer (content-hashed) instead of rebuilding.
IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "curl")
    .pip_install(
        "vllm>=0.11.0",
        "transformers>=4.57.0",
        "accelerate",
        "qwen-vl-utils[decord]>=0.0.10",
        "pillow",
        "huggingface_hub[hf_transfer]",
        "openai>=1.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Stage the 100-cell inputs + the one-shot prompt into the image.
    .add_local_dir(str(LOCAL_CELLS), remote_path=REMOTE_CELLS)
    .add_local_file(str(LOCAL_PROMPT), remote_path=REMOTE_PROMPT)
)

app = modal.App(name=APP_NAME, image=IMAGE)


def _load_prompt(path: str) -> str:
    """Extract the fenced ```text block from the one-shot prompt md."""
    text = Path(path).read_text()
    in_block, block = False, []
    for line in text.splitlines():
        s = line.strip()
        if s == "```text":
            in_block, block = True, []
            continue
        if in_block and s == "```":
            return "\n".join(block).strip()
        if in_block:
            block.append(line)
    return text.strip()


def _wait_for_server(url: str, served_name: str, timeout_s: int = 900) -> None:
    """Block until vLLM's localhost ``/v1/models`` advertises the model."""
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=10) as r:
                body = r.read().decode()
                if served_name in body:
                    print(f"[audit] vLLM ready after "
                          f"{int(timeout_s - (deadline - time.time()))}s")
                    return
                last = body[:200]
        except Exception as e:  # connection refused while still loading
            last = str(e)
        time.sleep(5)
    raise RuntimeError(f"vLLM not ready in {timeout_s}s; last={last}")


@app.function(
    gpu=GPU_KIND,
    timeout=3600,
    volumes={"/root/.cache/huggingface": HF_CACHE},
)
def audit(limit: int = 100, start: int = 1, model_id: str = MODEL_ID) -> dict:
    """Start vLLM on localhost, run ``limit`` cells, return {cid: text}."""
    import base64
    from openai import OpenAI

    # 1) Launch vLLM serve on localhost (same flags as qwen3vl_vlm.py serve()).
    cmd = [
        "vllm", "serve", model_id,
        "--host", "127.0.0.1", "--port", str(VLLM_PORT),
        "--dtype", "auto", "--kv-cache-dtype", "auto",
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.90",
        "--limit-mm-per-prompt", '{"image": 4}',
        "--trust-remote-code",
        "--served-model-name", model_id,
        "--enforce-eager",
    ]
    print(f"[audit] launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    base = f"http://127.0.0.1:{VLLM_PORT}"
    try:
        _wait_for_server(base, model_id, timeout_s=900)

        client = OpenAI(api_key="EMPTY", base_url=f"{base}/v1", timeout=300.0)
        system = _load_prompt(REMOTE_PROMPT)
        cells = Path(REMOTE_CELLS)

        out: dict = {}
        ok = fail = 0
        for i in range(start, start + limit):
            cid = f"cell_{i:03d}"
            try:
                blind = (cells / f"{cid}_blind.txt").read_text()
                content = [{"type": "text", "text": blind}]
                for name in ("channels", "composite", "tcell", "context"):
                    b = (cells / f"{cid}_{name}.png").read_bytes()
                    b64 = base64.b64encode(b).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
                resp = client.chat.completions.create(
                    model=model_id,
                    temperature=0.0,
                    max_tokens=1024,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": content},
                    ],
                )
                out[cid] = resp.choices[0].message.content
                ok += 1
                print(f"[ok]   {cid}  ({ok} done)", flush=True)
            except Exception as e:  # noqa: BLE001 — record + continue
                out[cid] = f"__ERROR__: {e}"
                fail += 1
                print(f"[FAIL] {cid}: {e}", flush=True)
        print(f"[audit] done ok={ok} fail={fail}")
        return out
    finally:
        proc.terminate()


@app.local_entrypoint()
def main(limit: int = 100, start: int = 1, out: str = "vlm_responses_qwen3vl32b") -> None:
    """Drive the in-container audit and write per-cell YAML locally."""
    model_id = os.environ.get("MODEL_ID", MODEL_ID)
    tag = model_id.split("/")[-1]
    results = audit.remote(limit=limit, start=start, model_id=model_id)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for cid, text in sorted(results.items()):
        if isinstance(text, str) and text.startswith("__ERROR__"):
            (out_dir / f"{cid}_{tag}.err").write_text(text)
            fail += 1
        else:
            (out_dir / f"{cid}_{tag}.yaml").write_text(text)
            ok += 1
    print(f"=== wrote {ok} responses, {fail} errors -> {out_dir} ===")
