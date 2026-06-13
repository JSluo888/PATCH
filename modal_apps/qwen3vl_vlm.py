"""Modal app: Qwen3-VL OpenAI-compatible vLLM endpoint.

Stands up an OpenAI-compatible HTTP server (``/v1/chat/completions``) backed
by vLLM serving Qwen3-VL. The existing
``scripts/vlm_audit/run_audit.py`` adapter (``call_vllm``) hits this URL with
the standard ``openai`` Python client.

Why a separate module from ``modal_app/vlm_endpoint.py``?

- ``modal_app/vlm_endpoint.py`` exposes a Modal ``@modal.method()`` returning
  a JSON-schema-constrained *action* dict for the agentic loop. That is
  invoked via ``modal.Cls.from_name(...).act.remote(...)`` — not a public
  HTTP endpoint, and the schema disagrees with the YAML one-shot audit
  prompt (see ``scripts/vlm_audit/cli_one_shot_prompt.md``).
- This module instead runs ``vllm serve --host 0.0.0.0 --port 8000`` inside
  a Modal web server and exposes ``/v1`` publicly. That matches the spec
  (``docs/e2_modal_qwen3vl_spec.md`` §3 + §5) and the existing audit
  harness's OpenAI-style adapters.

Deploy (one-shot, by the user — do NOT run this from CI):

    modal deploy modal_apps/qwen3vl_vlm.py

Then export the returned public URL:

    export VLLM_BASE_URL="https://<workspace>--proteoranker-qwen3vl-serve.modal.run"
    python scripts/vlm_audit/run_audit.py \
        --model Qwen/Qwen3-VL-32B-Instruct-FP8 \
        --base-url "$VLLM_BASE_URL" \
        --limit 100

Cost note: A100-80GB on Modal (~$2-4/h). First deploy downloads ~32GB FP8
weights into the hf-cache volume (one-time, ~5-8 min). A 100-cell audit once
warm is ~15-20 min. Budget ~$1.5-3 end to end. TEAR DOWN after: ``modal app
stop proteoranker-qwen3vl`` (Modal scales to zero between requests, but stop to
be sure). Use the dense 8B FALLBACK_MODEL on A10 if cost must be minimized.
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Optional

import modal


# ----------------------------------------------------------------------
# App / image
# ----------------------------------------------------------------------

APP_NAME = "proteoranker-qwen3vl"

# Primary model: the STRONGEST Qwen3-VL that serves reliably on a single
# A100-80GB. The dense 32B is the best non-MoE Qwen3-VL; the FP8 checkpoint
# (~32GB weights) is effectively lossless for inference and leaves ~40GB free
# for KV cache + the 4-image multimodal vision buffers, so it cannot OOM the
# way the 30B-A3B MoE (60GB bf16) did. The 235B-A22B MoE would need 4-8 GPUs
# and is not cost-justified for a 100-cell verifier audit. ``MODEL_ID`` /
# ``FALLBACK_MODEL`` env vars override without editing this file.
DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct-FP8"
# Safe dense <=8B fallback for an A10/L4 cost floor (still a real Qwen3-VL).
DEFAULT_FALLBACK_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# Single shared GPU. Qwen3-VL-32B-Instruct-FP8 is ~32GB of weights, so it fits
# an A100-80GB with large headroom (KV cache + 4-image vision buffers). The
# earlier 30B-A3B MoE default loaded ALL ~60GB bf16 (MoE = sparse compute, NOT
# sparse memory) and OOM'd at the 0.90 util target — FP8-32B removes that
# failure mode entirely. Override via QWEN3VL_GPU; keep A100-80GB unless you
# also drop FALLBACK_MODEL to the dense 8B. A100-80GB bills per second on
# Modal; tear the app down after the audit.
GPU_KIND = os.environ.get("QWEN3VL_GPU", "A100-80GB")

# vLLM listens on this port inside the container; the Modal web_server
# decorator publishes it.
VLLM_PORT = 8000

# Hugging Face cache volume reused across cold starts to avoid re-downloading
# the ~60GB weights on every run. Volume is created on first deploy.
HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)


IMAGE = (
    modal.Image.from_registry(
        # CUDA 12.8 base to match the cu128 wheels shipped by vLLM >=0.11,
        # which is the first line that ships the ``qwen3_vl`` / FP8 kernels
        # Qwen3-VL needs. (The old 12.4 + ``vllm>=0.6`` combo predated the
        # Qwen3-VL architecture entirely — that is why the server never bound.)
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "curl")
    .pip_install(
        # >=0.11 ships qwen3_vl + qwen3_vl_moe support and FP8 W8A8 kernels.
        "vllm>=0.11.0",
        # >=4.57 carries the Qwen3-VL processor/config classes.
        "transformers>=4.57.0",
        "accelerate",
        # qwen-vl-utils handles the image-token interleaving + tiling that
        # Qwen3-VL's preprocessor expects.
        "qwen-vl-utils[decord]>=0.0.10",
        "pillow",
        "huggingface_hub[hf_transfer]",
        # Smoke test in the local entrypoint uses the OpenAI client to hit
        # the running server.
        "openai>=1.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Forward model-selection env vars from the deploy-time shell into the
# container image so ``_resolve_model_id()`` (which runs at container start)
# honours a cost-driven model swap without editing this file. ``modal deploy``
# does not auto-forward the deploying shell's env to the served container, so
# we bake the chosen values in here. No-op when the vars are unset (keeps the
# spec default ``Qwen/Qwen3-VL-30B-A3B-Instruct``).
_CONTAINER_ENV = {
    k: os.environ[k]
    for k in ("FALLBACK_MODEL", "MODEL_ID")
    if os.environ.get(k)
}
if _CONTAINER_ENV:
    IMAGE = IMAGE.env(_CONTAINER_ENV)


app = modal.App(name=APP_NAME, image=IMAGE)


# ----------------------------------------------------------------------
# Web server: vllm serve --port 8000 ... exposed under /v1
# ----------------------------------------------------------------------


def _resolve_model_id() -> str:
    """Resolve the model id at container start.

    ``FALLBACK_MODEL`` overrides ``MODEL_ID``; both default to the spec
    primary / fallback. We resolve at runtime (not at module import) so
    operators can flip via a Modal secret without redeploy.
    """
    fallback = os.environ.get("FALLBACK_MODEL")
    if fallback:
        return fallback
    return os.environ.get("MODEL_ID", DEFAULT_MODEL)


@app.function(
    gpu=GPU_KIND,
    timeout=3600,
    volumes={"/root/.cache/huggingface": HF_CACHE},
    # Keep the container warm for 10 minutes between requests so back-to-back
    # audit cells don't pay cold-start each time.
    scaledown_window=600,
    # One serving container is sufficient for the 100-cell audit; raise
    # ``max_containers`` if running the audit in parallel sweeps.
    max_containers=1,
)
@modal.concurrent(max_inputs=8)
# 1500s headroom: first cold start downloads ~32GB FP8 weights into the
# hf-cache volume AND compiles flashinfer FP8 kernels before the port binds.
@modal.web_server(port=VLLM_PORT, startup_timeout=1500)
def serve() -> None:
    """Launch ``vllm serve`` as a subprocess on port 8000.

    Modal's ``@modal.web_server`` decorator forwards public HTTPS requests
    to the in-container port; the standard ``openai`` client therefore
    talks to ``https://<modal-url>/v1/chat/completions`` without any
    extra glue.
    """
    model_id = _resolve_model_id()
    print(f"[qwen3vl_vlm] starting vllm serve for {model_id} on :{VLLM_PORT}")

    # ``vllm serve`` is the CLI entrypoint to the OpenAI-compatible server.
    # ``--limit-mm-per-prompt`` bounds the number of images vLLM will
    # accept; the audit prompt sends 4 PNGs per cell.
    cmd = [
        "vllm",
        "serve",
        model_id,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        # ``auto`` lets vLLM honour the checkpoint dtype: FP8 weights stay FP8
        # (W8A8) with a bf16 compute path. Forcing float16 here would either
        # error on the FP8 checkpoint or silently upcast and waste VRAM.
        "--dtype", "auto",
        "--kv-cache-dtype", "auto",
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.90",
        # vLLM >=0.22 parses this as JSON (the old ``image=4`` key=value form
        # was removed and now hard-errors at arg-parse, crashing serve before
        # any weights load). Must be a JSON object string.
        "--limit-mm-per-prompt", '{"image": 4}',
        "--trust-remote-code",
        "--served-model-name", model_id,
        # Disable CUDA graph compilation for a faster cold start; cost is
        # ~5-10% throughput which is acceptable for a 100-cell audit.
        "--enforce-eager",
    ]
    # ``subprocess.Popen`` (not ``run``) — the web_server decorator wants
    # the function to return promptly while the child keeps serving in the
    # background. Modal polls the port until ``startup_timeout`` elapses.
    subprocess.Popen(cmd)

    # Block forever so the container stays alive while vllm serves.
    # Modal's lifecycle is driven by the web port being reachable; this
    # sleep loop is just a defensive parent process so the container does
    # not exit before vllm binds.
    while True:
        time.sleep(60)


# ----------------------------------------------------------------------
# Local entrypoint: smoke test against the running endpoint
# ----------------------------------------------------------------------


@app.local_entrypoint()
def smoke(base_url: Optional[str] = None, model: Optional[str] = None) -> None:
    """Send a single 1-image chat completion to validate the endpoint.

    Usage::

        modal run modal_apps/qwen3vl_vlm.py::smoke \\
            --base-url https://...modal.run \\
            --model Qwen/Qwen3-VL-30B-A3B-Instruct

    This does NOT touch the audit dataset or any vlm_100cells files; it
    sends a synthetic 8x8 PNG and asks for a one-word response.
    """
    import base64
    import io

    from openai import OpenAI
    from PIL import Image

    url = base_url or os.environ.get("VLLM_BASE_URL")
    if not url:
        raise SystemExit(
            "smoke: pass --base-url or set VLLM_BASE_URL to the deployed "
            "Modal web endpoint (e.g. https://...modal.run)"
        )
    model_id = model or os.environ.get("MODEL_ID", DEFAULT_MODEL)

    # 8x8 magenta PNG — small enough to round-trip in <1KB.
    img = Image.new("RGB", (8, 8), color=(255, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    client = OpenAI(api_key="EMPTY", base_url=f"{url.rstrip('/')}/v1")
    resp = client.chat.completions.create(
        model=model_id,
        temperature=0.0,
        max_tokens=32,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "One word: what colour is this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
    )
    print("smoke ok:", resp.choices[0].message.content)
