#!/usr/bin/env python3
"""Batch-run the 100-cell VLM audit.

Reads each cell's four PNGs + blind prompt text from vlm_100cells/, calls
the chosen VLM, and writes the response YAML to vlm_responses/cell_NNN.yaml.

Supported models:
  - gemini-2.5-pro, gemini-2.0-flash                 (GOOGLE_API_KEY)
  - claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5  (ANTHROPIC_API_KEY)
  - gpt-5.5, gpt-5, gpt-4o                             (OPENAI_API_KEY)
  - Qwen/Qwen3-VL-30B-A3B-Instruct, Qwen/Qwen2-VL-32B-Instruct, ...
        (any model name starting with "qwen", served via a vLLM
        OpenAI-compatible endpoint; pass --base-url or set VLLM_BASE_URL)

Usage:
    python scripts/vlm_audit/run_audit.py --model gemini-2.5-pro
    python scripts/vlm_audit/run_audit.py --model claude-opus-4-7 --limit 20
    python scripts/vlm_audit/run_audit.py --model gpt-5.5 --reasoning-effort xhigh --out vlm_resposes_gpt --resume
    python scripts/vlm_audit/run_audit.py --model Qwen/Qwen3-VL-30B-A3B-Instruct \\
        --base-url https://...modal.run --limit 100
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
CELLS_DIR = REPO / "vlm_100cells"     # default; overridable via --cells-dir
RESP_DIR = REPO / "vlm_responses"
DEFAULT_PROMPT_FILE = REPO / "scripts/vlm_audit/cli_one_shot_prompt.md"

# Maximum number of cells the audit will iterate over.  Bumped from 101
# to 501 when E3 was scoped — the 500-cell sample needs an inclusive
# upper bound on the range() (cells are 1-indexed).
MAX_CELLS = 501


def load_prompt(prompt_file: Path) -> str:
    """Read the fenced Prompt block from cli_one_shot_prompt.md when present."""
    text = prompt_file.read_text()
    lines = text.splitlines()
    in_block = False
    block: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "```text":
            in_block = True
            block = []
            continue
        if in_block and stripped == "```":
            return "\n".join(block).strip()
        if in_block:
            block.append(line)
    return text.strip()


def load_cell(i: int, cells_dir: Optional[Path] = None) -> Dict:
    """Load one cell's blind prompt + 4 PNGs.

    ``cells_dir`` overrides the legacy ``CELLS_DIR`` module-level default
    (``vlm_100cells``).  Set it to ``vlm_500cells`` to run E3.
    """
    base = Path(cells_dir) if cells_dir is not None else CELLS_DIR
    cid = f"cell_{i:03d}"
    blind = (base / f"{cid}_blind.txt").read_text()
    images = {
        "channels":  (base / f"{cid}_channels.png").read_bytes(),
        "composite": (base / f"{cid}_composite.png").read_bytes(),
        "tcell":     (base / f"{cid}_tcell.png").read_bytes(),
        "context":   (base / f"{cid}_context.png").read_bytes(),
    }
    return {"id": cid, "blind_text": blind, "images": images}


# ---------------------------------------------------------------------------
# VLM adapters. Each returns the raw text response given (system_prompt,
# user_text, image_bytes_dict).
# ---------------------------------------------------------------------------

def call_gemini(model: str, system: str, user_text: str, images: Dict[str, bytes]) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    mdl = genai.GenerativeModel(model, system_instruction=system)
    parts = [user_text]
    for name in ("channels", "composite", "tcell", "context"):
        parts.append({"mime_type": "image/png", "data": images[name]})
    resp = mdl.generate_content(parts, generation_config={"temperature": 0.0})
    return resp.text


def call_anthropic(model: str, system: str, user_text: str, images: Dict[str, bytes]) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    content = [
        {"type": "text", "text": user_text},
    ]
    for name in ("channels", "composite", "tcell", "context"):
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(images[name]).decode(),
            },
        })
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text


def call_openai(
    model: str,
    system: str,
    user_text: str,
    images: Dict[str, bytes],
    reasoning_effort: Optional[str] = None,
) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    content: List[Dict] = [{
        "type": "input_text",
        "text": f"BLIND TEXT:\n{user_text}\n\nReturn the YAML only.",
    }]
    for name in ("channels", "composite", "tcell", "context"):
        b64 = base64.b64encode(images[name]).decode()
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{b64}",
        })
    kwargs = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user",   "content": content},
        ],
        "max_output_tokens": 2048,
    }
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    resp = client.responses.create(
        **kwargs,
    )
    return resp.output_text


def call_vllm(
    model: str,
    system: str,
    user_text: str,
    images: Dict[str, bytes],
    base_url: str,
) -> str:
    """Call a vLLM OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Mirrors ``call_openai`` in structure but uses the plain
    ``chat.completions`` API (vLLM does not expose the Responses API). The
    server lives on Modal (see ``modal_apps/qwen3vl_vlm.py``); the audit
    harness only needs the base URL.

    Args:
        model: Model id as deployed on the server (e.g.
            ``Qwen/Qwen3-VL-30B-A3B-Instruct``). Forwarded verbatim — vLLM
            matches against ``--served-model-name``.
        system: System prompt text (see ``cli_one_shot_prompt.md``).
        user_text: Per-cell blind text (GMM posteriors + candidate set).
        images: Dict with keys ``channels``, ``composite``, ``tcell``,
            ``context`` -> raw PNG bytes.
        base_url: Endpoint base, e.g. ``https://...modal.run`` (no trailing
            ``/v1`` — we append it). ``api_key`` is unused by vLLM but the
            OpenAI client requires a non-empty string.
    """
    from openai import OpenAI

    client = OpenAI(api_key="EMPTY", base_url=f"{base_url.rstrip('/')}/v1")

    # Spec §4: text first, then 4 images in (channels, composite, tcell,
    # context) order — exploits Qwen3-VL's positional embeddings for
    # inter-modal coherence and matches the cognitive flow of the prompt.
    content: List[Dict] = [{"type": "text", "text": user_text}]
    for name in ("channels", "composite", "tcell", "context"):
        b64 = base64.b64encode(images[name]).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    )
    return resp.choices[0].message.content


def pick_adapter(
    model: str,
    reasoning_effort: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Callable:
    if model.startswith("gemini"):
        return lambda s, u, i: call_gemini(model, s, u, i)
    if model.startswith("claude"):
        return lambda s, u, i: call_anthropic(model, s, u, i)
    if model.startswith(("gpt-", "o4", "o3")):
        return lambda s, u, i: call_openai(model, s, u, i, reasoning_effort)
    if model.lower().startswith("qwen"):
        url = base_url or os.environ.get("VLLM_BASE_URL")
        if not url:
            raise ValueError(
                f"Model {model} requires a vLLM endpoint; pass --base-url "
                "or set VLLM_BASE_URL (see modal_apps/qwen3vl_vlm.py)."
            )
        return lambda s, u, i: call_vllm(model, s, u, i, url)
    raise ValueError(f"Unsupported model: {model}")


def required_api_key(model: str) -> Optional[str]:
    if model.startswith("gemini"):
        return "GOOGLE_API_KEY"
    if model.startswith("claude"):
        return "ANTHROPIC_API_KEY"
    if model.startswith(("gpt-", "o4", "o3")):
        return "OPENAI_API_KEY"
    # qwen* models talk to a vLLM endpoint; no provider API key needed.
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="gemini-2.5-pro | claude-opus-4-7 | gpt-5 | ...")
    ap.add_argument("--limit", type=int, default=100, help="Process first N cells only")
    ap.add_argument("--start", type=int, default=1, help="Start cell index (1-based)")
    ap.add_argument("--resume", action="store_true", help="Skip cells whose response file already exists")
    ap.add_argument("--out", default=str(RESP_DIR), help="Output directory for responses")
    ap.add_argument(
        "--cells-dir",
        default=str(CELLS_DIR),
        help=(
            "Directory holding cell_NNN_{blind.txt,channels.png,composite.png,"
            "tcell.png,context.png} files. Default: vlm_100cells.  Use "
            "vlm_500cells for the E3 expanded audit."
        ),
    )
    ap.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE), help="Markdown/text prompt file to use")
    ap.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        help="OpenAI Responses API reasoning effort, e.g. xhigh for GPT 5.5",
    )
    ap.add_argument(
        "--base-url",
        default=os.environ.get("VLLM_BASE_URL"),
        help=(
            "Base URL of a vLLM OpenAI-compatible server (used for qwen* "
            "models). Defaults to $VLLM_BASE_URL. No-op for "
            "gemini/claude/gpt models."
        ),
    )
    args = ap.parse_args()

    api_key = required_api_key(args.model)
    if api_key and not os.environ.get(api_key):
        raise SystemExit(f"Missing {api_key}; export it before running model {args.model}.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    system_prompt = load_prompt(Path(args.prompt_file))
    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort
    adapter = pick_adapter(args.model, reasoning_effort, args.base_url)

    # HF-style model ids contain "/" (e.g. "Qwen/Qwen3-VL-30B-A3B-Instruct")
    # which would create unintended subdirectories. Use the basename for the
    # response-filename tag while keeping the full id on the API call.
    model_tag = args.model.split("/")[-1]

    cells_dir = Path(args.cells_dir)

    ok = 0
    fail = 0
    for i in range(args.start, min(args.start + args.limit, MAX_CELLS)):
        outp = out / f"cell_{i:03d}_{model_tag}.yaml"
        if args.resume and outp.exists() and outp.stat().st_size > 10:
            print(f"[skip] {outp.name} already exists", flush=True)
            continue
        try:
            c = load_cell(i, cells_dir=cells_dir)
            resp = adapter(system_prompt, c["blind_text"], c["images"])
            outp.write_text(resp)
            print(f"[ok]   cell_{i:03d}", flush=True)
            ok += 1
        except Exception as e:
            print(f"[FAIL] cell_{i:03d}: {e}", flush=True, file=sys.stderr)
            (out / f"cell_{i:03d}_{model_tag}.err").write_text(str(e))
            fail += 1
        time.sleep(0.2)  # gentle rate limit

    print(f"\n=== Done. ok={ok}  fail={fail}  out={out}")


if __name__ == "__main__":
    main()
