# `modal_apps/` — Modal-hosted OSS VLM endpoints

This directory holds Modal apps that expose **OpenAI-compatible HTTP**
endpoints so the existing audit harness can drive OSS VLMs the same way it
drives Gemini / Anthropic / OpenAI.

> Sibling directory `modal_app/` (singular) hosts the agentic-loop
> `@modal.method()` endpoints that return JSON-schema actions. Don't
> confuse the two.

## Quick start: Qwen3-VL audit endpoint

```bash
# 1. One-time auth on the workstation (skip if you've already done it)
modal token new

# 2. Deploy. ~3-5 min build the first time; weights download on first request.
modal deploy modal_apps/qwen3vl_vlm.py

# 3. Modal prints a URL like:
#       https://<workspace>--proteoranker-qwen3vl-serve.modal.run
#    Export it so the audit harness picks it up:
export VLLM_BASE_URL="https://<workspace>--proteoranker-qwen3vl-serve.modal.run"

# 4. (Optional) Sanity-check the endpoint without touching audit data:
modal run modal_apps/qwen3vl_vlm.py::smoke --base-url "$VLLM_BASE_URL"

# 5. Run the 100-cell audit. --base-url defaults to $VLLM_BASE_URL.
python scripts/vlm_audit/run_audit.py \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --limit 100 \
    --out vlm_responses_qwen3vl \
    --prompt-file scripts/vlm_audit/cli_one_shot_prompt.md
```

## Cost

A10 list price is ~$0.60/hr on Modal. A 100-cell audit takes ~15–20 min
once warm (cold-start adds ~2–3 min on the first cell), so **expect
~$0.20 per 100 cells**.

## Switching to the fallback model

Per spec §1, if `Qwen3-VL-30B-A3B-Instruct` is unavailable, set
`FALLBACK_MODEL` in a Modal secret and re-deploy:

```bash
modal secret create qwen3vl-config FALLBACK_MODEL=Qwen/Qwen2-VL-32B-Instruct
modal deploy modal_apps/qwen3vl_vlm.py
```

Then point `--model` at the same id on the client side.
