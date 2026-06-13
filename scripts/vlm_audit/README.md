# VLM Audit Protocol (100 cells)

Inputs (the per-cell PNG/prompt bundle) are expected under `<repo>/vlm_100cells/`
(override with `--cells-dir`). They are NOT shipped in this release — they are
part of the Zenodo artifact bundle (see the top-level README "Data & model
artifacts"). The 100 cells are stratified 20 easy / 30 medium / 35 hard /
15 artifact. Each cell carries 3 PNGs (channel grid, RGB composite, T-cell
composite), a blind prompt (no GT), and an evaluation prompt (with GT).

This directory contains the audit code (runner + scorer + prompts).

## Files

| File | Purpose |
|---|---|
| `gemini_chatbox_instructions.md` | **Paste this at the start of a Gemini chat.** Tells Gemini the task, valid outputs, YAML schema, and shows 4 worked examples. |
| `suggested_30cell_sample.md` | If 100 is too many for a chatbox workflow, pick these 30 stratified cells instead. |
| `system_prompt.md` | Same content as `gemini_chatbox_instructions.md` but formatted for API use (used by `run_audit.py`). |
| `run_audit.py` | API batch runner — Gemini 2.5 Pro / Claude / GPT supported. |
| `score_audit.py` | Reads `vlm_responses/*.yaml`, computes the four numbers the paper needs. |

## Path A — Gemini chatbox (what you're doing)

1. Open a new Gemini 2.5 Pro conversation.
2. Paste the content of `gemini_chatbox_instructions.md` as the first
   message. Wait for Gemini to reply "READY".
3. For each cell N:
   - Upload `vlm_100cells/cell_NNN_channels.png`,
     `cell_NNN_composite.png`, `cell_NNN_tcell.png`.
   - Paste the content of `vlm_100cells/cell_NNN_blind.txt`.
   - Send.
   - Save Gemini's YAML reply to `vlm_responses/cell_NNN.yaml`.
4. When you are done, run the scorer:
   ```bash
   python scripts/vlm_audit/score_audit.py --responses vlm_responses/
   ```

Suggested: start with the 5-cell pilot (cells 001–005) to check Gemini's
output format is clean, then do the 30-cell stratified sample
(`suggested_30cell_sample.md`), then scale if time permits.

## Path B — API batch (for when you want automation)

```bash
export GOOGLE_API_KEY=...  # or ANTHROPIC_API_KEY / OPENAI_API_KEY
python scripts/vlm_audit/run_audit.py --model gemini-2.5-pro
python scripts/vlm_audit/score_audit.py --responses vlm_responses/ \
    --out vlm_audit_results.json
```

## Path C — Pathologist gold standard (for camera-ready)

Same protocol as A, but the pathologist is the "VLM". Their answers
become the gold-standard floor. You can then report both VLM accuracy
against conformal GT and VLM-pathologist agreement.

## What the paper gets

Four numbers per the scorer output:
- `singleton_precision_among_committed` → VLM §5.6 of agentic / §5.1.3 of genbio
- `abstention_rate` → supports the "abstain is a first-class action" claim
- `artefact_recall_on_artifact_tier` → validates the artefact detection
- `per_tier` accuracy breakdown → supports the stratified-audit claim

All four feed directly into the camera-ready prose. Point me at the saved
responses when you're done and I'll integrate the numbers into both
papers.
