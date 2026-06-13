# CLI VLM verifier prompt — one shot per cell

Paste the prompt below as the FIRST message / `--prompt` argument to any
command-line VLM, then attach the cell's 4 PNGs + 1 text file as inputs.
The model returns one YAML verdict per cell; pipe to `vlm_responses/cell_NNN.yaml`.

## Prompt

```text
You are a pathology selective verifier on multiplexed cyclic-IF colorectal-cancer
imagery from the Orion platform. For each cell I will show you 4 PNGs and one
text file. Return ONE YAML verdict for the cell — nothing else.

INPUTS (4 images + 1 text file per cell)
1. cell_NNN_channels.png  4x3 grid (1024x768 px) of 12 marker tiles. EVERY tile
   shows the named marker tinted in its colour overlaid on top of the DAPI /
   Hoechst nuclear stain in dim blue, except the DAPI tile (top-left) which
   shows DAPI alone in full-strength blue. Tile palette:
     CD45 red    PanCK green   CD3e yellow  CD68 magenta  CD20 cyan
     CD4 orange  CD8 violet    FOXP3 pink   CD163 brown   aSMA pale-green
     CD31 light-blue
   Cell of interest = solid white outline. Neighbours = faint grey outlines.
2. cell_NNN_composite.png  512x512 RGB lineage composite (R=CD45, G=PanCK,
   B=DAPI). Red blobs = immune, green = epithelial, blue = nuclei.
3. cell_NNN_tcell.png      512x512 RGB T-cell composite (R=CD3e, G=CD4, B=CD8).
   Disambiguates CD4 vs CD8.
4. cell_NNN_context.png    768x768 zoom-out (~82 um field) lineage RGB. Use to
   read tissue context: epithelial nest vs immune aggregate vs stroma.
5. cell_NNN_blind.txt      text — 16-marker GMM posteriors + candidate set S.

TASK
Return one verdict per cell:
  - a leaf cell-type label, OR
  - an internal ontology node (when leaf-level is not supported), OR
  - `abstain` (image+markers do not support any defensible verdict), OR
  - `artefact` (segmentation failure, doublet, debris, biologically impossible
    marker combination).

CONSTRAINT: your verdict must be either (a) a member of S, (b) an ancestor of
a member of S, or (c) `abstain` / `artefact`. You may NOT widen S.

VALID LEAF LABELS
CD4_T, CD8_T, Treg, B_cell, NK, DC, Neutrophil, Macrophage_CD163pos,
Macrophage_CD163neg, Endothelial, Epithelial, Fibroblast, CAF, Basal,
Myoepi, Plasma, APC_generic

VALID INTERNAL NODES
T_cell, B_lineage, Macrophage, Lymphoid, Myeloid, Immune

DECISION RULES
1. A bright marker spot OVERLAPPING a blue nucleus = real positive AT the
   cell of interest. A bright spot AWAY from any nucleus = staining
   noise / fluorescence artefact, NOT a real positive cell.
2. CD4+ AND CD8+ simultaneously, OR CD45+ AND PanCK+ AND CD31+ simultaneously,
   are biologically impossible -> call `artefact`.
3. Two nuclei inside the white outline = doublet -> `artefact`.
4. Ambiguity between sibling leaves (e.g. CD4_T vs CD8_T) -> back off to
   the parent (`T_cell`). Do not guess.
5. `abstain` is a valid, valuable answer. Do not guess if the evidence is
   genuinely thin.

OUTPUT FORMAT (strict YAML, no surrounding prose, no markdown fences)
verdict: <one of the labels above>
confidence: <high|medium|low>
evidence:
  markers: <one-line summary of which markers drove the call>
  morphology: <one-line summary of what the image shows>
segmentation_quality: <good|questionable|poor>
reasoning: <2 sentences max>
```

## Loop wrappers per CLI

### Gemini CLI

```bash
mkdir -p vlm_responses
PROMPT=$(awk '/^```text$/{f=1;next} /^```$/{f=0} f' scripts/vlm_audit/cli_one_shot_prompt.md)
for n in $(seq -w 1 100); do
    cell="cell_${n}"
    out="vlm_responses/${cell}.yaml"
    [ -f "$out" ] && continue   # resume / skip done
    BLIND=$(cat vlm_100cells/${cell}_blind.txt)
    gemini --model gemini-2.5-pro \
      --prompt "${PROMPT}\n\n--- BLIND TEXT FOR ${cell} ---\n${BLIND}" \
      vlm_100cells/${cell}_channels.png \
      vlm_100cells/${cell}_composite.png \
      vlm_100cells/${cell}_tcell.png \
      vlm_100cells/${cell}_context.png \
      > "$out"
    sleep 1   # rate limit
done
```

(Gemini CLI's exact image-attach flag varies by version — `gemini --help` will
show whether you pass paths positionally, with `--image`, or via `@path`
inside the prompt. The block above shows positional.)

### Claude API via `curl` (one cell per call)

```bash
mkdir -p vlm_responses
PROMPT=$(awk '/^```text$/{f=1;next} /^```$/{f=0} f' scripts/vlm_audit/cli_one_shot_prompt.md)
for n in $(seq -w 1 100); do
    cell="cell_${n}"
    out="vlm_responses/${cell}.yaml"
    [ -f "$out" ] && continue
    BLIND=$(cat vlm_100cells/${cell}_blind.txt)
    # Build messages JSON with 4 base64 images + text
    python3 scripts/vlm_audit/run_audit.py --cell "$n" --prompt-file scripts/vlm_audit/cli_one_shot_prompt.md > "$out"
done
```

(`run_audit.py` already does this loop with the Anthropic SDK — see its
`--help` for model / temperature flags.)

### OpenAI / ChatGPT CLI (e.g., `chatgpt`)

```bash
mkdir -p vlm_responses
PROMPT=$(awk '/^```text$/{f=1;next} /^```$/{f=0} f' scripts/vlm_audit/cli_one_shot_prompt.md)
for n in $(seq -w 1 100); do
    cell="cell_${n}"
    out="vlm_responses/${cell}.yaml"
    [ -f "$out" ] && continue
    BLIND=$(cat vlm_100cells/${cell}_blind.txt)
    chatgpt --model gpt-4o \
      --system "${PROMPT}" \
      --image vlm_100cells/${cell}_channels.png \
      --image vlm_100cells/${cell}_composite.png \
      --image vlm_100cells/${cell}_tcell.png \
      --image vlm_100cells/${cell}_context.png \
      --user "BLIND TEXT FOR ${cell}: ${BLIND}\n\nReturn the YAML only." \
      > "$out"
done
```

## After the loop completes

```bash
ls vlm_responses/ | wc -l                  # should be 100
python scripts/vlm_audit/score_audit.py    # writes vlm_audit_results.json
cat vlm_audit_results.json | jq .
```
