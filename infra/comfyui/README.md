# ComfyUI workflows

Larger, LLM-directed ComfyUI graphs for this node, plus the tooling that keeps them in the
repo. ComfyUI serves `:8188` on GPU0 and writes to `/mnt/data/ltx-out`.

**It is an on-demand service — `sudo systemctl start comfyui` first, and stop it when done.**
It is not enabled at boot, because a resident model costs ~26 GB of host RAM on a 61 GB box
shared with the vLLM lanes.

## The contract: UI graph is the source, API JSON is generated

You author and refine graphs in the ComfyUI editor. `sync_workflows.py export` pulls the
flattened API-format version out of the server's run history and writes it here, where
scripts and services can drive it.

```bash
python sync_workflows.py list                       # what's exportable right now
python sync_workflows.py export "MiniMax H3 - Director"
python sync_workflows.py run workflows/h3-director.api.json
```

Export reads the server's **history**, not the saved UI file, for two reasons: the server has
already flattened subgraphs (the LTX-2.5 templates go 7 UI nodes → 42 API nodes), and only a
graph that actually ran can be exported. **History is in-memory and clears on restart** —
export after a run, not next week.

Going the other way needs no tool: the ComfyUI frontend detects API-format JSON on load
(`isApiJson`/`loadApiJson`), so **drag any `workflows/*.api.json` onto the canvas** to open it
as an editable graph, then save it under a name to make it the source.

## h3-director — LLM shot list → N MiniMax-H3 shots → one cut

```
idea ─→ ProtoStructured (protolabs/smart, guided JSON) ─→ shot list
                                                   │
        ┌──────────────────────────────────────────┴─────────────┐
        ▼ shot 1                    ▼ shot 2                     ▼ shot 3
   ProtoJSONGet prompt/frames  ...                          ...
        ▼
   MiniMaxH3ImageToVideo ─→ SamplerCustomAdvanced ─→ VAEDecode ─┬─→ SaveVideo (per shot)
                                                   VAEDecodeAudio
        └──────────────── ImageBatch / AudioConcat ─────────────┴─→ CreateVideo → SaveVideo
```

`build_h3_director.py` generates it. Regenerate rather than hand-editing the JSON:

```bash
python build_h3_director.py --idea "..." --shots 3 --lora turbo8 --steps 8 \
  > workflows/h3-director.api.json
python sync_workflows.py run workflows/h3-director.api.json
```

**Measured 2026-08-19** (first run, 3 shots, `--lora turbo8 --steps 8`, 864x480 @ 0.4 MP 16:9):
170 s wall for all three shots plus the cut. Shot lengths came back 175 / 124 / 226 frames and
the assembled video is exactly 525 frames — the LLM's per-shot choice reached H3's `length`
on-grid, with 32 kHz audio on every clip.

### Decisions worth knowing

* **The model picks `frames` from an enum, not a duration in seconds.** H3 snaps length to a
  17k+5 grid; asked for "6 seconds" a model returns an off-grid number the node silently
  rounds. The schema offers `[124, 175, 226, 277]` (5.2 / 7.3 / 9.4 / 11.5 s at 24 fps, inside
  the trained 124–362 range), so the arithmetic never reaches the model.
* **Consistency is restated, not referenced.** Each shot is generated independently and cannot
  see the others, so the director system prompt requires every shot to repeat the style clause
  and re-describe the subject in the same concrete words. "The same man as before" renders as a
  different man.
* **Every shot is saved on its own as well as in the cut.** When one shot of three is wrong you
  re-cut from the per-shot files instead of regenerating the whole piece.
* **Shots share the sampler and scheduler but not the noise seed.** Shared scheduler keeps step
  count consistent across the cut; separate seeds keep the shots off identical motion.
* **`ProtoJSONGet` degrades instead of failing.** A short shot list falls back to the node's
  default prompt, so the graph still renders and the Show Text nodes tell you what happened.

### Prompt guidance baked into the director

The system prompt encodes what `infra/video-bridge/PROMPTING.md` measured for LTX and what H3's
own shipped examples show: one flowing paragraph under 150 words, present progressive,
chronological; a style clause repeated verbatim; camera always stated including where it ends
up; sound woven alongside the action rather than appended; dialogue as exact quoted words with
the voice described; restrained concrete language, no quality tags. H3's documented weak spot is
audio with no speech or music, so most shots are pushed to carry something definite to hear.

### `--ref-chain` — identity across shots

Shot 1 is generated from words alone (`MiniMaxH3ImageToVideo`, fl2va). Its **last frame** is
extracted and handed to every later shot as a reference image
(`MiniMaxH3ReferenceToVideo`, ref2va), which the director's prompts address as `<Picture 1>`.
Restating a description keeps the *style* consistent; only a reference image keeps the same
*person* on screen across a cut.

```bash
python build_h3_director.py --shots 6 --ref-chain --lora none --steps 20 \
  --idea "..." > workflows/h3-director-refchain.api.json
```

Notes on the shape:

* **One checkpoint for the whole graph.** ref2va accepts zero reference images and is then a
  plain t2v path (verified 2026-08-19), so shot 1 runs on ref2va too rather than pairing it
  with fl2va. One model, one scheduler, no mid-graph swap.
* **`--lora` is refused with `--ref-chain`.** The turbo LoRAs are fl2v-trained and don't apply
  to ref2va; silently applying them to half the shots would make the cut inconsistent.
* The reference frame is saved alongside the video (`*_ref_*.png`) so you can see what
  identity the later shots were actually given.
* Autogrow inputs are addressed by their dotted API name — `ref_images.ref_image_0`.

### Knobs

```
--shots N          more/fewer branches; ids stay in readable per-shot blocks
--lora turbo8      8-step distilled (verified). turbo4 is 4-step but trained for 768p.
--lora none --steps 20   base quality; ~80 s/shot instead of ~35 s
--megapixels 0.4   0.2 = 608x352, 0.4 = 864x480, 1.0 ≈ 1280x704
--model            any gateway chat lane for the director (default protolabs/smart)
--ref-chain        shot 1 t2v, its last frame as reference for shots 2..N
```

## Housekeeping done 2026-08-19

* **16 dangling model symlinks removed.** All pointed at weights deleted in the 2026-08-11
  reclaim (and the 2026-05-03 qwen-image hardlink incident): LTX-2.3 dev/distilled and the
  x1.5/x2 upscalers, the whole qwen-image and anima sets, the ltx-2 gemma fp4 encoder and
  abliterated LoRA. Verified first that no source file exists anywhere under `/mnt/models` or
  `/mnt/data`, and that nothing referenced them except `QWEN-T2I.json`, itself already broken.
  The full link→target list is in `removed-symlinks-20260819.txt` so any can be recreated if
  the weights come back. **LTX-2.3 still works** — the live paths are the `models-cold` copies
  (`distilled-1.1`, `-fp4`) and `latent_upscale_models/`, which were untouched.
* **`comfyui.service` runs ON DEMAND — `inactive` + `disabled`.** It is deliberately not a
  boot service: it holds ~26 GB of host RAM while a model is resident, on a 61 GB box that
  also runs the vLLM lanes. Bring it up for a session and take it down after:

  ```bash
  sudo systemctl start comfyui     # ~10 s to serve on :8188
  sudo systemctl stop comfyui      # releases host RAM and GPU0
  ```

  (It was briefly enabled on 2026-08-19 and then reverted to on-demand.)
* **`--highvram` added to `comfyui.service`** after the kernel OOM-killed ComfyUI twice
  mid-render. Backup unit: `/etc/systemd/system/comfyui.service.pre-highvram-20260819-*`.

### Why `--highvram`, and what it is not

Both kills were the same size — **48.4 GB and 48.5 GB anon-RSS** — one on a graph with two
checkpoints, one on a graph with one. **Checkpoint count was not the driver.** ComfyUI's
default dynamic-VRAM mode stages weights in *pinned host RAM* (`Enabled pinned memory 55227`)
and climbs toward that budget during any render. That left ~3 GB of margin on a 61 GB box also
running the Qwen3.8 smart lane, `daria-lane`, and two embed servers, with swap fully consumed —
so whenever anything else allocated (vLLM's engine the first time, `cadvisor` the second) the
kernel killed the fattest process. GPU0 had ~35 GB of VRAM free throughout.

The hardware ratio is backwards for that default: **97 GB VRAM per card, ~74 GB free on GPU0,
against 61 GB of system RAM.** Dynamic VRAM staging is built for the opposite. `--highvram`
keeps models in VRAM and turns dynamic VRAM off (`is_dynamic_vram()` returns False when
`--highvram` is set).

Measured on the same single-shot render before and after:

```
                 peak host RSS   render   GPU0 used
default (dynamic)      ~48 GB      40 s     23.5 GB
--highvram             26.2 GB     45 s     65.8 GB
```

**Separately, `POST /free` between runs.** ComfyUI held **41.3 GB RSS after one render** and
does not release it on its own; `{"unload_models": true, "free_memory": true}` drops it to
4.5 GB. `sync_workflows.py run` issues it before every submit — pass `--no-free` to keep a
warm model when you are iterating.

## Related

* `infra/protolab-nodes/` — the `protoLab/*` nodes these graphs use (LLM, Structured, JSON,
  Prompt, TTS/STT). Symlinked into `ComfyUI/custom_nodes/`.
* `infra/video-bridge/` — OpenAI `/v1/videos` shim over ComfyUI; `PROMPTING.md` is the
  measured prompt/settings reference.
* `experiments/video-worldmodel/` — the H3 vs LTX-2.5 physics battery; source of the proven
  H3 node map used here.
