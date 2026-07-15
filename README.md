# Krea 2 Turbo on Apple Silicon (ComfyUI / MPS)

Run [Krea 2](https://huggingface.co/Comfy-Org/Krea-2) — the open-weights 12B DiT from Krea AI —
**fully locally on a Mac**. This repo is the working recipe plus every MPS-specific failure
mode we hit getting there, so you don't burn the hours we did.

Krea 2 is commercially licensed, which makes it one of the few open t2i models whose output
you can ship to clients.

## Examples

All rendered locally on an M-series Mac with the exact workflow in this repo — bf16, 26 steps, no upscaler, no post. Same seed per pair; left has the Realism Engine node bypassed (the shipped default), right has it enabled at 0.4.

| Realism Engine off (default) | Realism Engine 0.4 |
|---|---|
| ![cafe off](examples/demo_cafe_bare_832x1216.png) | ![cafe RE](examples/demo_cafe_realism-engine_832x1216.png) |
| ![carpenter off](examples/demo_carpenter_bare_1024.png) | ![carpenter RE](examples/demo_carpenter_realism-engine_1024.png) |
| ![street off](examples/demo_street_bare_896x1152.png) | ![street RE](examples/demo_street_realism-engine_896x1152.png) |
| ![lake off](examples/demo_lake_bare_1216x832.png) | ![lake RE](examples/demo_lake_realism-engine_1216x832.png) |

## The file

One workflow — `Krea2-Turbo_Mac.json`. The core Turbo recipe (UNet → TE → 26-step sampler),
plus a Realism Engine LoRA node that **ships bypassed** so the base graph runs on just the three
required models — nothing else to download before your first render. Turn the LoRA on for cleaner
skin: select the purple node and press **Ctrl+B** (or right-click → Set Mode → Always); the same
keys toggle it back off. (Never set its strength to 0 — see gotcha 2.)

It ships with two in-graph info panels: a **model download panel** (direct HF links with sizes
and install paths) and a **recipe/gotchas panel** — so you don't need this README open while
you work. 100% ComfyUI core nodes, no custom nodes.

## Requirements

- Apple Silicon Mac with **48 GB unified memory** (validated). The three models total ~35.5 GB
  (UNet 26.3 + TE 8.9 + VAE 0.25), but they don't all sit resident at once: ComfyUI loads the
  TE, encodes, unloads it, *then* loads the UNet to sample — so **peak is ~26 GB (UNet) + working
  set**, not 35.5. In practice 36 GB runs but swaps hard (we watched a render pin swap at 21/22 GB);
  48 GB is comfortable, ≤32 GB isn't realistic on the bf16 path. Reports welcome.
- ComfyUI (recent build — needs the `krea2` CLIP type)
- Models from [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2):
  - `diffusion_models/krea2_turbo_bf16.safetensors`
  - `text_encoders/qwen3vl_4b_bf16.safetensors`
  - `vae/qwen_image_vae.safetensors`

**bf16 is not a choice, it's the only thing that runs.** We tested every smaller checkpoint
in the repo on MPS (ComfyUI current as of July 2026):

| Checkpoint | Size | On Mac |
|---|---|---|
| `krea2_turbo_bf16` | 26.3 GB | ✅ works — this repo |
| `krea2_turbo_fp8_scaled` | 13.1 GB | ❌ `Float8_e4m3fn` unsupported on MPS |
| `krea2_turbo_int8_convrot` | 13.5 GB | ❌ `UNETLoader` errors (`int8_tensorwise` unsupported; ConvRot kernels are CUDA-only) |
| `krea2_turbo_mxfp8` / `nvfp4` | 13.5 / 7.7 GB | ❌ NVIDIA Blackwell formats |
| GGUF quants | — | ❌ ComfyUI-GGUF predates the Krea 2 architecture (this is the future low-RAM path once supported) |

Don't spend time "optimizing" — bf16 or nothing, for now.

## The locked recipe

| Setting | Value |
|---|---|
| UNet | `UNETLoader` → `krea2_turbo_bf16.safetensors`, dtype `default` |
| Text encoder | `CLIPLoader` → `qwen3vl_4b_bf16.safetensors`, **type = `krea2`** |
| VAE | `qwen_image_vae.safetensors` (a Flux VAE decodes to scrambled noise) |
| Sampler | **26 steps · cfg 1.0 · euler · simple · denoise 1.0** |
| Negative | `ConditioningZeroOut` of the positive — at cfg 1 negatives are dead, don't write them |
| Latent | 1024×1024, 896×1152, 832×1216, or 1216×832 |

**Steps — the single biggest realism lever.** Turbo *runs* at 8, but 8 reads soft and
plasticky. A step sweep put the sweet spot at **20–26** (26 is the default here); past 26
barely differs, so it's not worth the time. Drop to **8–16 for fast seed hunting**, then
re-render keepers at 26 — composition shifts a little across a big step change, so judge the
final, not the 8-step preview.

At cfg 1 the positive prompt is the *only* steering, so how you phrase it matters more than on
a CFG model:

- **Lead with a photographic framing**, e.g. `sharp high-detail realistic photograph, natural
  skin texture with fine pores, crisp focus, of …` or, for night/flash scenes, `realistic
  candid iphone flash photo at night, natural detailed skin, of …`. Do **not** use `amateur
  smartphone photo, unedited` — that tag actively *softens* and lowers fidelity (it's what
  made early tests look low-quality).
- Then short comma-phrases: `[pose/action], [expression], [clothing], [setting], [lighting]`.
  State the expression explicitly or you get a neutral house-face.
- **Pin ethnicity/age** — anything you leave unstated drifts when you change steps, seed, or
  LoRA strength. `clear complexion, few freckles` cleans up freckle-heavy skin.

## MPS gotchas (the reason this repo exists)

These are Apple-Silicon backend bugs, not Krea 2 bugs — most will bite you on any large
DiT in ComfyUI on a Mac.

1. **`batch_size` must be 1** — but queue as many jobs as you like. These are different
   things: the `batch_size` widget (the batch dimension inside one render) is what breaks on
   MPS — set it to 4 and only *one* of the four denoises; the rest come out as pure static
   (tested: a batch of 4 gave three noise frames and one clean image). Large batches at full
   res also spike past 48 GB and OOM ComfyUI. Submitting many *separate* prompts to the queue,
   each at batch_size 1, is totally fine — that's how you render a sweep. Loop seeds, don't batch.
2. **Never set a LoRA to strength 0.0 — bypass the node instead.** A `LoraLoaderModelOnly` at
   0.0 NaNs the model on MPS and renders pure black. A zeroed patch is *not* a no-op here. To
   turn a LoRA off, right-click → Bypass (or Ctrl+B), which excludes it from the graph cleanly.
3. **First render is slow, then fine.** Cold-loading ~26 GB takes minutes; keep ComfyUI
   warm between renders. Run *without* `--disable-smart-memory` so the model stays resident
   across a seed hunt — but if the same instance must also load another large model, batch
   all Krea 2 work first; two big models co-resident will OOM.
4. **Cheap black/noise detection:** render a 512px probe and check mean luminance
   (PIL `ImageStat`) — mean < 8 = black output (gotcha 2), > 150 = noise (gotcha 1).
   Beats eyeballing full-res renders one by one.

## Optional add-on: Realism Engine LoRA

[Realism Engine for Krea 2](https://civitai.com/models/3109006) is a skin/texture LoRA that
takes the edge off Turbo's distillation look. Load it with **`LoraLoaderModelOnly`** — the
full `LoraLoader` also patches the text encoder, which re-interprets your prompt and shifts
the whole image. The node ships **bypassed** at strength **0.4** — the sweet spot for clean skin
without over-smoothing. Enable it with **Ctrl+B** (or right-click → Set Mode → Always) once the
LoRA file is in place; the same keys switch it back off. Never zero the strength (see gotcha 2).

**Identity drift warning:** at cfg 1 nothing anchors attributes you don't specify, and a LoRA
pulls every unspecified attribute toward its own training data. We watched a subject's
ethnicity flip between renders just from changing RE strength (0.8) or step count — same seed.
The fix isn't a magic strength value, it's the prompt: **state ethnicity/age explicitly** and
they hold across settings. Keep RE around 0.4; it cleans skin but doesn't fix freckles on its
own — add `clear complexion, few freckles` for that.

The same slot takes any Krea 2 model-only LoRA, including your own character LoRAs trained
on the undistilled Raw checkpoint (rank 32 / alpha 16 is a good starting point; disable
horizontal flip for asymmetric faces).

## Raw (undistilled)

`krea2_raw_bf16.safetensors` needs cfg ~4 and 25–30 steps. Use it as a LoRA-training base or
when the Turbo distillation look itself is the problem — for everything else Turbo is the
workhorse.
