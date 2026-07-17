# ComfyUI-BerniniR

A self-contained ComfyUI custom node package for **ByteDance's Bernini-R (1.3B / 14B)** video generation and editing model.

> **For developers and maintainers:** see [`README_DEVELOPER.md`](./README_DEVELOPER.md) for entry points, key functions, data flow, and common modification scenarios.

Bernini-R is a renderer-only diffusion model fine-tuned from **Wan2.1-T2V-1.3B**/**Wan 2.2-A14B**. It retains the T2V architecture but adds in-context conditioning, enabling a single checkpoint to perform both generation and editing tasks: text-to-video, image-to-video, video-to-video editing, reference-guided editing, and subject-to-video.

This package embeds model loading, VAE, temporal context windows, attention backends, TeaCache, NAG, a seven-mode guidance family, and dynamic guidance scheduling in one place.

---

## Table of Contents

- [What is Bernini-R?](#what-is-bernini-r)
- [Supported Tasks](#supported-tasks)
- [Features](#features)
- [Installation](#installation)
- [Required Models](#required-models)
- [Nodes Overview](#nodes-overview)
- [Guidance Modes](#guidance-modes)
- [Workflow Examples](#workflow-examples)
- [Segment Schedule](#segment-schedule)
- [Performance Tips](#performance-tips)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [License & Citation](#license--citation)
- [Acknowledgments](#acknowledgments)

---

## What is Bernini-R?

Bernini is a unified video generation and editing framework developed by ByteDance Research. The full pipeline combines:

- An **MLLM semantic planner** (Qwen 2.5-VL-7B) for understanding complex instructions.
- A **DiT renderer** based on Wan 2.2-A14B for pixel generation.

**Bernini-R** is the renderer-only variant. It drops the MLLM planner and operates as a fine-tuned Wan diffusion renderer with direct text and visual conditioning. This package supports both the **1.3B** and **14B** checkpoints. The 1.3B version is based on **Wan2.1-T2V-1.3B** / **Wan 2.2-A14B** and runs on consumer GPUs; the 14B version uses the larger Wan 2.2-A14B backbone. Both share the same architecture and load through the same nodes.

The 1.3B checkpoint handles style transfer, watermark removal, local edits, and reference-guided edits with quality close to the 14B variant. For complex human generation and multi-step reasoning, the 14B checkpoint provides a meaningful quality uplift.

> **Paper:** *Bernini: Latent Semantic Planning for Video Diffusion* (arXiv:2605.22344, 2026)  
> **Weights:** `ByteDance/Bernini-R-1.3B-Diffusers`/`ByteDance/Bernini-Diffusers`

---

## Supported Tasks

`BerniniR Prompt Embedding` exposes 12 task presets. The active behavior is determined by which visual inputs are connected to `BerniniR Conditioning`.

| Task preset | Inputs | Description |
|---|---|---|
| **Default / General** | any | Generic fallback system prompt. |
| **Text to Image** | prompt only | Single-image generation. |
| **Text to Video** | prompt only | Text-to-video generation. |
| **Image Editing** | prompt + `reference_images` | Edit a reference image according to the prompt. |
| **Subject-to-Image** | prompt + `reference_images` | Generate an image featuring the subject(s) in the reference images. |
| **Image-to-Video** | prompt + `reference_images` | Animate a reference image. |
| **Video Editing** | prompt + `source_video` | Text-guided video-to-video editing. |
| **Video Editing (Content Propagation)** | prompt + `source_video` | Edit while preserving/propagating content. |
| **Video Editing with Reference** | prompt + `source_video` + `reference_images` | Edit a video using reference images for style/subject. |
| **Ads / Content Insertion** | prompt + `source_video` + `reference_images` | Insert reference content into a source video. |
| **Video Editing (Action / Position)** | prompt + `source_video` | Motion/position edits. |
| **Video Editing (Style / Motion)** | prompt + `source_video` | Style or motion transfer edits. |

### How inputs map to tasks

| Input slot | Typical use |
|---|---|
| `source_video` | The video to be edited or transformed. |
| `reference_video` | A reference video for style/motion guidance. |
| `reference_images` | One or more reference images for subject, style, or image-to-video conditioning. |

All visual inputs are encoded into `context_latents` and attached to the conditioning, so the model can attend to them during denoising.

---

## Features

- **Lazy streaming model loading** — model loaders return lightweight handles at node execution; disk I/O, weight transfer to GPU, and LoRA merging are deferred until sampling starts. An LRU cache in the handle layer avoids reloading the same model on repeated runs, and warmup can pre-fetch weights before the first denoising step.
- **Unified generation & editing** — one checkpoint handles T2V, I2V, V2V, reference-guided editing, and more.
- **Seven-mode guidance family** — CFG, APG, RAAG, S2, Z2, STG_A, STG_R (see [Guidance Modes](#guidance-modes)).
- **Dynamic guidance strength schedule** — per-step guidance scale curves (cosine/linear/piecewise).
- **Segment prompt-travel** — split a video into prompt segments and denoise each through the context-window framework with linear crossfade at boundaries (see [Segment Schedule](#segment-schedule)).
- **Temporal context windows** — generate longer videos on limited VRAM via four window schedules (`static_standard`, `uniform_standard`, `uniform_looped`, `ordered_halving`) with overlap fusion.
- **Five attention backends** — auto-detect and switch between SageAttention 3, SageAttention, FlashAttention, xFormers, and SDPA.
- **TeaCache acceleration** — skip transformer blocks when latent change between steps falls below a threshold.
- **torch.compile support** — Windows-aware compile helper with SDK auto-detection and fallback handling.
- **Block swap** — GPU↔CPU transformer-block swapping for reduced VRAM footprint.
- **In-context conditioning** — encode `source_video`, `reference_video`, or one/multiple `reference_images` into `context_latents`.
- **NAG (Normalized Attention Guidance)** — inject negative-prompt attention paths to enhance detail.
- **Prompt Embedding** — task-aware prompt planning with 12 presets and text-embedding disk cache.
- **VAE tiling and color matching** — spatial tiling decode plus Reinhard, histogram, Monge-Kantorovich, and MVGD color transfer.

Together, lazy loading, context-window tiling, block swap, VAE chunked encoding, and CLIP offload form a layered VRAM management strategy. Context windows split long videos into overlapping temporal segments; block swap moves transformer blocks between GPU and CPU between segments; lazy loading avoids holding model weights in memory until they are needed; and VAE encode/decode uses spatial tiling to stay within budget. Each layer is independent and can be combined to match available GPU memory.

---

## Installation

### 1. Clone or copy the package

Place the folder inside your ComfyUI `custom_nodes` directory:

```text
ComfyUI/
└── custom_nodes/
    └── ComfyUI-BerniniR/
        ├── __init__.py
        ├── nodes/
        ├── models/
        └── ...
```

### 2. Install Python dependencies

Activate the ComfyUI virtual environment, then run:

```bash
# Core runtime dependencies (torch is already provided by ComfyUI — do not reinstall it)
pip install numpy>=1.21.0 einops tqdm

# Optional: attention backends (pick what works on your GPU)
pip install sageattention      
pip install flash-attn          # FlashAttention
pip install xformers            # xFormers backend
pip install kornia              # CIELAB path for Reinhard color matching
```

> **Note:** `requirements.txt` currently pins only the absolute minimum. Install the optional packages above for full functionality.

### 3. Restart ComfyUI

After restarting, you should see the message:

```text
[BerniniR] ComfyUI-BerniniR node package loaded successfully.
```

---

## Required Models

ComfyUI-BerniniR uses the standard ComfyUI model folders.

| Component | Folder | Typical Files |
|---|---|---|
| Diffusion model | `ComfyUI/models/diffusion_models/` | `Bernini-R-1.3B*.safetensors` (1.3B) / `Bernini-R-14B*.safetensors` (14B) |
| Text encoder | `ComfyUI/models/text_encoders/` | Wan T5-XXL `.safetensors` |
| VAE | `ComfyUI/models/vae/` | Wan 16-channel VAE `.safetensors` |

### Diffusion model

- **1.3B:** `ByteDance/Bernini-R-1.3B-Diffusers` converted to ComfyUI-compatible `.safetensors`. Based on **Wan2.1-T2V-1.3B**, single-expert (non-MoE) checkpoint.
- **14B:** The 14B Bernini-R variant also loads through the same nodes. Based on the larger Wan 2.2-A14B backbone.
- Standard Wan / Wan 2.1 diffusion models may also load, but I have been not testing it.

### Text encoder

- Wan T5-XXL text encoder (same as Wan 2.1).

### VAE

- Wan 2.1 VAE: 16 latent channels (4× temporal, 8×8 spatial compression).
- Wan 2.2 / WanI38B VAE with 48 latent channels is also supported internally.

---

## Nodes Overview

All 19 nodes appear under the **Bernini-R** category in ComfyUI.

### Loaders

| Node | Output | Description |
|---|---|---|
| `BerniniR Model Loader` | `MODEL` | Load a diffusion model. Supports `torch.compile` and attention-backend injection. |
| `BerniniR Compile Model` | `MODEL` | Apply `torch.compile` after LoRA loading. |
| `BerniniR CLIP Loader` | `CLIP` | Load the Wan T5 text encoder. |
| `BerniniR VAE Loader` | `VAE` | Load the Wan/Bernini-R VAE. |

### Conditioning & Prompts

| Node | Outputs | Description |
|---|---|---|
| `BerniniR Prompt Embedding` | `CONDITIONING`, `CONDITIONING`, `STRING`, `STRING` | Task-aware prompt planning with 12 presets; outputs positive, negative, system_prompt, and full_prompt, with disk cache and optional CLIP offload. |
| `BerniniR Conditioning` | `CONDITIONING`, `LATENT` | Create initial latents and encode `source_video`, `reference_video`, or `reference_images` as `context_latents`. Supports chunked VAE encoding for VRAM efficiency. |
| `BerniniR Apply NAG` | `CONDITIONING` | Inject normalized attention guidance into positive conditioning. |
| `BerniniR Segment Schedule` | `CONDITIONING`, `CONDITIONING` | Parse a prompt segment schedule and emit positive/negative conditioning with per-segment prompt embedding and crossfade ranges for segment prompt-travel. |

### Sampling & Control

| Node | Output | Description |
|---|---|---|
| `BerniniR KSampler` | `LATENT` | Enhanced sampler with native context-window, flow-shift, NAG, the seven-mode guidance family, and dynamic guidance support. |
| `BerniniR TeaCache Args` | `BERNINI_TEACACHE` | Configure TeaCache block skipping; connect its output to the `teacache_args` input of `BerniniR KSampler` to enable acceleration. Disconnect to disable. |
| `BerniniR Dual Expert Sampler` | `LATENT` | Switch between high-noise and low-noise model instances automatically. |
| `BerniniR Context Window` | `BERNINI_CTX` | Configure temporal context window schedule, overlap, and fusion method. |
| `BerniniR Guidance Strength Schedule` | `BERNINI_GUIDANCE` | Generate a per-step guidance scale curve. |
| `BerniniR Attention Config` | `BERNINI_ATTN` | Select the attention backend. |
| `BerniniR Guidance Config` | `BERNINI_GUIDANCE_CONFIG` | Typed guidance strategy plus hyper-parameters, consumed by the sampler. |

### VAE

| Node | Output | Description |
|---|---|---|
| `BerniniR VAE Decode` | `IMAGE` | Decode latents to images/video frames with optional tiling and color matching. |
| `BerniniR VAE Encode` | `LATENT` | Encode pixels to latents with tiling support. |

### LoRA & Memory

| Node | Output | Description |
|---|---|---|
| `BerniniR Load LoRA` | `MODEL` | Append a LoRA spec (inline merge) to the model handle. |
| `BerniniR Block Swap Args` | `BERNINI_BLOCKSWAP` | Configure GPU↔CPU transformer-block swapping. |

---

## Guidance Modes

The `BerniniR Guidance Config` node exposes a `guidance_mode` dropdown with **seven mutually exclusive strategies**. Exactly one strategy is active per run; the sampler routes the combine step accordingly.

| Mode | Forward passes | Description |
|---|---|---|
| `CFG` | 2 | Standard classifier-free guidance. Default. |
| `APG` | 2 | Adaptive Projected Guidance. |
| `RAAG` | 2 | Ratio-Aware Adaptive Guidance. |
| `S2` | 3 | Stochastic Self-Guidance (random sub-network repulsion). |
| `Z2` | 2 | Zero-Cost Zigzag Trajectories + trajectory-collapse stabilization. `z2_collapse` (0 = off) EMA-smooths the output velocity across steps. |
| `STG_A` | 3 | Spatiotemporal Skip Guidance, mode A (zero out self-attention output). |
| `STG_R` | 3 | Spatiotemporal Skip Guidance, mode R (skip self-attention residual). |

### Z2

`x0 = uncond + s·(cond − uncond)`, `z0 = uncond + s·(uncond − x0)`, then `v = uncond + s·(x0 − z0)`. Pure algebraic — no extra forward passes. When `z2_collapse > 0`, an EMA over the output velocity across denoising steps suppresses the per-step temporal jitter Z² introduces in video (the "trajectory collapse" stabilization).

Modes requiring 3 forward passes (`S2`, `STG_A`, `STG_R`) run an additional weak-model prediction per step and therefore cost more compute than the 2-pass modes.

### CFG

`D̃ = uncond + (cond − uncond) · cfg`

No additional parameters.

### APG

Adaptive Projected Guidance decomposes the CFG update into parallel and orthogonal components relative to the conditional prediction, then suppresses the parallel component (the main source of oversaturation and artifacts). This permits higher guidance scales without destroying color fidelity.

| Parameter | Default | Description |
|---|---|---|
| `apg_eta` | 0.15 | Attenuation of the parallel component. 0 = fully suppress, 1 = standard CFG. |
| `apg_rescale` | True | Rescale the APG result to the conditional prediction norm. |
| `apg_momentum` | 0.0 | Reverse momentum across steps. 0 disables. |

### RAAG

Ratio-Aware Adaptive Guidance adapts the guidance weight per step based on the divergence between conditional and unconditional predictions:

`ρ = ‖cond‖ / ‖uncond‖`

`w = 1 + (cfg_target − 1) · exp(−α · ρ)`

`D̃ = uncond + (cond − uncond) · w`

where `cfg_target` is the current step's guidance scale (from `Guidance Strength Schedule` when connected). Early steps with high divergence yield `w ≈ 1` (guidance effectively off); later steps converge to `cfg_target`.

| Parameter | Default | Description |
|---|---|---|
| `raag_alpha` | 1.0 | Decay rate. Range 0.1–10.0. |

### S2

Stochastic Self-Guidance constructs a weak model by randomly dropping transformer blocks per step, then repels the prediction away from it:

`D̃ = D_uncond + cfg · (D_cond − D_uncond) − ω · (D_sub − D_cond)`

| Parameter | Default | Description |
|---|---|---|
| `s2_omega` | 1.0 | Repulsion strength. Range 0.0–10.0. |

The dropped block set is sampled randomly per step (≈10% of blocks, excluding block 0) and is seed-reproducible.

### STG_A / STG_R

Spatiotemporal Skip Guidance uses a weak model obtained by skipping self-attention at specified blocks:

`D̃ = D_uncond + cfg · (D_cond − D_uncond) + stg_scale · (D_cond − D_skip)`

| Parameter | Default | Description |
|---|---|---|
| `stg_scale` | 1.0 | Guidance strength. Range 0.0–10.0. |
| `stg_block_idx` | `"10,20,27"` | Comma-separated block indices for self-attention skip. Wan 1.3B has 30 layers (0–29); the 14B variant has 40 layers (0–39). |

- **STG_A**: at marked blocks, the self-attention output is set to zero.
- **STG_R**: the self-attention residual is skipped while `x` is preserved unchanged.

### Recommended starting points

```text
# Balanced default
cfg: 6.0 - 8.0
guidance_mode: CFG

# Higher detail without oversaturation
cfg: 8.0 - 12.0
guidance_mode: APG
apg_eta: 0.15
apg_rescale: True
apg_momentum: 0.0
```

Guidance modes can be combined with `BerniniR Guidance Strength Schedule` for time-varying guidance scales.

---

## Workflow Examples

All examples share the same loader backbone:

```text
BerniniR Model Loader  ─┐
BerniniR CLIP Loader   ─┤
BerniniR VAE Loader    ─┤
                        ▼
               BerniniR Prompt Embedding
```

### 1. Text-to-Video

```text
Prompt Embedding (Text to Video) ──> CONDITIONING ──┐
Conditioning (no visual input) ──> LATENT        ──┤
Context Window ────────────────────> BERNINI_CTX  ──┤
Guidance Strength Schedule ────────> BERNINI_GUIDANCE ──┤
                                                       ▼
                                            BerniniR KSampler
                                                       │
                                                       ▼
                                             BerniniR VAE Decode
```

### 2. Image-to-Video

```text
Prompt Embedding (Image-to-Video) ──> CONDITIONING ──┐
Conditioning ──> LATENT                           ──┤
   └─ reference_images: [your image]                 │
Context Window ────> BERNINI_CTX                   ──┤
Guidance Strength Schedule ──> BERNINI_GUIDANCE    ──┤
                                                    ▼
                                         BerniniR KSampler
                                                    │
                                                    ▼
                                          BerniniR VAE Decode
```

### 3. Video-to-Video Editing

```text
Prompt Embedding (Video Editing) ──> CONDITIONING ──┐
Conditioning ──> LATENT                          ──┤
   ├─ source_video: [your video]                    │
   └─ mask (optional): [edit mask]                  │
Context Window ────> BERNINI_CTX                  ──┤
Guidance Strength Schedule ──> BERNINI_GUIDANCE   ──┤
                                                   ▼
                                        BerniniR KSampler
                                                   │
                                                   ▼
                                         BerniniR VAE Decode
```

> **Mask editing**: Connect an optional `mask` to `BerniniR Conditioning` to pin the background to `source_video` while only regenerating the masked region. White (`1`) = regenerate, black (`0`) = keep source — the same convention as ComfyUI `denoise_mask` and SAM2. `mask_mode="anneal"` (default) gives a softer WanVideoWrapper-style release with natural boundaries; `"freeze"` gives a hard pixel-level freeze.

### 4. Reference-Guided Video Editing

```text
Prompt Embedding (Video Editing with Reference)
   │
   ▼
Conditioning ──> LATENT
   ├─ source_video:      [video to edit]
   └─ reference_images:  [style/subject reference]
```

### Suggested starting parameters

| Parameter | Starter Value | Notes |
|---|---|---|
| Width / Height | 832×480 | Common Wan 2.1 / Bernini-R resolution. |
| Length | 81 | Latent frames (≈ 5 seconds at 16 fps). |
| Steps | 20-30 | Bernini-R typically needs 20+. Editing tasks may use fewer. |
| CFG / Guidance | 6-8 | Can be overridden by `BerniniR Guidance Strength Schedule`. |
| Flow shift | 3.0 | Wan 2.1 default. |
| Context frames | 81 | Lower this to reduce VRAM for long videos. |
| Context overlap | 16 | Higher overlap = smoother transitions. |

---

## Segment Schedule

`BerniniR Segment Schedule` emits positive/negative conditioning for prompt-travel across a video. It parses a schedule of the form `start-end: prompt` (segments separated by `;` or newline, frames 1-based) and encodes each segment's prompt, then attaches per-segment latent ranges and a crossfade overlap to the conditioning. The sampler builds a context-window wrapper from this data and swaps the positive prompt embedding per window, blending adjacent segments with a linear crossfade. This reuses the same context-window machinery as long-video generation rather than running independent denoising passes.

```text
BerniniR Segment Schedule ──> positive ──┐
                                        ├─> CONDITIONING ──> BerniniR KSampler
                              negative ──┘
```

- `total_frames`: total pixel frames of the output video.
- `transition_frames`: crossfade length at each segment boundary (latent frames = `transition_frames // 4`).
- `negative_prompt`: shared negative prompt for all segments.

---

## Performance Tips

### Reduce VRAM for long videos

1. Connect a `BerniniR Context Window` node to the sampler.
2. Lower `context_frames` (e.g., 41 or 25).
3. Increase `context_overlap` if you see seams.
4. Use `uniform_standard` for quality or `static_standard` for speed.
5. Optionally connect `BerniniR Block Swap Args` to offload transformer blocks to CPU.

### Speed up sampling

1. Add a `BerniniR TeaCache Args` node and connect its `teacache_args` output to `BerniniR KSampler`:
   - Start with `rel_l1_thresh=0.08`, `max_skip_blocks=15`, `start_block=3`.
2. Set an attention backend:
   - `sage3` or `sage` on RTX 30/40/50 series.
   - `flash` on Ampere/Hopper/Ada.
   - `xformers` or `sdpa` as fallback.
3. Enable `torch.compile` in the model loader:
   - On Windows, `reduce-overhead` and `max-autotune` are automatically downgraded to `default`.
   - Use `default` mode for the best Windows compatibility.
   - **TeaCache + torch.compile**: both can be enabled together. When a compiled
     model is also TeaCache-accelerated, TeaCache restores the eager transformer
     forward so its block-skipping hooks take effect (earlier versions could let
     compile silently disable TeaCache).

### Reduce host RAM / CLIP VRAM

In `BerniniR Prompt Embedding`:
- Set `force_offload=True` to move CLIP back to CPU after encoding.
- Set `use_disk_cache=True` to reuse embeddings across runs.

---

## Known Limitations

- **Renderer-only**: This package uses **Bernini-R**, the renderer-only variant. It does not include the 7B MLLM semantic planner from the full Bernini pipeline, so complex instruction decomposition and chain-of-thought reasoning are not available.
- **Model capacity**: The 1.3B checkpoint handles style transfer, watermark removal, local edits, and reference-guided edits. For complex human generation and multi-step reasoning, the 14B variant provides higher quality. Both are supported by this package.
- **Private types**: `BERNINI_CTX`, `BERNINI_GUIDANCE`, `BERNINI_ATTN`, and `BERNINI_BLOCKSWAP` are custom socket types. Mixing these nodes with native ComfyUI samplers requires adapters.
- **Windows torch.compile caveats**: `torch._dynamo.config.suppress_errors = True` is enabled, which hides compile failures and may silently fall back to eager mode.
---

## Troubleshooting

### ImportError / ModuleNotFoundError

Install the missing optional dependency:

```bash
pip install einops tqdm
# plus whichever attention backend you want:
pip install sageattention sageattn3 flash-attn xformers kornia
```

### Gray output or noisy frames

Make sure any `BerniniR Guidance Strength Schedule` node is connected correctly and uses the right curve. Guidance is applied in **noise-residual space**, not denoised space.

### Out of memory during sampling

- Lower `context_frames`.
- Use a smaller resolution.
- Set `force_offload=True` in the prompt planner.
- Connect `BerniniR Block Swap Args`.
- Try a lower attention backend.

### Out of memory during VAE decode

- Enable tiling in `BerniniR VAE Decode`.
- Reduce tile overlap.

### torch.compile fails silently or falls back

This is expected on Windows due to `suppress_errors=True`. Check the console for Inductor cache warnings, or disable compile and test eager-mode first.

If you change the model or suspect a stale compiled graph, purge the Inductor
cache. The cache is now **isolated** to `%TEMP%/bernini_r_inductor_cache`
(override with `BERNINI_COMPILE_CACHE_DIR`) and is auto-cleared only when the
code version changes. To force a purge:

- **In-workflow**: enable the `purge_cache` toggle on `BerniniR Compile Model`, or
- **Environment**: set `BERNINI_PURGE_COMPILE_CACHE=1` before launching ComfyUI.

This replaces the old behaviour of deleting torch's *global* inductor cache on
every start, which could wipe compile artifacts for other projects.

### Seams between context windows

- Increase `context_overlap`.
- Switch `fuse_method` to `pyramid`.
- Use `uniform_standard` instead of `static_standard`.

---

## License & Citation

This project is provided as a ComfyUI custom node package. The underlying **Bernini** weights and research are released by ByteDance under the **Apache License 2.0**.

If you use Bernini in your research, please cite:

```bibtex
@article{bernini,
  title   = {Bernini: Latent Semantic Planning for Video Diffusion},
  author  = {Chenchen Liu and Junyi Chen and Lei Li and Lu Chi and Mingzhen Sun and Zhuoying Li and Yi Fu and Ruoyu Guo and Yiheng Wu and Ge Bai and Zehuan Yuan},
  journal = {arXiv preprint arXiv:2605.22344},
  year    = {2026}
}
```

---

## Acknowledgments

- Built on top of [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
- Bernini research and weights by [ByteDance Research](https://github.com/bytedance/Bernini).
- The following guidance methods are integrated as open-source contributions:

  - **APG (Adaptive Projected Guidance)** — Sadat et al., *Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models*, ICLR 2025. [arXiv:2410.02416](https://arxiv.org/abs/2410.02416)
  - **RAAG (Ratio-Aware Adaptive Guidance)** — Zhu et al., *RAAG: Ratio Aware Adaptive Guidance*, 2025. [arXiv:2508.03442](https://arxiv.org/abs/2508.03442)
  - **S² (Stochastic Self-Guidance)** — Chen et al., *Stochastic Self-Guidance for Training-Free Enhancement of Diffusion Models*, ICLR 2026. [arXiv:2508.12880](https://arxiv.org/abs/2508.12880)
  - **Z² (Zero-Cost Zigzag Sampling)** — Li et al., *Z²-Sampling: Zero-Cost Zigzag Trajectories for Semantic Alignment in Diffusion Models*, 2026. [arXiv:2604.23536](https://arxiv.org/abs/2604.23536)
  - **STG (Spatiotemporal Skip Guidance)** — Hyung et al., *Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling*, CVPR 2025. [arXiv:2411.18664](https://arxiv.org/abs/2411.18664)
- Model architecture derived from the Wan 2.1 video diffusion family.
- Context-window logic inspired by `ComfyUI-WanVideoWrapper` and `AnimateDiff-Evolved`.
- TeaCache, NAG, and attention-backend integrations adapted from the open-source video-generation community.
