# ComfyUI-BerniniR — Developer / Maintainer Notes

Companion to [`README.md`](./README.md). This document covers entry points,
data flow, and the gotchas that bite when you edit this package. Keep it in
sync with the code — the items below were all verified against the current
source, not guessed.

## Entry points

- `nodes/__init__.py` — exports the 19 node classes that the root
  `__init__.py` registers via `NODE_CLASS_MAPPINGS` /
  `NODE_DISPLAY_NAME_MAPPINGS`.
- `models/` — `wan_model.py` (Wan DiT + NAG / RoPE patches) and
  `wan_compile.py` (torch.compile helper with Windows SDK detection).
- `utils/` — `injection.py` (`InjectionContext`, the single injection call
  site), `teacache.py`, `block_swap.py`, `model_manager.py`, `types.py`,
  `vram.py`.
- `attention/` — five-backend detection + `torch.custom_op` registration.
- `context/` — temporal context-window schedulers (ported from
  ComfyUI-WanVideoWrapper / AnimateDiff-Evolved).

> **No `web/` directory and no `WEB_DIRECTORY`.** All UI is expressed through
> node sockets. In particular, `BerniniR_GuidanceConfig` is a typed node, not a
> front-end widget, and `BerniniR_KSampler` no longer owns guidance widgets.
> Do not re-add a `web/` script expecting it to toggle sampler widgets — the
> sampler does not expose them.

## Sampling data flow

1. Loaders (`loaders.py`) return a lazy `BerniniRModelHandle`; disk I/O, GPU
   transfer, and LoRA merging are deferred to `handle.load()` at sampling time.
2. `BerniniR_KSampler.sample()` builds an `InjectionContext` once
   (`injection.build`) and applies it through `apply_options`,
   `apply_block_swap`, and `apply_noise`. **FreeNoise is applied exactly once,
   inside `apply_noise`** — there is no second call site in the sampler.
3. Guidance arrives as a `BERNINI_GUIDANCE_CONFIG` socket from
   `BerniniR_GuidanceConfig`; it is not a widget on the sampler.

## Gotchas (easy to regress)

- **FreeNoise is applied exactly once.** Do not also call `_apply_freenoise`
  inside the sampler — that double-shuffles the noise (permutation squared,
  changes the result when a window spans ≥ 3 latent frames). The dual-expert
  path (`bernini_sample_dual`) must call `inj.apply_noise` too.
- **TeaCache vs torch.compile.** When a model is compiled, TeaCache
  (`teacache.py`) restores the *eager* transformer forward
  (`_original_transformer_forward` saved by `wan_compile.py`) before patching
  `block.forward`, so its skip hooks actually take effect. If compile's
  save/restore of the forward changes, TeaCache can silently no-op. Mirror the
  guard used in `apply_block_swap`.
- **Compile cache is isolated and version-gated.** `wan_compile.py` redirects
  `TORCHINDUCTOR_CACHE_DIR` to `%TEMP%/bernini_r_inductor_cache` (override with
  `BERNINI_COMPILE_CACHE_DIR`) and auto-purges only when the code version
  changes (`.bernini_cache_version` sentinel). Never `rmtree` torch's global
  inductor cache on import — use `purge_compile_cache()`, the
  `BERNINI_PURGE_COMPILE_CACHE=1` env var, or the `purge_cache` toggle on
  `BerniniR Compile Model`.
- **`rel_l1_thresh` is an absolute mean L1 distance, not normalized** — despite
  the name.

## Common modification scenarios

- **Add a guidance mode** → extend `guidance_config.py` and the single
  `self._dispatch` int in `bernini_sampling.py` (do not duplicate the dispatch
  computation).
- **Add a node** → register it in both `nodes/__init__.py` and the root
  `__init__.py`.
- **Change window math** → edit `context/windows.py`. Pixel→latent conversion
  happens in `utils/types.py` (`BerniniContext.__post_init__`), not in the
  sampler.
