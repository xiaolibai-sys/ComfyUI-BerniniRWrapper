# ComfyUI-BerniniR — Developer / Maintainer Notes

Companion to [`README.md`](./README.md). This document covers entry points,
data flow, and the gotchas that bite when you edit this package. Keep it in
sync with the code — the items below were all verified against the current
source, not guessed.

## Entry points

- `nodes/__init__.py` — exports the 19 node classes that the root
  `__init__.py` registers via `NODE_CLASS_MAPPINGS` /
  `NODE_DISPLAY_NAME_MAPPINGS`.
- `models/wan_model.py` — Wan DiT + NAG / RoPE patches + block swap manager
  construction in `pre_forward` / `transformer_forward`.
- `models/wan_compile.py` — torch.compile helper with Windows SDK detection.
- `utils/block_swap.py` — `BlockSwapManager` (ring buffer + SlotEntry +
  PinStage), the unified GPU-placement engine for all block-swap modes.
- `utils/model_manager.py` — `BerniniRModelHandle.load()`, the single entry
  point for loading a model (whether block swap is enabled or not).
- `utils/types.py` — typed payloads (`BerniniBlockSwap`, `BerniniContext`,
  `SlotEntry`) and the `Conditioning` helper.
- `utils/injection.py` — `InjectionContext` factory + per-run config.
- `attention/` — five-backend detection + `torch.custom_op` registration.
- `context/` — temporal context-window schedulers.

> **No `web/` and no `WEB_DIRECTORY`.** All UI is expressed through node
> sockets. Do not re-add a `web/` script expecting it to toggle sampler
> widgets — the sampler does not expose them.

## Unified block-swap architecture (v1.1.x)

Every model loaded via `BerniniRModelHandle.load()` receives a
`BerniniBlockSwap` dataclass that controls weight placement:

| `block_to_swap` | Loading mode | GPU placement | CPU home pool |
|---|---|---|---|
| 0 | — | Standard ComfyUI (`mm.load_models_gpu`) | None (zero CPU overhead) |
| >0 | Full | BlockSwapManager (ring buffer + SlotEntry) | (N-W) blocks in CPU RAM |
| >0 | Streaming | BlockSwapManager + DiskHome | Blocks loaded from disk on demand |

When `block_to_swap > 0`:

1. **`model.load()`** stores `_block_swap_config` on `patcher.model.diffusion_model`
   and sets `load_device = offload_device` so ComfyUI leaves weights on CPU.
2. **`pre_forward()`** reads `self._block_swap_config` and creates a
   `BlockSwapManager` with `window = total - block_to_swap` GPU ring slots.
3. **`transformer_forward()`** calls `_bswap.prepare(i)` for each block `i`,
   which loads the block's weights into a GPU ring slot via `SlotEntry`.
4. After forward, blocks outside the window are offloaded back to the CPU home
   pool (RingHome or DiskHome).

The `_block_swap` flag in `transformer_options` is **no longer used** — the
model reads its config directly from `self._block_swap_config`.

### Standard KSampler fallback

When `block_to_swap = 0` (or `block_swap_config` is `None`), the model follows
ComfyUI's standard GPU-only path — no ring buffer, no SlotEntry, no home pool.
`load_device = get_torch_device()` and `mm.load_models_gpu` handles everything.

## Key data types

### `BerniniBlockSwap` (`utils/types.py`)
```python
@dataclass(frozen=True)
class BerniniBlockSwap:
    block_to_swap: int          # 0 = disabled >0 = offload this many blocks
    prefetch: bool = True       # CUDA stream for async H2D overlap
    prefetch_count: int = 1     # blocks to prefetch ahead of window
    pin_memory: bool = False    # page-lock CPU home pool for true async H2D
    loading_mode: str = "Streaming"  # "Full" or "Streaming"

    @property
    def lazy(self) -> bool:
        return self.loading_mode == "Streaming"
```

### `SlotEntry` (`utils/types.py`)
Unifies regular `torch.Tensor` and `QuantizedTensor` (fp8) for the swap engine:
- `empty_like(param, device)` — allocate a matching buffer
- `copy_from(param)` — copy weight data into the buffer
- `assign_to(block, param_name)` — replace `block._parameters[name]` with
  this buffer's data (creates a fresh `nn.Parameter` to bypass
  `__torch_wrapper_subclass__` interception of `.data =`)

### `PinStage` (`utils/block_swap.py`)
```python
@dataclass
class PinStage:
    """Pinned CPU staging ring, size = prefetch_count.
    Built lazily from GPU entry metadata (safe device-independent ctor).
    ```
    slots: list[dict | None]    # dict[str, SlotEntry]
    built: list[bool]
    
    def ensure(ring_idx, template_dict, pin_memory) -> dict
    ```
```

## GPU ring buffer

GPU slots use a cursor-based ring buffer instead of a free-list:

```python
def _ring_acquire(self) -> int:
    idx = self._gpu_cursor % ng
    self._gpu_cursor += 1
    return idx
```

Cursor auto-advances with the sliding window — the slot that just fell out of
the window is the next one to be reused. No explicit `release()` needed.

## Sampling data flow

1. `BerniniR_KSampler.sample()` (or dual-expert) receives typed socket inputs:
   `model_handle`, `block_swap_args`, `context_options`, etc.
2. If TeaCache is active, loads the model early via
   `model_handle.load(block_swap_config=block_swap_args)` and attaches block-skip
   hooks.
3. Calls `bernini_sample()` / `bernini_sample_dual()` which calls
   `model_handle.load(block_swap_config=block_swap_args)`:
   - If block swap is enabled: `load_device = offload_device`, model stays CPU,
     `_block_swap_config` stored on diffusion model.
   - If disabled: `load_device = GPU`, `mm.load_models_gpu` moves model to GPU.
4. `InjectionContext.build()` extracts guidance, NAG, STG, differential diffusion
   params from conditioning + config nodes.
5. `BerniniModelWrapper.__call__()` → `_forward_batched()` → context-window
   wrapper (`sampler.py`) → `apply_model_func()` → `model_base.apply_model()` →
   `diffusion_model.forward()` → `pre_forward()`:
   - Creates `BlockSwapManager` from `self._block_swap_config`.
   - Applies NAG context projection.
   - Calls `transformer_forward()` which iterates blocks with `_bswap.prepare(i)`.
6. After sampling, `unload()` → `BlockSwapManager.shutdown()` → drops all pool
   tensor references → `collect_garbage(aggressive=True)`.

## Peak RAM

| Mode | Startup | Steady state |
|---|---|---|
| Block swap OFF | 1× model on GPU | 1× model on GPU |
| Block swap ON + Full | ~2× CPU (all params + all slots briefly coexisting) | 1× (W GPU + N-W CPU) |
| Block swap ON + Streaming | 0 blocks loaded | 1× (W GPU + N-W CPU), no startup spike |

## Gotchas (easy to regress)

- **FreeNoise is applied exactly once.** Do not also call `_apply_freenoise`
  inside the sampler — that double-shuffles the noise (permutation squared,
  changes the result when a window spans ≥ 3 latent frames). The dual-expert
  path (`bernini_sample_dual`) must call `inj.apply_noise` too.
- **TeaCache pre-load must pass block_swap_args.** If `model_handle.load()` is
  called without `block_swap_config` before `bernini_sample()`, the guard
  returns a GPU-loaded model without `_block_swap_config`. Block swap then
  silently fails and the model runs on CPU.
- **`p.data = d` is a no-op on `__torch_wrapper_subclass__`.** Never assign
  `.data` on a QuantizedTensor parameter — it bypasses `__torch_dispatch__`
  and the old storage stays referenced. Use `SlotEntry.assign_to()` which
  replaces `module._parameters[key] = nn.Parameter(new_tensor)`.
- **`torch.empty_like(cuda_tensor, device='cpu', pin_memory=True)` can OOM.**
  On Windows, CUDA's pinned-memory allocator has a per-process limit.
  `PinStage` only pins `prefetch_count` blocks (default 3) instead of all
  `W+prefetch` ring slots, and the pin pool is only populated during prefetch
  loads (`non_blocking=True`).
- **`non_blocking=True` on unpinned memory is synchronous.** Only pinned CPU
  memory enables truly async H2D copies. Without pinning, `non_blocking=True`
  is a no-op — the copy blocks the issuing stream.
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

## Common modification scenarios

- **Add a guidance mode** → extend `guidance_config.py` and the single
  `self._dispatch` int in `bernini_sampling.py` (do not duplicate the dispatch
  computation).
- **Add a node** → register it in both `nodes/__init__.py` and the root
  `__init__.py`.
- **Change window math** → edit `context/windows.py`. Pixel→latent conversion
  happens in `utils/types.py` (`BerniniContext.__post_init__`), not in the
  sampler.
