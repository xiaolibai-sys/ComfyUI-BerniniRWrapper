"""
Typed data-transfer payloads for Bernini-R nodes.

This module centralises the structured data that flows between custom nodes.
ComfyUI native containers (CONDITIONING, LATENT, CLIP, VAE, IMAGE) keep their
external shape so other ComfyUI nodes can still consume them; only the custom
``BERNINI_*`` sockets are represented as typed dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

import torch
import torch.nn as nn

# Lazy import for comfy_kitchen (only resolved in ComfyUI environment).
try:
    from comfy_kitchen.tensor.base import QuantizedTensor, get_layout_class as _get_layout_class
except ImportError:
    QuantizedTensor = None  # type: ignore[assignment]
    _get_layout_class = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContextSchedule(str, Enum):
    """Context-window scheduling strategies."""
    UNIFORM_STANDARD = "uniform_standard"
    UNIFORM_LOOPED = "uniform_looped"
    STATIC_STANDARD = "static_standard"


class FuseMethod(str, Enum):
    """Context-window fusion methods."""
    LINEAR = "linear"
    SMOOTH = "smooth"
    PYRAMID = "pyramid"
    NONE = "none"


class GuidanceMode(str, Enum):
    """Guidance strategies supported by BerniniModelWrapper."""
    CFG = "CFG"
    APG = "APG"
    RAAG = "RAAG"
    S2 = "S2"
    Z2 = "Z2"
    STG_A = "STG_A"
    STG_R = "STG_R"

    @property
    def is_stg(self) -> bool:
        return self.value.startswith("STG")

    @property
    def stg_variant(self) -> str:
        """Return 'A' or 'R' for STG modes, empty string otherwise."""
        return self.value[4:] if self.is_stg else ""


class MaskMode(str, Enum):
    """Differential-diffusion mask behaviours."""
    ANNEAL = "anneal"
    FREEZE = "freeze"


# ---------------------------------------------------------------------------
# Custom socket payloads
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BerniniContext:
    """Temporal context-window configuration consumed by the sampler.

    Pixel-frame values are converted to latent-frame values once at construction
    time so the sampler no longer repeats the ``// 4`` arithmetic on every run.
    """
    schedule: ContextSchedule
    context_frames: int
    context_stride: int
    context_overlap: int
    freenoise: bool
    fuse_method: FuseMethod
    rope_ntk_scale: float
    latent_frames: int = field(init=False)
    latent_overlap: int = field(init=False)
    latent_stride: int = field(init=False)

    def __post_init__(self):
        # Wan VAE uses a 4x temporal downscale.  ``(F - 1) // 4 + 1`` matches
        # ComfyUI's latent-frame calculation for the last partial group.
        object.__setattr__(
            self, "latent_frames",
            max(1, (self.context_frames - 1) // 4 + 1)
        )
        object.__setattr__(
            self, "latent_overlap",
            max(0, self.context_overlap // 4)
        )
        # Stride is provided in log2 levels by the UI; keep at least 1.
        object.__setattr__(
            self, "latent_stride",
            max(1, self.context_stride // 4)
        )


@dataclass(frozen=True)
class BerniniBlockSwap:
    """Block-swap / VRAM-offloading configuration."""
    block_to_swap: int
    prefetch: bool = True
    prefetch_count: int = 1
    pin_memory: bool = False
    loading_mode: str = "Streaming"  # "Full" or "Streaming"

    @property
    def lazy(self) -> bool:
        """Whether to load block weights on demand (streaming) or all at once.
        
        - Streaming: blocks loaded from disk on demand by _DiskPrefetcher.
        - Full: all block weights loaded into CPU RAM at startup, then
          BlockSwapManager moves them to the GPU window as needed.
        """
        return self.loading_mode == "Streaming"


@dataclass(frozen=True)
class DiskLoadRequest:
    """A single disk→RAM block-load job handed to the prefetch thread pool.

    Carries everything ``_DiskPrefetcher`` needs to populate one transformer
    block, so the forward thread dispatches work to the pool with the same
    typed ``@dataclass(frozen=True)`` payload convention used everywhere else
    in Bernini-R instead of passing a bare ``block_idx`` integer.
    """

    block_idx: int
    group_key: str


# ---------------------------------------------------------------------------
# Slot entry — unified type for ring-buffer slot pool
# ---------------------------------------------------------------------------

class SlotEntry:
    """Unified slot entry for the engine's ring-buffer slot pool.

    Wraps either a regular ``torch.Tensor`` or the internal components of a
    comfy_kitchen ``QuantizedTensor`` (``_qdata`` + ``_params.scale``), so
    the swap engine never branches on parameter type.
    """

    __slots__ = ('is_qt', 'data', 'scale', 'layout_cls', 'orig_dtype',
                 'orig_shape', 'lora')

    def __init__(self, *, data: torch.Tensor,
                 scale: torch.Tensor | None = None,
                 layout_cls: str = '',
                 orig_dtype: torch.dtype | None = None,
                 orig_shape: tuple[int, ...] | None = None,
                 lora: list | None = None):
        self.is_qt = scale is not None
        self.data = data
        self.scale = scale
        self.layout_cls = layout_cls
        self.orig_dtype = orig_dtype
        self.orig_shape = orig_shape
        # Co-located LoRA payload (list of entry dicts), or None when the
        # block weights in this slot are already merged.  The unified slot is
        # ``(block, lora)`` before folding and ``(block)`` after — "folded"
        # is inferred from whether ``lora`` is still present, so no external
        # bookkeeping set is needed.
        self.lora = lora

    @classmethod
    def empty_like(cls, param: nn.Parameter, device, pin_memory: bool = False):
        """Pre-allocate a slot buffer matching *param* on *device*."""
        d = param.data
        if hasattr(d, '_qdata'):
            return cls(
                data=torch.empty_like(
                    d._qdata, device=device,
                    pin_memory=(device == "cpu" and pin_memory)),
                scale=torch.empty_like(
                    d._params.scale, device=device,
                    pin_memory=(device == "cpu" and pin_memory)),
                layout_cls=d._layout_cls,
                orig_dtype=d._params.orig_dtype,
                orig_shape=tuple(d.shape),
            )
        return cls(
            data=torch.empty_like(d, device=device,
                                  pin_memory=(device == 'cpu' and pin_memory)),
        )

    @classmethod
    def empty_like_entry(cls, entry: "SlotEntry", device,
                         pin_memory: bool = False):
        """Pre-allocate a slot buffer on *device* matching *entry*'s layout.

        Unlike ``empty_like`` (which reads from an ``nn.Parameter``), this
        reads from another ``SlotEntry`` — safe when the source parameter
        may already be on a different device.
        """
        if entry.is_qt:
            return cls(
                data=torch.empty_like(
                    entry.data, device=device,
                    pin_memory=(device == "cpu" and pin_memory)),
                scale=torch.empty_like(
                    entry.scale, device=device,
                    pin_memory=(device == "cpu" and pin_memory)),
                layout_cls=entry.layout_cls,
                orig_dtype=entry.orig_dtype,
                orig_shape=entry.orig_shape,
            )
        return cls(
            data=torch.empty_like(
                entry.data, device=device,
                pin_memory=(device == 'cpu' and pin_memory)),
        )

    def copy_from(self, param: nn.Parameter, non_blocking: bool = False) -> None:
        """Copy *param*'s weight data into this pre-allocated buffer."""
        d = param.data
        if self.is_qt:
            self.data.copy_(d._qdata, non_blocking=non_blocking)
            self.scale.copy_(d._params.scale, non_blocking=non_blocking)
        else:
            self.data.copy_(d, non_blocking=non_blocking)

    def assign_to(self, block: nn.Module, param_name: str) -> None:
        """Wire this buffer's data into *block*'s submodule ``_parameters``.

        Quantized entries: reconstruct a fresh GPU ``QuantizedTensor`` and
        replace the parameter via ``_parameters`` dict (bypasses the broken
        ``p.data = ...`` on wrapper subclasses).
        Regular entries: ``_parameters[name].data = ...`` (works normally).
        """
        *mod_path, pn = param_name.split('.')
        sub = block
        for part in mod_path:
            sub = getattr(sub, part)

        if self.is_qt:
            layout_cls = _get_layout_class(self.layout_cls)
            params = layout_cls.Params(
                scale=self.scale,
                orig_dtype=self.orig_dtype,
                orig_shape=self.orig_shape,
            )
            qt = QuantizedTensor(self.data, self.layout_cls, params)
            sub._parameters[pn] = nn.Parameter(qt, requires_grad=False)
        else:
            sub._parameters[pn].data = self.data


# ── LoraTensorEntry: typed metadata for a LoRA tensor in an offset plan ─────


@dataclass
class LoraTensorEntry:
    """Metadata for one LoRA tensor in an offset plan (no materialised tensor).

    ``kind`` is one of ``"A"``, ``"B"``, ``"alpha"``, ``"diff_b"``,
    ``"diff"``.  ``spec_idx`` is the index into the owning
    ``LoraBlockReader._spec_strengths`` list.
    """
    fd_idx: int
    offset: int
    length: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    kind: str
    spec_idx: int


@dataclass(frozen=True)
class BerniniTeaCache:
    """TeaCache block-caching configuration."""
    max_skip_blocks: int
    start_block: int
    rel_l1_thresh: float
    warmup_steps: int
    cooldown_steps: int


@dataclass(frozen=True)
class BerniniAttention:
    """Attention-backend selection returned by BerniniR_AttentionConfig."""
    backend: str
    force_backend: bool
    available: tuple[str, ...]
    best: str


@dataclass(frozen=True)
class BerniniGuidance:
    """Per-step guidance scale schedule."""
    values: list[float]
    curve: str
    steps: int
    start: float
    end: float

    def __post_init__(self):
        # Defensive: ensure the list length matches the declared step count.
        if len(self.values) != self.steps:
            object.__setattr__(
                self, "values",
                [self.start] * self.steps if self.steps > 0 else []
            )


@dataclass(frozen=True)
class BerniniGuidanceConfig:
    """Guidance strategy selection + per-mode hyper-parameters.

    Replaces the 8 individual guidance widgets on the sampler node with
    a single ``BERNINI_GUIDANCE_CONFIG`` socket, following the same pattern
    as ``BERNINI_CTX`` / ``BERNINI_BLOCKSWAP`` / ``BERNINI_TEACACHE``.
    """
    mode: GuidanceMode = GuidanceMode.CFG

    # APG
    apg_eta: float = 0.15
    apg_rescale: bool = True
    apg_momentum: float = 0.0

    # RAAG
    raag_alpha: float = 1.0

    # S²
    s2_omega: float = 1.0

    # STG
    stg_scale: float = 1.0
    stg_block_idx: str = "10,20,27"

    # Z²
    z2_collapse: float = 0.3


# ---------------------------------------------------------------------------
# Segment schedule (hard-cut denoising)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentSpec:
    """A single segment in a hard-cut prompt schedule.

    Each segment has a single prompt and covers a contiguous frame range.
    No interpolation — the prompt is constant across the segment.
    """
    start_frame: int   # 1-based pixel frame, inclusive
    end_frame: int     # 1-based pixel frame, inclusive
    prompt: str        # single prompt (no "/" syntax)


@dataclass
class EncodedSegment:
    """A segment with its CLIP-encoded prompt embedding.

    Passed from ``BerniniR_SegmentSchedule`` to the sampler via the
    ``segment_specs`` key in the positive conditioning extra dict.

    ``start_latent`` / ``end_latent`` are the pre-computed latent-space frame
    bounds (``end_latent`` exclusive) so the sampler's context-window framework
    can build prompt-travel windows without repeating the pixel→latent math.
    """
    start_frame: int
    end_frame: int
    embed: torch.Tensor
    pooled: dict
    start_latent: int = 0
    end_latent: int = 0


@dataclass
class SegmentWindow:
    """A latent-space window paired with its prompt embedding.

    Consumed by the context-window wrapper (``_build_context_window_wrapper``)
    to run temporal prompt-travel: each window denoises with its own text
    embedding, and adjacent windows crossfade in their overlap region.  All Wan
    text embeds share shape ``(1, 512, 4096)``, so swapping the embedding per
    window is always shape-safe.
    """
    start_latent: int
    end_latent: int          # exclusive
    embed: torch.Tensor      # (1, 512, 4096)


# ---------------------------------------------------------------------------
# Conditioning helper
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    """Single ComfyUI conditioning entry: a prompt embedding plus extra dict."""
    embed: torch.Tensor
    extra: dict[str, Any]

    def to_comfy(self) -> list:
        """Return the native ComfyUI representation ``[tensor, dict]``."""
        return [self.embed, self.extra]


@dataclass
class Conditioning:
    """List-compatible wrapper around ComfyUI's ``[[tensor, dict], ...]`` format.

    Nodes still return the underlying list on the wire so ComfyUI and other
custom nodes can consume the output; internally we use this class to avoid
    magic indices like ``positive[0][1]``.
    """
    items: list[Condition]

    @classmethod
    def from_comfy(cls, comfy_cond: list | Conditioning | None) -> "Conditioning":
        """Build a ``Conditioning`` helper from a native ComfyUI list."""
        if comfy_cond is None:
            return cls([])
        if isinstance(comfy_cond, Conditioning):
            return cls(list(comfy_cond.items))
        parsed = []
        for entry in comfy_cond:
            if isinstance(entry, Condition):
                parsed.append(entry)
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                parsed.append(Condition(embed=entry[0], extra=dict(entry[1] or {})))
            else:
                raise ValueError(f"Invalid conditioning entry: {entry!r}")
        return cls(parsed)

    def to_comfy(self) -> list[list]:
        """Return the native ComfyUI representation."""
        return [item.to_comfy() for item in self.items]

    # -- list-like interface ------------------------------------------------

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Condition]:
        return iter(self.items)

    def __getitem__(self, index: int | slice):
        result = self.items[index]
        if isinstance(result, list):
            return Conditioning(result)
        return result

    def __bool__(self) -> bool:
        return bool(self.items)

    # -- convenience accessors ----------------------------------------------

    @property
    def first(self) -> Condition | None:
        """Return the first conditioning entry, or ``None`` if empty."""
        return self.items[0] if self.items else None

    @property
    def embed(self) -> torch.Tensor | None:
        """Return the embedding of the first entry."""
        first = self.first
        return first.embed if first is not None else None

    @property
    def extra(self) -> dict[str, Any] | None:
        """Return the extra dict of the first entry."""
        first = self.first
        return first.extra if first is not None else None

    def get_extra(self, key: str, default: Any = None) -> Any:
        """Read a value from the first entry's extra dict."""
        extra = self.extra
        if extra is None:
            return default
        return extra.get(key, default)

    def set_extra(self, key: str, value: Any) -> "Conditioning":
        """Set a value on every entry's extra dict (in-place)."""
        for item in self.items:
            item.extra[key] = value
        return self

    def with_extra(self, **values) -> "Conditioning":
        """Return a new ``Conditioning`` with merged extra dicts."""
        new_items = []
        for item in self.items:
            merged = dict(item.extra)
            merged.update(values)
            new_items.append(Condition(embed=item.embed, extra=merged))
        return Conditioning(new_items)

    def copy(self) -> "Conditioning":
        """Shallow copy: new ``Condition`` objects sharing the same tensors."""
        return Conditioning([
            Condition(embed=item.embed, extra=dict(item.extra))
            for item in self.items
        ])
