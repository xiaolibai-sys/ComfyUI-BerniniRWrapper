"""
Segment-based hard-cut prompt schedule for Bernini-R.

Replaces the old prompt_travel / slerp / aido interpolation system with
simple hard-cut segments.  Each segment has exactly one prompt that is
constant across its frame range.  Between segments, the last decoded
frame of the previous segment is VAE-encoded and injected as temporal
context for the next segment — producing natural transitions without
cross-attention injection.

Format::

    1-30: a cat walking
    31-60: a dog running
    61-81: a bird flying

- Segments separated by ``;``, ``\\n``, or ``；`` (full-width semicolon).
- Frame range uses ``-`` or ``－`` (full-width hyphen).
- Separator between range and prompt is ``:`` or ``：`` (full-width colon).
- One prompt per segment only (``/`` syntax removed).
- Uncovered frames forward-fill from the last segment.
"""

from __future__ import annotations

import re

import torch
import folder_paths

from ..utils.types import SegmentSpec, EncodedSegment
from ..utils.text_cache import (
    _load_cached_conditioning,
    _save_cached_conditioning,
    _encode_text,
    _load_clip_internal,
)
from ..utils.vram import collect_garbage

from ..utils.log import get_logger as _get_logger

logger = _get_logger("Segment")
# Full-width and half-width variants accepted.
_SEGMENT_RE = re.compile(r"(\d+)\s*[-－]\s*(\d+)\s*[:：]\s*(.+)")
_SEP_RE = re.compile(r"[;\n；]")

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_segments(text: str) -> list[SegmentSpec]:
    """Parse a segment schedule string into a list of ``SegmentSpec``.

    Returns segments sorted by start frame.  Uncovered regions between
    segments are NOT filled here — the sampling pipeline handles forward-fill.
    """
    segments: list[SegmentSpec] = []
    for part in _SEP_RE.split(text):
        part = part.strip()
        if not part:
            continue
        m = _SEGMENT_RE.match(part)
        if not m:
            logger.warning(
                "SegmentSchedule: skipping malformed line: %r", part)
            continue
        start = int(m.group(1))
        end = int(m.group(2))
        if start > end:
            # Reversed range (e.g. "30-10") is almost always a typo; swap it
            # rather than producing an empty / negative-length segment that
            # would break the scheduler's frame arithmetic.
            logger.warning(
                "SegmentSchedule: malformed range %d-%d "
                "(start > end); swapping to %d-%d.",
                start, end, end, start,
            )
            start, end = end, start
        prompt = m.group(3).strip()
        if not prompt:
            continue
        segments.append(SegmentSpec(start_frame=start, end_frame=end, prompt=prompt))

    segments.sort(key=lambda s: s.start_frame)
    return segments


def segments_cover_all(
    segments: list[SegmentSpec],
    total_frames: int,
) -> list[SegmentSpec]:
    """Forward-fill gaps and ensure coverage from frame 1 to *total_frames*.

    Returns a new list where every frame is covered by exactly one segment.
    """
    if not segments:
        return [SegmentSpec(start_frame=1, end_frame=total_frames, prompt="")]

    result: list[SegmentSpec] = []
    carry_prompt = segments[0].prompt

    # Fill leading gap.
    if segments[0].start_frame > 1:
        result.append(SegmentSpec(
            start_frame=1,
            end_frame=segments[0].start_frame - 1,
            prompt=carry_prompt,
        ))

    for i, seg in enumerate(segments):
        # Gap between previous and this segment.
        if result and result[-1].end_frame + 1 < seg.start_frame:
            result.append(SegmentSpec(
                start_frame=result[-1].end_frame + 1,
                end_frame=seg.start_frame - 1,
                prompt=carry_prompt,
            ))
        result.append(seg)
        carry_prompt = seg.prompt

    # Fill trailing gap.
    if result[-1].end_frame < total_frames:
        result.append(SegmentSpec(
            start_frame=result[-1].end_frame + 1,
            end_frame=total_frames,
            prompt=carry_prompt,
        ))

    return result


class BerniniR_SegmentSchedule:
    """Parse a hard-cut segment schedule and encode prompts through CLIP.

    Outputs encoded conditioning for each segment.  The sampler runs
    each segment as an independent denoising pass, with the previous
    segment's last decoded frame injected as VAE context.

    Inputs (required):
        text (STRING): Multi-line segment schedule.
        total_frames (INT): Total pixel frames of the output video.

    Inputs (optional):
        clip (CLIP): Text encoder. If not connected, clip_name is used.
        clip_name (COMBO): CLIP model file.
        clip_type (COMBO): CLIP architecture type.
        clip_device (COMBO): "default" (GPU) or "cpu".
        force_offload (BOOLEAN): Offload CLIP after encoding.
        use_disk_cache (BOOLEAN): Cache embeddings to disk.
        cache_tag (STRING): Optional cache isolation tag.
        negative_prompt (STRING): Negative prompt (same for all segments).
        width (INT): Output width (for latent shape computation).
        height (INT): Output height.

    Outputs:
        segments (BERNINI_SEGMENTS): Encoded segment list.
        negative (CONDITIONING): Shared negative conditioning.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_prompt": ("STRING", {
                    "default": (
                        "1-40: a cat walking on a sunny street; "
                        "41-81: a dog running in the park"
                    ),
                    "multiline": True,
                    "tooltip": (
                        "Segment schedule. Format: 'a-b: prompt'. "
                        "Segments separated by ; or newline. "
                        "Frames are 1-based. One prompt per segment."
                    ),
                }),
                "total_frames": ("INT", {
                    "default": 81, "min": 1, "max": 4096, "step": 1,
                    "tooltip": "Total pixel frames of the output video.",
                }),
                "negative_prompt": ("STRING", {
                    "default": DEFAULT_NEGATIVE_PROMPT, "multiline": True,
                    "tooltip": "Negative prompt (same for all segments).",
                }),
            },
            "optional": {
                "transition_frames": ("INT", {
                    "default": 8, "min": 0, "max": 256, "step": 4,
                    "tooltip": (
                        "Crossfade width between adjacent segments, in pixel "
                        "frames (converted to latent frames by /4). Adjacent "
                        "segments overlap by this amount and blend smoothly. "
                        "0 = hard cut (no crossfade)."
                    ),
                }),
                "clip": ("CLIP", {
                    "tooltip": "CLIP text encoder. If connected, clip_name is ignored.",
                }),
                "clip_name": (folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "CLIP model file (ignored if 'clip' input is connected).",
                }),
                "clip_type": (["wan"], {
                    "default": "wan", "tooltip": "CLIP architecture type.",
                }),
                "clip_device": (["default", "cpu"], {
                    "default": "cpu", "tooltip": "'cpu' saves VRAM during encoding.",
                }),
                "force_offload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Offload CLIP to CPU after encoding.",
                }),
                "use_disk_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache encoded embeddings to disk.",
                }),
                "cache_tag": ("STRING", {
                    "default": "", "tooltip": "Optional cache isolation tag.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "build"
    CATEGORY = "Bernini-R/Conditioning"
    DESCRIPTION = (
        "Segment prompt-travel schedule: each segment has a single prompt "
        "covering a frame range. Outputs standard CONDITIONING — connect "
        "directly to KSampler or DualExpertSampler. Compatible with NAG. "
        "The sampler denoises the whole video in one pass and swaps the "
        "text embedding per temporal window; adjacent segments crossfade "
        "over 'transition_frames' for smooth prompt transitions."
    )

    @staticmethod
    def _derive_clip_tag(clip) -> str:
        try:
            model = clip.patcher.model
            total = sum(p.numel() for p in model.parameters())
            dtypes = sorted({str(p.dtype) for p in model.parameters()})
            return f"{model.__class__.__name__}_{total}_{'-'.join(dtypes)}"
        except Exception:
            return "unknown"

    def build(
        self,
        positive_prompt: str = "1-81: a cinematic video",
        total_frames: int = 81,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        transition_frames: int = 8,
        clip=None,
        clip_name: str = "",
        clip_type: str = "wan",
        clip_device: str = "cpu",
        force_offload: bool = True,
        use_disk_cache: bool = True,
        cache_tag: str = "",
    ):
        # ── Parse ──────────────────────────────────────────────────────
        raw = parse_segments(positive_prompt)
        segments = segments_cover_all(raw, total_frames)
        logger.info(
            "SegmentSchedule: %d raw segments → %d after fill, "
            "%d total frames", len(raw), len(segments), total_frames,
        )

        # ── Collect unique prompts ─────────────────────────────────────
        unique: list[str] = []
        seen: set[str] = set()
        for seg in segments:
            if seg.prompt not in seen:
                seen.add(seg.prompt)
                unique.append(seg.prompt)
        if not unique:
            unique = [""]
            segments = [SegmentSpec(start_frame=1, end_frame=total_frames, prompt="")]

        # ── Cache tag ──────────────────────────────────────────────────
        user_tag = cache_tag.strip()
        if user_tag:
            tag = user_tag
        elif clip is not None:
            tag = self._derive_clip_tag(clip)
        elif clip_name:
            tag = f"{clip_name}:{clip_type}:{clip_device}"
        else:
            tag = "unknown"

        # ── Encode all unique prompts ───────────────────────────────────
        embeddings: dict[str, torch.Tensor] = {}
        pooled: dict[str, dict] = {}
        uncached: list[str] = []

        for p in unique:
            cached = _load_cached_conditioning(p, tag) if use_disk_cache else None
            if cached is not None:
                embeddings[p] = cached[0][0]
                pooled[p] = cached[0][1]
            else:
                uncached.append(p)

        # Negative prompt
        neg_cached = _load_cached_conditioning(negative_prompt, tag) if use_disk_cache else None
        if neg_cached is not None:
            neg_cond_out = neg_cached
        else:
            uncached.append(negative_prompt)
            neg_cond_out = None

        clip_loaded_internally = False
        if uncached:
            try:
                if clip is None:
                    if not clip_name:
                        raise ValueError(
                            "No CLIP input and no clip_name; cannot encode.")
                    logger.info(
                        "SegmentSchedule: loading CLIP on demand: %s",
                        clip_name)
                    clip = _load_clip_internal(clip_name, clip_type, clip_device)
                    clip_loaded_internally = True

                for p in uncached:
                    cond_out = _encode_text(clip, p)
                    if p == negative_prompt:
                        neg_cond_out = cond_out
                    else:
                        embeddings[p] = cond_out[0][0]
                        pooled[p] = cond_out[0][1]
                    if use_disk_cache:
                        _save_cached_conditioning(p, cond_out, tag)
            finally:
                if force_offload and clip is not None and clip_loaded_internally:
                    # Only release a CLIP we loaded ourselves; a CLIP passed in
                    # by the user is shared input and must not be unloaded here.
                    del clip
                    collect_garbage()

        # ── Build positive conditioning with segment metadata ──────────
        first_emb = embeddings[unique[0]]
        first_pooled = pooled.get(unique[0], {})

        # Pixel→latent frame mapping (matches Wan VAE: (n-1)//4 + 1).
        def _to_lat(px: int) -> int:
            return max(0, (px - 1) // 4)

        overlap_latent = max(0, transition_frames // 4)

        encoded_segments = []
        for seg in segments:
            emb = embeddings.get(seg.prompt, first_emb)
            pool = pooled.get(seg.prompt, first_pooled)
            encoded_segments.append(EncodedSegment(
                start_frame=seg.start_frame,
                end_frame=seg.end_frame,
                embed=emb,
                pooled=pool,
                start_latent=_to_lat(seg.start_frame),
                end_latent=_to_lat(seg.end_frame) + 1,
            ))

        positive = [[first_emb, {
            **first_pooled,
            "segment_specs": encoded_segments,
            "segment_overlap_latent": overlap_latent,
        }]]

        # ── Negative conditioning (shared) ────────────────────────────
        neg_embed = neg_cond_out[0][0] if neg_cond_out is not None else torch.zeros(1, 512, 4096)
        neg_pool = neg_cond_out[0][1] if neg_cond_out is not None else {}
        negative = [[neg_embed, neg_pool]]

        return (positive, negative)
