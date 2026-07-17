"""
Context window scheduling algorithms.

Strictly ported from ComfyUI-WanVideoWrapper's context_windows/context.py
(originally from AnimateDiff-Evolved by Kosinkadink).

Provides:
  - Window schedulers: uniform_looped, uniform_standard, static_standard
  - Mask generation: create_window_mask (linear / pyramid blending)
  - Window tracker for persistent teacache states
"""

import numpy as np
from typing import Callable, Optional, List

import torch


def ordered_halving(val: int) -> float:
    """Bit-reversal permutation for uniform scheduler stratification."""
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)
    return as_int / (1 << 64)


def does_window_roll_over(window: list[int], num_frames: int) -> tuple[bool, int]:
    """Check if a window wraps around (index decreasing → roll-over)."""
    prev_val = -1
    for i, val in enumerate(window):
        val = val % num_frames
        if val < prev_val:
            return True, i
        prev_val = val
    return False, -1


def _ramp(n: int, device, smooth: bool = True):
    """Build a 0→1 ramp of length *n* along the temporal axis (dim=1).

    With *smooth* (default) the ramp is smoothstep-eased (``3t²-2t³``), which
    is C1-continuous: the slope is zero at both ends, so the crossfade has no
    visible slope discontinuity — the transition band that reads as a brightness
    "bump"/graying at window seams is strongly attenuated versus raw linear.
    Partition of unity is preserved either way.
    """
    t = torch.linspace(0, 1, n, device=device, dtype=torch.float32)
    if smooth:
        t = 3 * t ** 2 - 2 * t ** 3
    return t.view(1, -1, 1, 1)


def shift_window_to_start(window: list[int], num_frames: int):
    """Shift window so its first element is 0 (modulo num_frames)."""
    start_val = window[0]
    for i in range(len(window)):
        window[i] = ((window[i] - start_val) + num_frames) % num_frames


def shift_window_to_end(window: list[int], num_frames: int):
    """Shift window to the end of the frame range (modulo-safe)."""
    shift_window_to_start(window, num_frames)
    end_val = window[-1]
    end_delta = num_frames - end_val - 1
    for i in range(len(window)):
        window[i] = (window[i] + end_delta) % num_frames


def get_missing_indexes(windows: list[list[int]], num_frames: int) -> list[int]:
    """Return frame indices that are NOT covered by any window."""
    all_indexes = list(range(num_frames))
    for w in windows:
        for val in w:
            try:
                all_indexes.remove(val)
            except ValueError:
                pass
    return all_indexes


# ---------------------------------------------------------------------------
# Window schedulers
# ---------------------------------------------------------------------------

def uniform_looped(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    """Yield windows that wrap around cyclically.

    Different windows may be produced at each denoising step — good for
    reducing temporal seams in looped video.
    """
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            yield [e % num_frames for e in range(j, j + context_size * context_step, context_step)]


def uniform_standard(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
) -> list[list[int]]:
    """Uniform multi-stride windows, deduplicated and shifted.

    Returns a list (not generator) so it can be cached for static schedules.
    """
    windows = []
    if num_frames <= context_size:
        windows.append(list(range(num_frames)))
        return windows

    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            windows.append([e % num_frames for e in range(j, j + context_size * context_step, context_step)])

    # Shift any windows that roll over, and delete duplicate windows
    delete_idxs = []
    win_i = 0
    while win_i < len(windows):
        is_roll, roll_idx = does_window_roll_over(windows[win_i], num_frames)
        if is_roll:
            roll_val = windows[win_i][roll_idx]
            shift_window_to_end(windows[win_i], num_frames=num_frames)
            # Always insert the spill window — the neighbour may cover
            # *some* frames at a different stride but not all of them.
            # Dedup below will remove true duplicates.
            windows.insert(win_i + 1, [e % num_frames for e in range(roll_val, min(roll_val + context_size, num_frames))])
        for pre_i in range(0, win_i):
            if windows[win_i] == windows[pre_i]:
                delete_idxs.append(win_i)
                break
        win_i += 1

    for i in reversed(delete_idxs):
        windows.pop(i)
    return windows


def static_standard(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
) -> list[list[int]]:
    """Simple sliding windows with fixed overlap = context_overlap.

    Always returns the same windows regardless of step — optimal for caching.
    The last window is shifted back so the actual overlap with the previous
    window never exceeds ``context_overlap``.
    """
    windows = []
    if num_frames <= context_size:
        windows.append(list(range(num_frames)))
        return windows

    delta = context_size - context_overlap
    if delta <= 0:
        return [list(range(min(num_frames, context_size)))]

    for start_idx in range(0, num_frames, delta):
        ending = start_idx + context_size
        if ending >= num_frames:
            # Last window — align to end at num_frames.
            final_start_idx = max(0, num_frames - context_size)
            # Clamp overlap with the previous window to context_overlap.
            if windows:
                prev_end = windows[-1][-1] + 1
                actual_overlap = prev_end - final_start_idx
                if actual_overlap > context_overlap:
                    final_start_idx = prev_end - context_overlap
            # Never exceed the frame range.
            final_start_idx = max(0, final_start_idx)
            end_idx = min(final_start_idx + context_size, num_frames)
            windows.append(list(range(final_start_idx, end_idx)))
            break
        windows.append(list(range(start_idx, start_idx + context_size)))
    return windows


def get_context_scheduler(name: str) -> Callable:
    """Return the scheduler function for the given name."""
    if name == "uniform_looped":
        return uniform_looped
    elif name == "uniform_standard":
        return uniform_standard
    elif name == "static_standard":
        return static_standard
    else:
        raise ValueError(f"Unknown context schedule '{name}'")


def get_total_steps(
    scheduler,
    timesteps: List[int],
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
) -> int:
    """Count total window-model calls across all timesteps."""
    return sum(
        len(list(scheduler(i, None, num_frames, context_size, context_stride, context_overlap)))
        for i in range(len(timesteps))
    )


# ---------------------------------------------------------------------------
# Window mask (blending)
# ---------------------------------------------------------------------------

def create_window_mask(
    noise_pred_context,
    c: list[int],
    latent_video_length: int,
    context_overlap: int = 4,
    left_overlap: int | None = None,
    right_overlap: int | None = None,
    looped: bool = False,
    window_type: str = "linear",
):
    """Create a blending mask for a context window.

    The input tensor MUST have the temporal dimension at position 1 (dim=1).
    Pass ``x[:, 0, :tw]`` (4D: ``[B, T, H, W]``) to ensure the ramps are
    applied to the time axis, not channels.

    *left_overlap* / *right_overlap* (if set) override *context_overlap*
    for the corresponding edge.  Use these when the actual overlap with a
    neighbouring window differs from the nominal *context_overlap*.
    """
    length = noise_pred_context.shape[1]
    device = noise_pred_context.device
    lo = left_overlap if left_overlap is not None else context_overlap
    ro = right_overlap if right_overlap is not None else context_overlap

    if window_type == "pyramid":
        half = length // 2
        if length % 2 == 0:
            up = torch.arange(1, half + 1, device=device, dtype=torch.float32)
            down = torch.arange(half, 0, -1, device=device, dtype=torch.float32)
            weights_tensor = torch.cat([up, down]) / half
        else:
            up = torch.arange(1, half + 1, device=device, dtype=torch.float32)
            mid = torch.tensor([half + 1], device=device, dtype=torch.float32)
            down = torch.arange(half, 0, -1, device=device, dtype=torch.float32)
            weights_tensor = torch.cat([up, mid, down]) / (half + 1)

        weights_tensor = weights_tensor.view(1, -1, 1, 1)
        window_mask = weights_tensor.expand(
            noise_pred_context.shape[0], length,
            noise_pred_context.shape[2], noise_pred_context.shape[3],
        ).clone()

        if not looped:
            if min(c) == 0 and lo > 0:
                left_ramp = torch.linspace(0, 1, lo, device=device,
                                           dtype=torch.float32).view(1, -1, 1, 1)
                window_mask[:, :lo] = torch.maximum(window_mask[:, :lo], left_ramp)
            if max(c) == latent_video_length - 1 and ro > 0:
                right_ramp = torch.linspace(1, 0, ro, device=device,
                                            dtype=torch.float32).view(1, -1, 1, 1)
                window_mask[:, -ro:] = torch.maximum(window_mask[:, -ro:], right_ramp)
    else:
        window_mask = torch.ones(
            noise_pred_context.shape, dtype=torch.float32, device=device)
        smooth = (window_type == "smooth")
        if (min(c) > 0 or (looped and max(c) == latent_video_length - 1)) and lo > 0:
            ramp_up = _ramp(lo, device, smooth)
            window_mask[:, :lo] = ramp_up
        if (max(c) < latent_video_length - 1 or (looped and min(c) == 0)) and ro > 0:
            ramp_down = 1.0 - _ramp(ro, device, smooth)
            # Use minimum so short windows (tw < lo+ro) don't lose the
            # left ramp to an overwrite.
            window_mask[:, -ro:] = torch.minimum(window_mask[:, -ro:], ramp_down)

    return window_mask
