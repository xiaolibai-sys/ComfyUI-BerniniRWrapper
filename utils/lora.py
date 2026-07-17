"""
LoRA key standardization and inline merging for Bernini-R / Wan models.

The public ``standardize_lora_keys`` function converts common Wan LoRA key
formats to ComfyUI's expected format.  The new inline merge helpers bypass
ComfyUI's ``load_lora_for_models`` / patcher mechanism and directly fold LoRA
weights into the base state dict, giving Bernini-R full control over when and
how LoRAs are applied.
"""
from __future__ import annotations

import logging

import torch
import comfy.utils

logger = logging.getLogger(__name__)


def standardize_lora_keys(lora_sd: dict) -> dict:
    """Convert common Wan LoRA key formats to ComfyUI's expected format.

    Supported formats:
      - ComfyUI native: ``diffusion_model.blocks.0.self_attn.q.lora_A.weight``
      - Plain blocks:   ``blocks.0.self_attn.q.lora_A.weight``
      - Diffusers:      ``transformer.blocks.0.self_attn.q.lora_A.weight``
      - Kohya/AI-Toolkit: ``lora_unet_blocks_0_self_attn_q.lora_A.weight``
      - LyCORIS:        ``lycoris_blocks_...``
      - Fun LoRA:       ``lora_unet__blocks_0_self_attn_q.lora_A.weight``
    """
    new_sd = {}
    for k, v in lora_sd.items():
        k = _standardize_key(k)
        new_sd[k] = v
    return new_sd


def _standardize_key(k: str) -> str:
    """Normalize a single LoRA key."""
    # ── LyCORIS / AI-Toolkit underscore format ─────────────────────────
    if k.startswith("lycoris_blocks_"):
        k = k.replace("lycoris_blocks_", "blocks.")
        k = k.replace("_cross_attn_", ".cross_attn.")
        k = k.replace("_self_attn_", ".self_attn.")
        k = k.replace("_ffn_net_0_proj", ".ffn.0")
        k = k.replace("_ffn_net_2", ".ffn.2")
        k = k.replace("to_out_0", "o")

    # ── Common prefixes ────────────────────────────────────────────────
    if k.startswith("transformer."):
        k = k.replace("transformer.", "diffusion_model.", 1)
    if k.startswith("pipe.dit."):
        k = k.replace("pipe.dit.", "diffusion_model.", 1)
    if k.startswith("base_model.model."):
        k = k.replace("base_model.model.", "diffusion_model.", 1)
    if k.startswith("blocks."):
        k = k.replace("blocks.", "diffusion_model.blocks.", 1)
    if k.startswith("vace_blocks."):
        k = k.replace("vace_blocks.", "diffusion_model.vace_blocks.", 1)

    # ── Fun LoRA format: lora_unet__blocks_0_self_attn_q.lora_A.weight ─
    if k.startswith("lora_unet__"):
        k = _convert_fun_lora_key(k)

    # ── Kohya / AI-Toolkit lora_unet_... format ────────────────────────
    if k.startswith("lora_unet_") and not k.startswith("lora_unet__"):
        # ``lora_unet_blocks_0_self_attn_q.lora_A.weight``
        # Strip prefix and replace underscores with dots, but be careful with
        # numeric block indices and component names.
        body = k[len("lora_unet_"):]
        parts = body.split(".")
        main_part = parts[0]
        weight_part = ".".join(parts[1:]) if len(parts) > 1 else ""

        # Convert underscore path to dot path.
        tokens = main_part.split("_")
        path_tokens = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == "blocks" and i + 1 < len(tokens) and tokens[i + 1].isdigit():
                path_tokens.extend(["blocks", tokens[i + 1]])
                i += 2
                continue
            # Map attention components.
            if t in ("self", "cross") and i + 1 < len(tokens) and tokens[i + 1] == "attn":
                path_tokens.append(f"{t}_attn")
                i += 2
                continue
            if t in ("q", "k", "v", "o"):
                path_tokens.append(t)
                i += 1
                continue
            if t == "ffn" and i + 1 < len(tokens) and tokens[i + 1].isdigit():
                path_tokens.extend(["ffn", tokens[i + 1]])
                i += 2
                continue
            if t == "to" and i + 1 < len(tokens) and tokens[i + 1] == "out":
                path_tokens.append("o")
                i += 2
                continue
            path_tokens.append(t)
            i += 1

        new_main = "diffusion_model." + ".".join(path_tokens)
        if weight_part:
            if weight_part in ("lora_down.weight", "lora_down"):
                weight_part = "lora_A.weight"
            elif weight_part in ("lora_up.weight", "lora_up"):
                weight_part = "lora_B.weight"
            elif weight_part == "alpha":
                weight_part = "alpha"
            k = f"{new_main}.{weight_part}" if weight_part != "alpha" else f"{new_main}.alpha"
        else:
            k = new_main

    # ── Finetrainer attention naming ───────────────────────────────────
    if ".attn1." in k:
        k = k.replace(".attn1.", ".cross_attn.")
        k = k.replace(".to_k.", ".k.")
        k = k.replace(".to_q.", ".q.")
        k = k.replace(".to_v.", ".v.")
        k = k.replace(".to_out.0.", ".o.")
    elif ".attn2." in k:
        k = k.replace(".attn2.", ".cross_attn.")
        k = k.replace(".to_k.", ".k.")
        k = k.replace(".to_q.", ".q.")
        k = k.replace(".to_v.", ".v.")
        k = k.replace(".to_out.0.", ".o.")

    # ── Misc cleanups ──────────────────────────────────────────────────
    k = k.replace(".default.", ".")
    k = k.replace(".diff_m", ".modulation.diff")

    # Fix ``diffusion.model`` → ``diffusion_model``
    if k.startswith("diffusion.model."):
        k = k.replace("diffusion.model.", "diffusion_model.", 1)

    # ── Generic Kohya diffusers naming: lora_down/lora_up -> lora_A/lora_B ──
    # This MUST come after all prefix conversions so that keys already under
    # ``diffusion_model.blocks.N.<mod>.lora_down.weight`` are caught.
    if k.endswith(".lora_down.weight"):
        k = k[: -len(".lora_down.weight")] + ".lora_A.weight"
    elif k.endswith(".lora_up.weight"):
        k = k[: -len(".lora_up.weight")] + ".lora_B.weight"

    return k


def _convert_fun_lora_key(k: str) -> str:
    """Convert ``lora_unet__blocks_0_self_attn_q.lora_A.weight`` style keys."""
    parts = k.split(".")
    main_part = parts[0]  # e.g. lora_unet__blocks_0_self_attn_q
    weight_part = ".".join(parts[1:]) if len(parts) > 1 else ""

    if "blocks_" not in main_part:
        # Fallback: replace prefix and underscores globally.
        new_key = main_part.replace("lora_unet__", "diffusion_model.")
        new_key = new_key.replace("_self_attn", ".self_attn")
        new_key = new_key.replace("_cross_attn", ".cross_attn")
        new_key = new_key.replace("_ffn", ".ffn")
        new_key = new_key.replace("blocks_", "blocks.")
        new_key = new_key.replace("head_head", "head.head")
    else:
        components = main_part[len("lora_unet__"):].split("_")
        new_key = "diffusion_model"

        i = 0
        if components[i] == "blocks":
            new_key += f".blocks.{components[i + 1]}"
            i += 2

        while i < len(components):
            t = components[i]
            if t == "self" and i + 1 < len(components) and components[i + 1] == "attn":
                new_key += ".self_attn"
                i += 2
            elif t == "cross" and i + 1 < len(components) and components[i + 1] == "attn":
                new_key += ".cross_attn"
                i += 2
            elif t == "ffn":
                new_key += ".ffn"
                i += 1
            elif t in ("q", "k", "v", "o"):
                new_key += f".{t}"
                i += 1
            elif t == "img":
                # Append _img to previous component.
                new_key += "_img"
                i += 1
            else:
                new_key += f".{t}"
                i += 1

    # Convert weight suffix.
    if weight_part:
        if weight_part in ("lora_down.weight", "lora_down"):
            weight_part = "lora_A.weight"
        elif weight_part in ("lora_up.weight", "lora_up"):
            weight_part = "lora_B.weight"
        elif weight_part == "alpha":
            return f"{new_key}.alpha"
        return f"{new_key}.{weight_part}"
    return new_key

# ---------------------------------------------------------------------------
# Key prefix normalization (used by inline merge to align lora ↔ base keys)
# ---------------------------------------------------------------------------

def _normalize_unet_key(k: str) -> str:
    """Strip common model-prefixes so lora keys match base state-dict keys.

    Matches the streaming path's ``_normalize_unet_key`` in ``wan_model.py``.
    """
    for prefix in ("model.diffusion_model.", "diffusion_model.", "model.", "video_model."):
        if k.startswith(prefix):
            return k[len(prefix):]
    return k


# ---------------------------------------------------------------------------
# Inline LoRA/DoRA merge (bypasses ComfyUI's load_lora_for_models)
# ---------------------------------------------------------------------------

def load_lora_state_dict(path: str) -> dict:
    """Load a LoRA checkpoint (safetensors or pt) and standardize its keys."""
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    return standardize_lora_keys(sd)


def _lora_delta(A: torch.Tensor, B: torch.Tensor, alpha, strength: float, base_shape=None):
    """Compute the LoRA delta ``strength * (alpha/rank) * (B @ A)`` in float32.

    Handles both Diffusers (``lora_A=[rank,in]``, ``lora_B=[out,rank]``) and
    Kohya (``lora_down=[in,rank]`` -> ``lora_A``, ``lora_up=[rank,out]`` ->
    ``lora_B``) orientations.  When ``base_shape`` is supplied the two possible
    orientations are validated against the target ``[out, in]`` shape, which is
    required for square weight matrices (e.g. Wan self_attn q/k/v/o are
    ``[dim, dim]``) where rank-based heuristics are ambiguous.
    """
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    if base_shape is not None and len(base_shape) == 2:
        out, in_ = base_shape[0], base_shape[1]
        if B.shape[1] == A.shape[0] and (B @ A).shape == (out, in_):
            delta = B @ A
            rank = A.shape[0]
        elif A.shape[1] == B.shape[0] and (A @ B).T.shape == (out, in_):
            delta = (A @ B).T
            rank = A.shape[1]
        else:
            # Fallback: assume Diffusers-style B @ A.
            delta = B @ A
            rank = A.shape[0]
    else:
        if A.shape[1] == B.shape[0]:
            delta = (A @ B).T
            rank = A.shape[1]
        else:
            delta = B @ A
            rank = A.shape[0]
    if alpha is not None:
        try:
            alpha_val = float(alpha.item() if alpha.numel() == 1 else alpha)
        except Exception:
            alpha_val = float(rank)
    else:
        alpha_val = float(rank)
    return delta * (strength * (alpha_val / rank)), rank


def _requantize_fp8(tensor: torch.Tensor):
    """Re-quantize a float32 weight back to fp8_e4m3fn with a fresh per-tensor scale."""
    maxv = torch.finfo(torch.float8_e4m3fn).max
    scale = tensor.abs().amax() / maxv
    scale = scale.clamp(min=torch.finfo(torch.float32).tiny)
    if isinstance(scale, torch.Tensor):
        scale = scale.reshape(())
    else:
        scale = torch.tensor(scale, dtype=torch.float32)
    q = (tensor / scale).clamp(-maxv, maxv).to(torch.float8_e4m3fn)
    return q, scale


def merge_lora_into_state_dict(
    base_sd: dict,
    lora_sd: dict,
    strength: float = 1.0,
) -> dict:
    """Fold a single LoRA/DoRA directly into *base_sd* without using ComfyUI patches.

    For each LoRA key group ``diffusion_model.blocks.N.<module>.lora_{A,B}.weight``
    the delta is computed as ``scale * (B @ A)`` and added to the corresponding
    base weight.

    **DoRA** (Weight-Decomposed Low-Rank Adaptation) support:

    - ``.diff_b`` on 2-D (Linear) weights: DoRA magnitude delta.
      ``W_final = (||W0|| + diff_b) * W_temp / ||W_temp||`` (row-wise).
    - ``.diff_b`` on 1-D weights (LayerNorms that have ``.bias``): bias delta.
    - ``.diff`` on 1-D (norm) weights: direct additive delta to norm weight.

    The optional ``.alpha`` scalar is used to compute ``scale = strength * alpha / rank``,
    matching ComfyUI's ``comfy.lora.calculate_weight`` behaviour.

    fp8_scaled (quantized) base weights are handled correctly: the stored fp8
    value is dequantized with its ``weight_scale`` (``stored * scale``), the
    delta is folded in float32, then the result is re-quantized back to fp8 with
    a fresh ``weight_scale``.  Plain bf16/fp16 weights are merged in place as before.

    Parameters
    ----------
    base_sd:
        Base model state dict (mutated in place).
    lora_sd:
        Standardized LoRA state dict.
    strength:
        LoRA strength.  ``0.0`` is a no-op.

    Returns
    -------
    The mutated *base_sd*.
    """
    if strength == 0.0:
        return base_sd

    # Group A/B/alpha/diff_b/diff by the corresponding base weight key.
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in lora_sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.endswith(".lora_A.weight"):
            base_key = k[: -len(".lora_A.weight")] + ".weight"
            groups.setdefault(base_key, {})["A"] = v
        elif k.endswith(".lora_B.weight"):
            base_key = k[: -len(".lora_B.weight")] + ".weight"
            groups.setdefault(base_key, {})["B"] = v
        elif k.endswith(".alpha"):
            base_key = k[: -len(".alpha")] + ".weight"
            groups.setdefault(base_key, {})["alpha"] = v
        elif k.endswith(".diff_b"):
            base_key = k[: -len(".diff_b")] + ".weight"
            groups.setdefault(base_key, {})["diff_b"] = v
        elif k.endswith(".diff"):  # DoRA magnitude for norm layers
            base_key = k[: -len(".diff")] + ".weight"
            groups.setdefault(base_key, {})["diff"] = v

    for lora_base_key, parts in groups.items():
        # Normalize the lora key prefix so it matches base_sd.
        base_key = _normalize_unet_key(lora_base_key)
        A = parts.get("A")
        B = parts.get("B")
        diff_b = parts.get("diff_b")
        diff = parts.get("diff")
        has_lora = (A is not None and B is not None)
        has_do_dora = (diff_b is not None or diff is not None)

        # Skip entirely if nothing to do for this key.
        if not has_lora and not has_do_dora:
            continue

        # ── Handle pure norm-diff / bias-diff (no A/B pair) ──────────────
        if not has_lora and has_do_dora:
            if base_key in base_sd and diff is not None and base_sd[base_key].dim() == 1:
                base_sd[base_key] = base_sd[base_key].to(torch.float32) + diff.to(torch.float32)
                logger.debug("[BerniniR] Norm diff applied: %s", base_key)
            if diff_b is not None:
                bias_key = base_key[: -len(".weight")] + ".bias"
                if bias_key in base_sd:
                    base_sd[bias_key] = base_sd[bias_key].to(torch.float32) + diff_b.to(torch.float32)
                    logger.debug("[BerniniR] Bias diff applied: %s", bias_key)
            continue

        # ── Normal LoRA merge (A/B pair required) ────────────────────────
        if base_key not in base_sd:
            logger.warning(
                "[BerniniR] LoRA base key not found in model state dict: %s",
                base_key,
            )
            continue

        base_weight = base_sd[base_key]
        is_fp8 = base_weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        if is_fp8:
            scale_key = base_key[: -len(".weight")] + ".weight_scale"
            scale = base_sd.get(scale_key)
            if scale is None:
                logger.warning(
                    "[BerniniR] LoRA target %s is fp8 but has no weight_scale; skipping",
                    base_key,
                )
                continue
            base_f = base_weight.to(torch.float32) * scale.to(torch.float32)
        else:
            base_f = base_weight.to(torch.float32)

        # 1. LoRA delta
        delta, _ = _lora_delta(A, B, parts.get("alpha"), strength, base_weight.shape)
        w_temp = base_f + delta

        # 2. DoRA for 2-D (Linear) weights
        if diff_b is not None and base_weight.dim() == 2:
            # W_final = (||W0|| + diff_b) * W_temp / ||W_temp||
            init_norm = base_f.norm(dim=1, keepdim=True).clamp(min=1e-8)
            temp_norm = w_temp.norm(dim=1, keepdim=True).clamp(min=1e-8)
            m = init_norm + diff_b.to(torch.float32).reshape(-1, 1).clamp(min=0.0)
            base_f = m * w_temp / temp_norm
        else:
            base_f = w_temp

        # 3. Norm diff for 1-D weights
        if diff is not None and base_weight.dim() == 1:
            base_f = base_f + diff.to(torch.float32)

        if is_fp8:
            new_w, new_scale = _requantize_fp8(base_f)
            base_sd[base_key] = new_w
            base_sd[scale_key] = new_scale
        else:
            base_sd[base_key] = base_f.to(base_weight.dtype)

    return base_sd


def apply_loras_to_state_dict(
    base_sd: dict,
    lora_specs: list[tuple[str, float]],
) -> dict:
    """Sequentially merge multiple LoRAs into *base_sd*.

    Each spec is ``(lora_path, strength)``.  LoRAs are applied in order.
    """
    for lora_path, strength in lora_specs:
        if strength == 0.0:
            continue
        try:
            lora_sd = load_lora_state_dict(lora_path)
            merge_lora_into_state_dict(base_sd, lora_sd, strength=strength)
            logger.info(
                "[BerniniR] Inline merged LoRA: %s (strength=%.3f)",
                lora_path, strength,
            )
        except Exception as e:
            logger.error("[BerniniR] Failed to merge LoRA %s: %s", lora_path, e)
            raise
    return base_sd
