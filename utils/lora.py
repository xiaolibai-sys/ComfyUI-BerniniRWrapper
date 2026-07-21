"""
LoRA key standardization and inline merging for Bernini-R / Wan models.

The public ``standardize_lora_keys`` function converts common Wan LoRA key
formats to ComfyUI's expected format.  The new inline merge helpers bypass
ComfyUI's ``load_lora_for_models`` / patcher mechanism and directly fold LoRA
weights into the base state dict, giving Bernini-R full control over when and
how LoRAs are applied.
"""
from __future__ import annotations


import torch
import comfy.utils

from .log import get_logger as _get_logger

logger = _get_logger("LoRA")
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

from .keys import _normalize_unet_key  # noqa: F401  (canonical impl)


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
                logger.debug("Norm diff applied: %s", base_key)
            if diff_b is not None:
                bias_key = base_key[: -len(".weight")] + ".bias"
                if bias_key in base_sd:
                    base_sd[bias_key] = base_sd[bias_key].to(torch.float32) + diff_b.to(torch.float32)
                    logger.debug("Bias diff applied: %s", bias_key)
            continue

        # ── Normal LoRA merge (A/B pair required) ────────────────────────
        if base_key not in base_sd:
            logger.warning(
                "LoRA base key not found in model state dict: %s",
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
                    "LoRA target %s is fp8 but has no weight_scale; skipping",
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
                "Inline merged LoRA: %s (strength=%.3f)",
                lora_path, strength,
            )
        except Exception as e:
            logger.error("Failed to merge LoRA %s: %s", lora_path, e)
            raise
    return base_sd


# ---------------------------------------------------------------------------
# Streaming LoRA folding (used by models.loader's group-wise checkpoint load)
# ---------------------------------------------------------------------------

def _build_streaming_lora_groups(lora_specs):
    """Load all LoRAs and group A/B/alpha by canonical base weight key."""
    if not lora_specs:
        return {}
    groups = {}
    for lora_path, strength in lora_specs:
        if strength == 0.0:
            continue
        try:
            lora_sd = load_lora_state_dict(lora_path)
        except Exception as e:
            logger.error("Failed to load LoRA %s: %s", lora_path, e)
            raise
        per = {}
        for k, v in lora_sd.items():
            if k.endswith(".lora_A.weight"):
                base = k[:-len(".lora_A.weight")] + ".weight"
                per.setdefault(base, {})["A"] = v
            elif k.endswith(".lora_B.weight"):
                base = k[:-len(".lora_B.weight")] + ".weight"
                per.setdefault(base, {})["B"] = v
            elif k.endswith(".alpha"):
                base = k[:-len(".alpha")] + ".weight"
                per.setdefault(base, {})["alpha"] = v
            elif k.endswith(".diff_b"):
                base = k[:-len(".diff_b")] + ".weight"
                per.setdefault(base, {})["diff_b"] = v
            elif k.endswith(".diff"):
                base = k[:-len(".diff")] + ".weight"
                per.setdefault(base, {})["diff"] = v
        for base, parts in per.items():
            if "A" not in parts or "B" not in parts:
                # Norm-only diffs / bias diffs (no A/B pair)
                if parts.get("diff_b") is not None or parts.get("diff") is not None:
                    norm_base = _normalize_unet_key(base)
                    groups.setdefault(norm_base, []).append({
                        "A": None, "B": None, "alpha": None,
                        "diff_b": parts.get("diff_b"),
                        "diff": parts.get("diff"),
                        "strength": float(strength),
                    })
                continue
            norm_base = _normalize_unet_key(base)
            groups.setdefault(norm_base, []).append({
                "A": parts["A"],
                "B": parts["B"],
                "alpha": parts.get("alpha"),
                "diff_b": parts.get("diff_b"),
                "diff": parts.get("diff"),
                "strength": float(strength),
            })
        logger.info("Inline merged LoRA: %s (strength=%.3f)",
                    lora_path, strength)
    return groups


def _collect_block_lora(lora_groups: dict, bidx: int) -> dict | None:
    """Return ``{pname: entries}`` for transformer block *bidx* and remove
    those keys from *lora_groups*.

    This co-locates a block's LoRA with the block itself (``block._lora_payload``)
    and lets the global LoRA pool shrink toward empty as every block is attached —
    the unified (block, lora) → (block) slot design needs no persistent pool.
    """
    if not lora_groups:
        return None
    prefix = f"blocks.{bidx}."
    out = {}
    to_pop = []
    for key, val in lora_groups.items():
        if key.startswith(prefix) and key.endswith(".weight"):
            # Keep the full param name (incl. ".weight") so it matches the
            # slot's ``named_parameters()`` key, e.g. "self_attn.q.weight".
            pname = key[len(prefix):]
            out[pname] = val
            to_pop.append(key)
    for k in to_pop:
        del lora_groups[k]
    return out or None


def _groups_have_work(groups: list) -> bool:
    """Whether any entry carries LoRA (A/B) or DoRA (diff/diff_b) content."""
    for g in groups:
        if g.get("A") is not None and g.get("B") is not None:
            return True
        if g.get("diff_b") is not None or g.get("diff") is not None:
            return True
    return False


def _fold_deltas(base_f: torch.Tensor, groups: list, base_shape) -> torch.Tensor:
    """Apply LoRA/DoRA deltas to a float32 base weight.  Shared core of
    ``_apply_streaming_loras`` and the nvfp4 variant.

    Device of *base_f* owns the computation: LoRA tensors (usually read to
    CPU) are moved there so the fold runs on GPU when the base lives on GPU.
    All moves are no-ops in the legacy all-CPU path.
    """
    dora_diff_b = None
    dora_diff = None
    for g in groups:
        if g.get("diff_b") is not None:
            dora_diff_b = g["diff_b"]
        if g.get("diff") is not None:
            dora_diff = g["diff"]

    dev = base_f.device

    # Capture the ORIGINAL base row/vector norm BEFORE any LoRA delta is applied.
    # DoRA needs the target norm = ||W0|| + diff_b, so we must snapshot ||W0|| now
    # (after the delta loop base_f would already be W0 + Δ).
    if base_f.dim() == 2:
        init_norm = base_f.norm(dim=1, keepdim=True).clamp(min=1e-8)
    else:
        init_norm = None

    # 1. Apply LoRA deltas
    for g in groups:
        if g.get("A") is not None and g.get("B") is not None:
            A, B = g["A"], g["B"]
            if A.device != dev:
                A = A.to(dev)
            if B.device != dev:
                B = B.to(dev)
            delta, _ = _lora_delta(A, B, g.get("alpha"), g["strength"], base_shape)
            base_f = base_f + delta

    # 2. DoRA for 2-D (Linear) weights
    #    W_final = (||W0|| + diff_b) * W_temp / ||W_temp||
    #    where W_temp = W0 + Σ delta, and init_norm is ||W0|| captured above.
    #    NOTE: diff_b is the magnitude *difference* (tiny, ~1e-4), NOT the target
    #    magnitude.  Using m = diff_b alone (without ||W0||) zeroes the weight.
    if dora_diff_b is not None and base_f.dim() == 2:
        temp_norm = base_f.norm(dim=1, keepdim=True).clamp(min=1e-8)
        m = (init_norm + dora_diff_b.to(device=dev, dtype=torch.float32).reshape(-1, 1)).clamp(min=0.0)
        base_f = m * base_f / temp_norm
    else:
        # Norm diff for 1-D weights (handled here only for the LoRA+diff case;
        # pure norm-only groups are applied by _fold_group_loras).
        if dora_diff is not None and base_f.dim() == 1:
            base_f = base_f + dora_diff.to(device=dev, dtype=torch.float32)
    return base_f


def _apply_streaming_loras(base: torch.Tensor, groups: list, scale: torch.Tensor | None = None):
    """Fold pre-grouped LoRAs/DoRAs into a single base weight tensor.

    ``scale`` is the ``weight_scale`` for fp8_scaled (quantized) weights.  When
    provided the base is dequantized (``stored * scale``), the LoRA deltas are
    folded in float32, then the result is re-quantized back to fp8 and returned
    together with a fresh ``scale`` (caller must update the group).  When
    ``scale`` is None the weight is plain bf16/fp16 and the original dtype is kept.

    DoRA support (from ``diff_b`` / ``diff`` in the group dicts):
      - ``diff_b`` on 2-D (Linear) weights: DoRA magnitude delta applied row-wise.
      - ``diff``  on 1-D (norm) weights: direct additive delta to the norm weight.

    Returns ``(weight, scale_or_None)``.
    """
    # Quick return if nothing to do: no groups, or groups with only A=None entries.
    if not _groups_have_work(groups):
        return base, scale

    if scale is not None:
        base_f = base.to(torch.float32) * scale.to(torch.float32)
        is_fp8 = True
    else:
        base_f = base.to(torch.float32)
        is_fp8 = False

    base_f = _fold_deltas(base_f, groups, base.shape)

    if is_fp8:
        return _requantize_fp8(base_f)
    return base_f.to(base.dtype), None


# ---------------------------------------------------------------------------
# nvfp4 folding (dual-scale: per-tensor ``scale`` + per-16-block ``block_scale``)
# ---------------------------------------------------------------------------

def _nvfp4_layout_cls():
    """Resolve the nvfp4 layout class lazily (ComfyUI runtime only)."""
    try:
        from comfy_kitchen.tensor.base import get_layout_class
    except ImportError:
        from comfy.quant_ops import get_layout_class
    return get_layout_class("TensorCoreNVFP4Layout")


def _apply_streaming_loras_nvfp4(qdata, scale, block_scale, orig_dtype,
                                 orig_shape, meta, groups):
    """Fold pre-grouped LoRAs/DoRAs into an nvfp4-packed 2-D weight.

    Dequantizes via the layout class (unpack fp4 × block_scale × scale), folds
    deltas in float32, then re-quantizes to nvfp4 with fresh dual scales.
    Returns ``(qdata, scale, block_scale)`` — all new tensors on *qdata*'s
    device.  When *groups* has no LoRA/DoRA content the inputs are returned
    unchanged (re-quantization is lossy, so untouched tensors are never
    round-tripped).
    """
    if not _groups_have_work(groups):
        return qdata, scale, block_scale
    # orig_dtype may be None when the QuantizedTensor was materialised from a
    # checkpoint (ComfyUI leaves it unset); bf16 is the Bernini-R/Wan compute
    # dtype and the de-facto pre-quant dtype of these checkpoints.
    if orig_dtype is None:
        orig_dtype = torch.bfloat16
    L = _nvfp4_layout_cls()
    params = L.Params(scale=scale, orig_dtype=orig_dtype,
                      orig_shape=orig_shape, block_scale=block_scale,
                      **(meta or {}))
    base_f = L.dequantize(qdata, params).to(torch.float32)
    base_f = _fold_deltas(base_f, groups, orig_shape)
    # The nvfp4 quantize kernel only accepts FP16/BF16 input — cast the
    # folded fp32 result back to the original storage dtype first.
    new_q, new_params = L.quantize(base_f.to(orig_dtype))
    return new_q, new_params.scale, new_params.block_scale


def _rebuild_params(params, **replacements):
    """Rebuild a frozen layout ``Params`` dataclass with fields replaced."""
    import dataclasses
    kwargs = {f.name: getattr(params, f.name) for f in dataclasses.fields(params)}
    kwargs.update(replacements)
    return type(params)(**kwargs)


def _replace_quant_param(owner: torch.nn.Module, param_name: str,
                         qdata, layout_cls: str, params) -> None:
    """Swap ``owner``'s quantized parameter for a fresh QuantizedTensor.

    Required instead of writing ``param.data._params.<field>`` in place:
    ``param.data`` on a QuantizedTensor returns an ephemeral wrapper whose
    scale tensors are *copies* — in-place writes there are silently lost.
    """
    try:
        from comfy_kitchen.tensor.base import QuantizedTensor
    except ImportError:
        from comfy.quant_ops import QuantizedTensor
    qt = QuantizedTensor(qdata, layout_cls, params)
    owner._parameters[param_name] = torch.nn.Parameter(qt, requires_grad=False)


def _fold_module_loras(module: torch.nn.Module, prefix: str, lora_groups: dict | None) -> int:
    """Fold LoRAs whose base key lives under *prefix* into *module*'s parameters
    in place, on the device the parameters already live on.

    Companion to ``_fold_group_loras`` (which folds a CPU tensor dict *before*
    ``load_state_dict``).  This variant runs *after* the weights have landed
    in the module, so when the module is GPU-resident (plain, non-block-swap
    mode) the fp32 fold math runs on GPU — orders of magnitude faster than
    the CPU dict fold for full-coverage LoRAs on large models.

    Handles both regular params and fp8 ``QuantizedTensor`` params (dequant →
    fp32 fold → requant with a fresh scale, mirroring ``_fold_lora_on_entries``).
    Norm-only entries: ``diff`` → 1-D norm weight, ``diff_b`` → bias of the
    same owner.  Returns the number of folded weight targets.
    """
    if not lora_groups:
        return 0
    folded = 0
    # Params are inference tensors (model built under InferenceMode) — wrap
    # all writes, same convention as _stream_load_group.
    with torch.inference_mode():
        for full_key, g_list in lora_groups.items():
            if not full_key.startswith(prefix):
                continue
            local = full_key[len(prefix):]  # e.g. "self_attn.q.weight"
            if not local.endswith(".weight"):
                continue
            try:
                owner = module.get_submodule(local[: -len(".weight")])
                param = owner.weight
            except (AttributeError, KeyError):
                continue
            if param is None:
                continue

            lora_entries = [g for g in g_list
                            if g.get("A") is not None and g.get("B") is not None]
            norm_only = [g for g in g_list
                         if not (g.get("A") is not None and g.get("B") is not None)]

            d = param.data
            if lora_entries:
                # The Parameter itself is the QuantizedTensor — ``param._params``
                # is canonical; ``param.data._params`` is an ephemeral copy.
                qt = param if hasattr(param, "_qdata") else (
                    d if hasattr(d, "_qdata") else None)
                if qt is not None:
                    pr = qt._params
                    if "NVFP4" in getattr(qt, "_layout_cls", ""):
                        meta = {}
                        if hasattr(pr, "transposed"):
                            meta["transposed"] = pr.transposed
                        new_q, new_scale, new_bs = _apply_streaming_loras_nvfp4(
                            qt._qdata, pr.scale, pr.block_scale,
                            pr.orig_dtype, pr.orig_shape, meta, lora_entries)
                        qt._qdata.copy_(new_q)  # shared storage — persists
                        new_pr = _rebuild_params(
                            pr, scale=new_scale, block_scale=new_bs)
                        _replace_quant_param(
                            owner, "weight", qt._qdata, qt._layout_cls, new_pr)
                    else:
                        new_q, new_scale = _apply_streaming_loras(
                            qt._qdata, lora_entries, pr.scale)
                        qt._qdata.copy_(new_q)  # shared storage — persists
                        new_pr = _rebuild_params(pr, scale=new_scale)
                        _replace_quant_param(
                            owner, "weight", qt._qdata, qt._layout_cls, new_pr)
                else:
                    new_w, _ = _apply_streaming_loras(d, lora_entries, None)
                    d.copy_(new_w)
                folded += 1

            for g in norm_only:
                if g.get("diff") is not None and d.dim() == 1:
                    merged = d.to(torch.float32) + g["diff"].to(
                        device=d.device, dtype=torch.float32)
                    d.copy_(merged.to(d.dtype))
                if g.get("diff_b") is not None:
                    bias = getattr(owner, "bias", None)
                    if bias is not None and bias.data is not None:
                        b = bias.data
                        merged_b = b.to(torch.float32) + g["diff_b"].to(
                            device=b.device, dtype=torch.float32)
                        b.copy_(merged_b.to(b.dtype))
    return folded


def _fold_group_loras(sub_group: dict, prefix: str, lora_groups: dict | None):
    """Fold LoRAs whose base key lives under *prefix* into *sub_group* in place.

    Used by the eager streaming loader so block-swap can fold LoRA at the
    exact moment a block group is read from disk.
    """
    if not lora_groups:
        return

    def _cast_back(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # Norm weights/biases in this model are fp16 (not fp8), so a plain
        # cast back to the original dtype is sufficient.  The fp8 branch is
        # kept only for theoretical fp8-scaled norms.
        if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            return _requantize_fp8(t.to(torch.float32))[0]
        return t.to(dtype)

    for full_key, g_list in lora_groups.items():
        if not full_key.startswith(prefix):
            continue
        local = full_key[len(prefix):]  # e.g. "self_attn.q.weight"
        if local not in sub_group:
            continue
        # Split into LoRA entries (have A & B) and norm-only entries (no A/B).
        lora_entries = [g for g in g_list
                        if g.get("A") is not None and g.get("B") is not None]
        norm_only = [g for g in g_list
                     if not (g.get("A") is not None and g.get("B") is not None)]

        if lora_entries:
            # sub_group keys keep the full tensor name (e.g. "self_attn.q.weight_scale"),
            # so the scale key is local + "_scale" (local already ends in ".weight").
            scale = sub_group.get(local + "_scale")
            new_w, new_scale = _apply_streaming_loras(sub_group[local], lora_entries, scale)
            sub_group[local] = new_w
            if new_scale is not None:
                sub_group[local + "_scale"] = new_scale

        # Norm-only groups: mirror merge_lora_into_state_dict's pure
        # norm-diff / bias-diff branch.  diff -> norm weight (1-D);
        # diff_b -> corresponding bias (1-D).
        for g in norm_only:
            tgt = sub_group[local]
            if g.get("diff") is not None and tgt.dim() == 1:
                sub_group[local] = _cast_back(
                    tgt.to(torch.float32) + g["diff"].to(torch.float32), tgt.dtype)
            if g.get("diff_b") is not None:
                bias_local = local[: -len(".weight")] + ".bias"
                if bias_local in sub_group:
                    bt = sub_group[bias_local]
                    sub_group[bias_local] = _cast_back(
                        bt.to(torch.float32) + g["diff_b"].to(torch.float32), bt.dtype)
