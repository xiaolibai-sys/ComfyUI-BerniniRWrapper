/**
 * Bernini-R frontend extension: dynamically show/hide widgets.
 *
 * DOM-based visibility toggle.  On node creation / workflow load / dropdown
 * change we find the rendered DOM elements and set ``display:none``.
 *
 * On initial page load (workflow restore) Vue may not have rendered the
 * node DOM yet, so we poll with ``requestAnimationFrame`` until the node
 * element appears, then apply visibility.
 */

import { app } from "../../scripts/app.js";

const SAMPLER_NODES = ["BerniniR_KSampler", "BerniniR_KSamplerTeaCache"];
const APG_WIDGETS = ["apg_eta", "apg_rescale", "apg_momentum"];
const RAAG_WIDGETS = ["raag_alpha"];
const S2_WIDGETS = ["s2_omega"];
const STG_WIDGETS = ["stg_scale", "stg_block_idx"];

const SCHEDULE_NODE = "BerniniR_GuidanceStrengthSchedule";
const CURVE_SPECIFIC_WIDGETS = ["hold_start", "hold_end", "guidance_mid", "transition"];
const CURVE_VISIBILITY = {
    cosine:    { hold_start: true,  hold_end: true,  guidance_mid: false, transition: false },
    linear:    { hold_start: false, hold_end: false, guidance_mid: false, transition: false },
    piecewise: { hold_start: false, hold_end: false, guidance_mid: true,  transition: true },
};

// ── DOM helpers ───────────────────────────────────────────────────

function _getNodeEl(nodeId) {
    return document.querySelector(`[data-node-id="${nodeId}"]`)
        || document.getElementById(String(nodeId));
}

function _setWidgetVisible(nodeId, widgetName, visible) {
    const nodeEl = _getNodeEl(nodeId);
    if (!nodeEl) return false;

    let found = false;
    const display = visible ? "" : "none";

    // Match the input element by data-property attribute
    const inputEl = nodeEl.querySelector(`[data-property="${widgetName}"]`);
    if (inputEl) {
        inputEl.style.display = display;
        const label = inputEl.previousElementSibling;
        if (label) label.style.display = display;
        found = true;
    }

    // Fallback: match by text content (label matching)
    for (const el of nodeEl.querySelectorAll("label, span, div, p")) {
        if ((el.textContent || "").trim() === widgetName) {
            el.style.display = display;
            const next = el.nextElementSibling;
            if (next) next.style.display = display;
            found = true;
        }
    }
    return found;
}

function _applyVisibility(node, widgetNames, visible) {
    for (const name of widgetNames) {
        _setWidgetVisible(node.id, name, visible);
    }
}

// ── Poll until DOM ready, then apply ─────────────────────────────

function _whenDOMReady(node, fn, maxFrames = 120) {
    let attempts = 0;
    function poll() {
        if (_getNodeEl(node.id)) {
            fn();
            return;
        }
        if (++attempts < maxFrames) {
            requestAnimationFrame(poll);
        }
    }
    requestAnimationFrame(poll);
}

// ── Sampler visibility ───────────────────────────────────────────

function _samplerVis(node) {
    const m = (node.widgets || []).find(w => w.name === "guidance_mode");
    if (!m) return;
    const mode = m.value;
    _applyVisibility(node, APG_WIDGETS, mode === "APG");
    _applyVisibility(node, RAAG_WIDGETS, mode === "RAAG");
    _applyVisibility(node, S2_WIDGETS, mode === "S2");
    _applyVisibility(node, STG_WIDGETS, mode.startsWith("STG"));
}

function registerSamplerNode(node) {
    const m = (node.widgets || []).find(w => w.name === "guidance_mode");
    if (!m) return;

    const origCB = m.callback;
    m.callback = function (v, ...a) {
        const r = origCB?.call(this, v, ...a);
        _samplerVis(node);
        return r;
    };

    // On workflow load, widget values are restored during onConfigure.
    // Vue renders the DOM after onConfigure, so poll for it.
    const origCfg = node.onConfigure;
    node.onConfigure = function (...a) {
        const r = origCfg?.apply(this, a);
        _whenDOMReady(node, () => _samplerVis(node));
        return r;
    };

    // If DOM already exists (live edit), apply immediately
    _samplerVis(node);
}

// ── Schedule visibility ──────────────────────────────────────────

function _curveVis(node) {
    const c = (node.widgets || []).find(w => w.name === "curve");
    if (!c) return;
    const map = CURVE_VISIBILITY[c.value] || CURVE_VISIBILITY.linear;
    for (const name of CURVE_SPECIFIC_WIDGETS) {
        _setWidgetVisible(node.id, name, Boolean(map[name]));
    }
}

function registerScheduleNode(node) {
    const c = (node.widgets || []).find(w => w.name === "curve");
    if (!c) return;

    const origCB = c.callback;
    c.callback = function (v, ...a) {
        const r = origCB?.call(this, v, ...a);
        _curveVis(node);
        return r;
    };

    const origCfg = node.onConfigure;
    node.onConfigure = function (...a) {
        const r = origCfg?.apply(this, a);
        _whenDOMReady(node, () => _curveVis(node));
        return r;
    };

    _curveVis(node);
}

// ── Extension ─────────────────────────────────────────────────────

app.registerExtension({
    name: "BerniniR.GuidanceUI",
    async nodeCreated(node) {
        if (SAMPLER_NODES.includes(node.comfyClass)) {
            registerSamplerNode(node);
        } else if (node.comfyClass === SCHEDULE_NODE) {
            registerScheduleNode(node);
        }
    },
});
