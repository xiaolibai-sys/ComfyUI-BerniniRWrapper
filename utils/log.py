"""Global logging for Bernini-R.

One package-level logger (``BerniniR``) with a single handler and a uniform
``[BerniniR][Tag] message`` format.  Modules call ``get_logger(tag)`` instead
of ``logging.getLogger(__name__)`` so every line carries a stable subsystem
tag and obeys one level control.

Level: env ``BERNINI_LOG_LEVEL`` = DEBUG/INFO/WARNING/ERROR (default INFO).
The logger has its own handler and ``propagate = False``, so lines are never
duplicated by ComfyUI's root logging configuration.
"""

from __future__ import annotations

import logging
import os

_NAME = "BerniniR"


def _env_level() -> int:
    raw = os.environ.get("BERNINI_LOG_LEVEL", "INFO").upper()
    return getattr(logging, raw, logging.INFO)


def _build_root() -> logging.Logger:
    logger = logging.getLogger(_NAME)
    if not any(getattr(h, "_bernini_r", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler._bernini_r = True  # mark so re-imports don't stack handlers
        handler.setFormatter(
            logging.Formatter("[BerniniR][%(tag)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(_env_level())
    logger.propagate = False
    return logger


_ROOT = _build_root()


class _TagAdapter(logging.LoggerAdapter):
    """Injects the subsystem tag into every record."""

    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        extra["tag"] = self.extra["tag"]
        return msg, kwargs


def get_logger(tag: str) -> logging.Logger:
    """Return a logger prefixing every record with ``[BerniniR][tag]``."""
    return _TagAdapter(_ROOT, {"tag": tag})


def set_level(level) -> None:
    """Change the package log level at runtime (int or level name)."""
    _ROOT.setLevel(level)
