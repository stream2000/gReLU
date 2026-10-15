"""Canonical Saijou four-cell-type task order.

Shared by the AlphaGenome/Borzoi training scripts, which feed bigwigs to the
model in this order, and their finetuned ISM adapters, which read prediction
channels back out assuming this same order. A checkpoint's saved model_params
records only the task COUNT (``n_tasks``), never the order, so there is no
value in a checkpoint an adapter could verify this list against -- this
module is the single source of truth for channel identity. Do not redefine
this list elsewhere.
"""

from __future__ import annotations

TASK_NAMES = ["hsc", "mac", "lsec", "chol"]
