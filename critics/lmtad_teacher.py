"""
LM-TAD Teacher Wrapper
======================

Purpose
-------
Provide a small, readable wrapper around the external LM-TAD codebase to:
- Load a trained LM-TAD checkpoint
- Expose next-token distribution for a given history (grid tokens)
- Handle dtype/AMP context and minor checkpoint quirks

Notes
-----
- This module intentionally lives outside the core model code to keep
  the original training script untouched unless distillation is enabled.
- It assumes the LM-TAD repo is available locally and provides the
  same import structure as used in eval_porto.py (models, datasets).
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from typing import Optional, Tuple

import torch


class LMTADTeacher:
    """Thin wrapper to load and query a trained LM-TAD model.

    Parameters
    ----------
    repo_path: str
        Path to the LM-TAD repository root (expects a `code/` subfolder).
    ckpt_path: str
        Path to the LM-TAD checkpoint (.pt) produced by train_LMTAD.py.
    device: str
        Torch device string, e.g. 'cuda:0' or 'cpu'.
    dtype: str
        One of {'float16','bfloat16','float32'} for AMP context.
    window: int
        Max history length to feed the teacher (truncate from the left).
    """

    def __init__(
        self,
        repo_path: str,
        ckpt_path: str,
        device: str,
        dtype: str = "float16",
        window: int = 64,
    ) -> None:
        self.repo_path = repo_path
        self.ckpt_path = ckpt_path
        self.device = device
        self.window = int(window)

        # AMP precision per LMTAD defaults
        if dtype not in {"float16", "bfloat16", "float32"}:
            dtype = "float16"
        self._ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]

        # Add LM-TAD repo to sys.path and import lazily
        code_path = f"{self.repo_path}/code"
        if code_path not in sys.path:
            sys.path.insert(0, code_path)

        # Robust import of LM-TAD modules without colliding with local `models` package
        import importlib.util
        import os as _os

        def _load_module(module_name: str, module_path: str):
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module {module_name} from {module_path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            return mod

        # Load model directly from models/LMTAD.py and datasets from code/datasets.py
        lmtad_model_py = _os.path.join(code_path, 'models', 'LMTAD.py')
        lmtad_utils_py = _os.path.join(code_path, 'utils.py')
        # We do not need datasets for distillation; skip importing datasets.py
        if not _os.path.exists(lmtad_model_py):
            raise ImportError(f"LM-TAD repo missing {lmtad_model_py}")
        # Ensure LM-TAD's 'utils' is used by model code
        import sys as _sys
        prev_utils = _sys.modules.get('utils')
        try:
            if _os.path.exists(lmtad_utils_py):
                lmtad_utils = _load_module('lmtad_utils', lmtad_utils_py)
                _sys.modules['utils'] = lmtad_utils
            lmtad_models = _load_module('lmtad_models', lmtad_model_py)
        finally:
            # Do not leave a broken state; restore previous utils if any
            if prev_utils is not None:
                _sys.modules['utils'] = prev_utils
            else:
                _sys.modules.pop('utils', None)
        LMTAD = getattr(lmtad_models, 'LMTAD', None)
        if LMTAD is None:
            raise ImportError("LMTAD class not found in LM-TAD model file")

        # Prefer weights-only checkpoints: {'state_dict','model_config' (plain dict)}
        try:
            checkpoint = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.ckpt_path, map_location=self.device)

        state_dict = checkpoint.get("state_dict") or checkpoint.get("model")
        if state_dict is None:
            raise RuntimeError("Checkpoint missing 'state_dict'/'model'")
        model_conf = checkpoint.get("model_config")
        if model_conf is None:
            raise RuntimeError("Checkpoint missing 'model_config'")

        # If config is a dataclass-like object, turn into plain dict
        if hasattr(model_conf, "__dict__") and not isinstance(model_conf, dict):
            model_conf = dict(model_conf.__dict__)

        # Fallbacks: infer vocab_size and block_size from weights if missing or invalid
        if not model_conf.get("vocab_size"):
            emb_weight = state_dict.get("transformer.wte.weight")
            if emb_weight is not None:
                model_conf["vocab_size"] = emb_weight.shape[0]
        block_size_val = model_conf.get("block_size")
        if block_size_val is None or block_size_val <= 0:
            wpe_weight = state_dict.get("transformer.wpe.weight")
            if wpe_weight is not None:
                model_conf["block_size"] = wpe_weight.shape[0]

        # Ensure expected optional flags exist
        if "integer_poe" not in model_conf:
            model_conf["integer_poe"] = False
        if "bias" not in model_conf:
            model_conf["bias"] = False

        model_conf["logging"] = False
        self.model = LMTAD(type("Cfg", (), model_conf))

        unwanted_prefix = "_orig_mod."
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)

        self.dataset_config = checkpoint.get("dataset_config", None)

        # Optional: build dataset to access dictionary (SOT/EOT/PAD)
        self.dictionary = None
        # Dictionary is not required for distillation (SOT optional)
        self.dictionary = None

        # AMP/autocast context used for teacher forward
        if self.device.startswith("cuda"):
            self._ctx = torch.amp.autocast(device_type="cuda", dtype=self._ptdtype)
        else:
            self._ctx = nullcontext()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def get_grid_size_hw(self) -> Optional[Tuple[int, int]]:
        """Return (height, width) for Beijing-style datasets if available.

        LM-TAD training code stores it under `dataset_config.grip_size`.
        Returns None if not present.
        """
        if self.dataset_config is None:
            return None
        # Some configs use attribute name 'grip_size'
        h_w = getattr(self.dataset_config, "grip_size", None)
        if isinstance(h_w, (list, tuple)) and len(h_w) == 2:
            return int(h_w[0]), int(h_w[1])
        return None

    def sot_token(self) -> Optional[int]:
        """Return SOT token id if dictionary is available."""
        if self.dictionary is None:
            return None
        # Try multiple access patterns for robustness
        for name in ("sot_token", "SOT"):
            tok = getattr(self.dictionary, name, None)
            if callable(tok):
                try:
                    return int(tok())  # type: ignore[arg-type]
                except Exception:
                    pass
        # Named lookup
        try:
            return int(self.dictionary["SOT"])  # type: ignore[index]
        except Exception:
            return None

    @torch.no_grad()
    def predict_next_distribution(self, history_tokens: torch.LongTensor) -> torch.Tensor:
        """Return next-token probability distribution over the LM-TAD vocab.

        Parameters
        ----------
        history_tokens: torch.LongTensor
            Shape (T,) or (1,T). This should contain grid-tokenized history.

        Returns
        -------
        torch.Tensor
            Shape (V,), probabilities over the entire vocabulary.
        """
        if history_tokens.dim() == 1:
            x = history_tokens.unsqueeze(0)
        else:
            x = history_tokens

        # Truncate to window
        if x.size(1) > self.window:
            x = x[:, -self.window:]

        x = x.to(self.device)
        with self._ctx:
            logits, _ = self.model(x)  # (B, T, V)
        logits_last = logits[:, -1, :]  # (B,V)
        probs = torch.softmax(logits_last, dim=-1)
        if probs.size(0) == 1:
            return probs[0]
        return probs


