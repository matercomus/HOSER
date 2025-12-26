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
from contextlib import contextmanager
from contextlib import nullcontext
from types import ModuleType
from typing import Iterator, Optional, Tuple

import torch


def _load_module_from_path(module_name: str, module_path: str) -> ModuleType:
    """Load a Python module from an explicit file path.

    This is used to load external LM-TAD files without requiring LM-TAD to be
    installed as a package.
    """

    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


@contextmanager
def _lmtad_sys_namespace(repo_path: str) -> Iterator[None]:
    """Temporarily make LM-TAD importable as `models.*` and `utils`.

    This is required because LM-TAD checkpoints may contain pickled objects
    referencing `models.LMTAD` during `torch.load()`.
    """

    import os

    code_path = f"{repo_path}/code"
    if code_path not in sys.path:
        sys.path.insert(0, code_path)
    elif sys.path[0] != code_path:
        sys.path.remove(code_path)
        sys.path.insert(0, code_path)

    lmtad_utils_py = os.path.join(code_path, "utils.py")
    lmtad_model_py = os.path.join(code_path, "models", "LMTAD.py")
    if not os.path.exists(lmtad_model_py):
        raise ImportError(f"LM-TAD repo missing {lmtad_model_py}")

    saved_modules: dict[str, ModuleType] = {}
    for name, module in list(sys.modules.items()):
        if name == "utils" or name == "models" or name.startswith("models."):
            if isinstance(module, ModuleType):
                saved_modules[name] = module
            sys.modules.pop(name, None)

    try:
        import importlib

        # Import under their canonical names so recursive imports work.
        if os.path.exists(lmtad_utils_py):
            importlib.import_module("utils")

        # Pre-import so it's available for checkpoint unpickling.
        importlib.import_module("models.LMTAD")
        yield
    finally:
        # Remove any LM-TAD-injected modules so we can restore HOSER's.
        for name in list(sys.modules.keys()):
            if name == "utils" or name == "models" or name.startswith("models."):
                sys.modules.pop(name, None)

        sys.modules.update(saved_modules)


def _import_lmtad_LMTAD_class(repo_path: str) -> type:
    """Import LM-TAD's `LMTAD` class from `<repo>/code/models/LMTAD.py`."""

    import importlib

    with _lmtad_sys_namespace(repo_path):
        module = importlib.import_module("models.LMTAD")
        LMTAD = getattr(module, "LMTAD", None)
        if LMTAD is None:
            raise ImportError("LMTAD class not found in models.LMTAD")
        return LMTAD


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

        # LM-TAD checkpoints may include pickled objects referencing
        # `models.LMTAD`, so make LM-TAD importable during `torch.load()`.
        with _lmtad_sys_namespace(self.repo_path):
            LMTAD = _import_lmtad_LMTAD_class(self.repo_path)
            try:
                checkpoint = torch.load(
                    self.ckpt_path, map_location=self.device, weights_only=False
                )
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
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)

        self.dataset_config = checkpoint.get("dataset_config", None)

        # Load or infer SOT token ID
        try:
            # Try to load dictionary from checkpoint first
            if "dictionary" in checkpoint:
                self.dictionary = checkpoint["dictionary"]
            else:
                # Infer SOT token ID from model embeddings
                # LM-TAD typically puts SOT as the last token (highest ID)
                # Use the already-extracted state_dict (works for both "state_dict" and "model" keys)
                if "transformer.wte.weight" in state_dict:
                    vocab_size = state_dict["transformer.wte.weight"].shape[0]
                    self.sot_id = vocab_size - 1  # SOT is typically the last token
                    print(
                        f"[distill] Inferred SOT token ID: {self.sot_id} (vocab_size: {vocab_size})"
                    )
                else:
                    # Fallback: assume SOT is token 0
                    self.sot_id = 0
                    print(f"[distill] Using fallback SOT token ID: {self.sot_id}")
        except Exception as e:
            print(f"Warning: Could not determine SOT token: {e}")
            self.sot_id = None
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

    def vocab_size(self) -> Optional[int]:
        """Return vocabulary size from model.

        Returns vocab_size from model config or infers from embedding weights.
        Returns None if cannot be determined.
        """
        # Try to get from model config
        if hasattr(self.model, "config") and hasattr(self.model.config, "vocab_size"):
            return int(self.model.config.vocab_size)

        # Fallback: infer from embedding weights
        try:
            if hasattr(self.model, "transformer") and hasattr(
                self.model.transformer, "wte"
            ):
                return int(self.model.transformer.wte.weight.shape[0])
        except Exception:
            pass

        return None

    def sot_token(self) -> Optional[int]:
        """Return SOT token id."""
        return self.sot_id

    @torch.no_grad()
    def predict_next_distribution(
        self, history_tokens: torch.LongTensor
    ) -> torch.Tensor:
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
            x = x[:, -self.window :]

        x = x.to(self.device)
        with self._ctx:
            logits, _ = self.model(x)  # (B, T, V)
        logits_last = logits[:, -1, :]  # (B,V)
        probs = torch.softmax(logits_last, dim=-1)
        if probs.size(0) == 1:
            return probs[0]
        return probs

    def predict_next_distribution_cached(
        self, history_tokens: torch.LongTensor
    ) -> torch.Tensor:
        """Cached version of predict_next_distribution for repeated queries."""
        # Disable caching for now - the overhead of creating cache keys from GPU tensors
        # is likely more expensive than the teacher forward pass for unique sequences
        return self.predict_next_distribution(history_tokens)

        # Original caching code (disabled due to performance concerns):
        # - Converting GPU tensors to Python tuples forces expensive GPU->CPU transfer
        # - With diverse training data, cache hit rate is likely very low
        # - The teacher forward pass is already optimized with torch.compile


def validate_tokenized_trajectory_for_lmtad(
    tokens, vocab_size: int, min_length: int = 2, max_duplicate_ratio: float = 0.1
) -> tuple[bool, str, dict]:
    """Validate a tokenized (LM-TAD grid token) trajectory before querying the teacher.

    This mirrors the validation used for raw road IDs but checks token space
    semantics (i.e., token values against `vocab_size`).
    """
    # Basic checks
    if not tokens:
        return False, "Empty trajectory", {}

    if len(tokens) < min_length:
        return False, f"Trajectory too short: {len(tokens)} < {min_length}", {}

    # Check token range and types
    invalid_tokens = []
    for i, t in enumerate(tokens):
        if not isinstance(t, int):
            return False, f"Non-integer token at position {i}: {t}", {}
        if t < 0:
            invalid_tokens.append(f"negative token: {t}")
        elif t >= vocab_size:
            invalid_tokens.append(f"token {t} >= vocab_size {vocab_size}")

    if invalid_tokens:
        # Provide structured diagnostics for callers instead of forcing them
        # to parse a human-readable message.
        # Attempt to extract numeric token IDs where possible.
        numeric_tokens = []
        for tok in invalid_tokens:
            parts = [p for p in tok.split() if p.lstrip("-").isdigit()]
            if parts:
                try:
                    numeric_tokens.append(int(parts[-1]))
                except Exception:
                    pass
        return (
            False,
            f"Invalid tokens: {', '.join(invalid_tokens[:5])}",
            {"invalid_tokens": numeric_tokens},
        )

    # Duplicate checks. Allow callers to disable duplicate-based rejection by
    # setting `max_duplicate_ratio >= 1.0` (temporary evaluation mode).
    # When `max_duplicate_ratio >= 1.0` we treat duplicate checks as disabled
    # to allow forcing evaluation for diagnostic purposes.
    if max_duplicate_ratio < 1.0:
        unique = set(tokens)
        duplicate_ratio = 1 - (len(unique) / len(tokens))
        if duplicate_ratio > max_duplicate_ratio:
            return (
                False,
                f"Excessive duplicates: {duplicate_ratio:.1%} > {max_duplicate_ratio:.1%}",
                {"duplicate_ratio": float(duplicate_ratio)},
            )

        # Consecutive duplicates
        consecutive_duplicates = 0
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i - 1]:
                consecutive_duplicates += 1

        if consecutive_duplicates > 0:
            return (
                False,
                f"Consecutive duplicate tokens: {consecutive_duplicates}",
                {"consecutive_duplicates": consecutive_duplicates},
            )

    return True, "Valid", {}
