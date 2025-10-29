#!/usr/bin/env python3
import argparse
import os
import sys
from dataclasses import asdict, is_dataclass

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Export LM-TAD weights-only checkpoint"
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Path to LM-TAD repo root (expects code/ under it)",
    )
    parser.add_argument(
        "--ckpt_in", required=True, help="Path to original LM-TAD checkpoint .pt"
    )
    parser.add_argument(
        "--ckpt_out", required=True, help="Output path for weights-only checkpoint .pt"
    )
    parser.add_argument(
        "--grip_size",
        type=str,
        default=None,
        help="Grid dimensions for LM-TAD (e.g., '205 252' for Beijing, '46 134' for Porto)",
    )
    args = parser.parse_args()

    code_path = os.path.join(args.repo, "code")
    sys.path.insert(0, code_path)

    # Ensure model/dataset/utils symbols resolve during torch.load
    try:
        # Some checkpoints may require datasets/ or utils during load; ensure importable
        import utils as lmtad_utils  # noqa: F401
    except Exception:
        pass

    ckpt = torch.load(args.ckpt_in, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        state = ckpt.get("model") or ckpt.get("state_dict")
        model_conf = ckpt.get("model_config")

        # If no separate state_dict, the whole checkpoint might be the model
        if state is None and "model_config" not in ckpt:
            # Try to extract from a loaded model object
            if hasattr(ckpt, "state_dict"):
                state = ckpt.state_dict()
                model_conf = getattr(ckpt, "config", None)
            else:
                # Assume the checkpoint is the state dict itself
                state = ckpt
                model_conf = None

        if state is None:
            raise RuntimeError("Input checkpoint missing 'model' or 'state_dict'")
    else:
        # Checkpoint is likely a model object
        if hasattr(ckpt, "state_dict"):
            state = ckpt.state_dict()
            model_conf = getattr(ckpt, "config", None)
        else:
            raise RuntimeError("Unrecognized checkpoint format")

    # Create a minimal config if none exists
    if model_conf is None:
        print(
            "Warning: No model_config found, creating minimal config from LM-TAD train.sh"
        )
        model_conf = {
            "n_layer": 8,
            "n_head": 12,
            "n_embd": 768,
            "dropout": 0.2,
            "grip_size": args.grip_size
            if args.grip_size
            else "205 252",  # CLI argument or Beijing default
            "vocab_size": None,  # Will be inferred
            "block_size": -1,
        }

    if is_dataclass(model_conf):
        model_conf = asdict(model_conf)
    elif hasattr(model_conf, "__dict__"):
        model_conf = dict(model_conf.__dict__)

    # Sanitize: remove non-serializable fields
    model_conf.pop("log_file", None)

    out = {
        "state_dict": state,
        "model_config": model_conf,
    }
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    torch.save(out, args.ckpt_out)
    print(f"Saved weights-only checkpoint to {args.ckpt_out}")


if __name__ == "__main__":
    main()
