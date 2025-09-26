#!/usr/bin/env python3
import argparse
import os
import sys
from dataclasses import asdict, is_dataclass

import torch


def main():
    parser = argparse.ArgumentParser(description="Export LM-TAD weights-only checkpoint")
    parser.add_argument("--repo", required=True, help="Path to LM-TAD repo root (expects code/ under it)")
    parser.add_argument("--ckpt_in", required=True, help="Path to original LM-TAD checkpoint .pt")
    parser.add_argument("--ckpt_out", required=True, help="Output path for weights-only checkpoint .pt")
    args = parser.parse_args()

    code_path = os.path.join(args.repo, "code")
    sys.path.insert(0, code_path)

    # Ensure model/dataset/utils symbols resolve during torch.load
    from models import LMTAD, LMTADConfig  # type: ignore
    try:
        # Some checkpoints may require datasets/ or utils during load; ensure importable
        import utils as lmtad_utils  # noqa: F401
    except Exception:
        pass

    ckpt = torch.load(args.ckpt_in, map_location="cpu", weights_only=False)
    state = ckpt.get("model") or ckpt.get("state_dict")
    if state is None:
        raise RuntimeError("Input checkpoint missing 'model' or 'state_dict'")

    model_conf = ckpt.get("model_config")
    if model_conf is None:
        raise RuntimeError("Input checkpoint missing 'model_config'")

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


