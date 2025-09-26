"""
Distillation Hook for HOSER Training
====================================

Encapsulates LM-TAD teacher loading and KL distillation computation.
Kept separate from the original training code to minimize invasiveness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

from .lmtad_teacher import LMTADTeacher
from .grid_mapper import GridMapper, GridConfig


@dataclass
class DistillConfig:
    enabled: bool = False
    repo_path: str = ""
    ckpt_path: str = ""
    dtype: str = "float16"
    window: int = 64
    lambda_kl: float = 0.01
    temperature: float = 2.0
    sample_steps_per_trace: int = 1
    grid_size: Optional[float] = None
    downsample_factor: int = 1
    verify_grid_dims: bool = True


class DistillationManager:
    """Owns teacher model and mapping; computes KL loss on batches."""

    def __init__(
        self,
        distill_cfg: DistillConfig,
        device: str,
        boundary_min_lat: float,
        boundary_max_lat: float,
        boundary_min_lng: float,
        boundary_max_lng: float,
        road_centroids_lat: np.ndarray,
        road_centroids_lng: np.ndarray,
        logger=None,
    ) -> None:
        self.cfg = distill_cfg
        self.device = device
        self.logger = logger

        # Teacher
        self.teacher = LMTADTeacher(
            repo_path=self.cfg.repo_path,
            ckpt_path=self.cfg.ckpt_path,
            device=self.device,
            dtype=self.cfg.dtype,
            window=self.cfg.window,
        )

        # Validate or infer grid dims
        verify_hw: Optional[Tuple[int, int]] = None
        teacher_hw = self.teacher.get_grid_size_hw()
        if self.cfg.verify_grid_dims and teacher_hw is not None:
            verify_hw = teacher_hw
            if self.logger:
                self.logger.info(f"[distill] Teacher grid dims (H,W) = {teacher_hw}")

        # Require grid_size for mapping; keep explicit and readable
        if self.cfg.grid_size is None:
            raise ValueError(
                "distill.grid_size must be provided to map roads to grid tokens"
            )

        grid_cfg = GridConfig(
            min_lat=float(boundary_min_lat),
            max_lat=float(boundary_max_lat),
            min_lng=float(boundary_min_lng),
            max_lng=float(boundary_max_lng),
            grid_size=float(self.cfg.grid_size),
            downsample_factor=int(self.cfg.downsample_factor),
        )

        # Road centroids (lat,lng)
        road_centroids = np.stack(
            [road_centroids_lat.astype(np.float64), road_centroids_lng.astype(np.float64)],
            axis=1,
        )

        # Mapper
        self.mapper = GridMapper(grid_cfg, road_centroids, verify_hw=verify_hw)
        self.road_to_token = self.mapper.map_all()  # np.int64 of shape (N,)

        # Pre-resolve SOT token id if available
        self.sot_id = self.teacher.sot_token()
        if self.logger:
            self.logger.info(
                f"[distill] Initialized teacher; SOT token id: {self.sot_id if self.sot_id is not None else 'None'}"
            )

    @torch.no_grad()
    def _make_history_tokens(self, road_ids_1d: torch.Tensor) -> torch.LongTensor:
        """Map a 1D tensor of road IDs to grid tokens; prepend SOT if available."""
        roads_np = road_ids_1d.detach().cpu().numpy().astype(np.int64, copy=False)
        tokens_np = self.road_to_token[roads_np]
        if self.sot_id is not None:
            tokens_np = np.concatenate([[int(self.sot_id)], tokens_np], axis=0)
        return torch.from_numpy(tokens_np.astype(np.int64))

    def compute_kl_for_batch(
        self,
        logits: torch.Tensor,                      # (B, T, Cmax)
        batch_trace_road_id: torch.Tensor,         # (B, T)
        batch_trace_len: torch.Tensor,             # (B,)
        batch_candidate_road_id: torch.Tensor,     # (B, T, Cmax)
        batch_candidate_len: torch.Tensor,         # (B, T)
    ) -> torch.Tensor:
        """Compute average KL loss across sampled steps in the batch.

        Strategy: sample last valid step per trace (configurable up to N steps).
        This keeps compute low and code straightforward.
        """
        B, T, Cmax = logits.shape
        device = logits.device

        kl_accum: List[torch.Tensor] = []

        num_samples = int(self.cfg.sample_steps_per_trace)
        # Only last step for now (simplest and robust)
        for b in range(B):
            t = int(batch_trace_len[b].item()) - 1
            if t < 0:
                continue

            # History roads → tokens
            roads_1d = batch_trace_road_id[b, : t + 1]
            hist_tokens = self._make_history_tokens(roads_1d)

            # Teacher dist over full vocab
            q_full = self.teacher.predict_next_distribution(hist_tokens)

            # Candidate slice and lengths
            cand_len = int(batch_candidate_len[b, t].item())
            if cand_len <= 0:
                continue
            cand_roads = batch_candidate_road_id[b, t, :cand_len]

            # Map candidates to tokens and gather teacher probs
            cand_np = cand_roads.detach().cpu().numpy().astype(np.int64, copy=False)
            cand_tokens = self.road_to_token[cand_np]
            q_c = q_full[cand_tokens].to(device)
            q_c = q_c / torch.clamp(q_c.sum(), min=1e-9)

            # Student probs (temperature)
            s_logits = logits[b, t, :cand_len]
            T = float(self.cfg.temperature)
            p_tau = F.softmax(s_logits / T, dim=-1)
            q_tau = torch.pow(q_c, 1.0 / T)
            q_tau = q_tau / torch.clamp(q_tau.sum(), min=1e-9)

            # KL(q||p)
            kl = torch.sum(q_tau * (torch.log(torch.clamp(q_tau, min=1e-9)) - torch.log(torch.clamp(p_tau, min=1e-9))))
            kl_accum.append(kl)

            # Only 1 sample per trace by default
            if len(kl_accum) >= (b + 1) * num_samples:
                continue

        if not kl_accum:
            return torch.tensor(0.0, device=device)

        kl_mean = torch.stack(kl_accum).mean()
        return kl_mean


