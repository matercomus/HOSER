"""
Distillation Hook for HOSER Training
====================================

Encapsulates LM-TAD teacher loading and KL distillation computation.
Kept separate from the original training code to minimize invasiveness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

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
        self.road_to_token = torch.from_numpy(self.mapper.map_all()).to(device)  # Move to GPU immediately

        # Pre-resolve SOT token id if available
        self.sot_id = self.teacher.sot_token()
        if self.logger:
            if self.sot_id is not None:
                self.logger.info(f"[distill] Initialized teacher; SOT token id: {self.sot_id}")
            else:
                self.logger.warning("[distill] SOT token not available - teacher may not be properly initialized")
        
        # Pre-allocate SOT tensor to avoid repeated creation during training
        if self.sot_id is not None:
            self.sot_tensor = torch.tensor([self.sot_id], device=device, dtype=torch.long)


    @torch.no_grad()
    def _make_history_tokens(self, road_ids_1d: torch.Tensor) -> torch.LongTensor:
        """Map a 1D tensor of road IDs to grid tokens; prepend SOT if available."""
        # Optimized: Keep everything on GPU, no CPU round-trip
        tokens = self.road_to_token[road_ids_1d]
        if self.sot_id is not None:
            tokens = torch.cat([self.sot_tensor, tokens], dim=0)
        return tokens

    def compute_kl_for_batch(
        self,
        logits: torch.Tensor,
        batch_trace_road_id: torch.Tensor,
        batch_trace_len: torch.Tensor,
        batch_candidate_road_id: torch.Tensor,
        batch_candidate_len: torch.Tensor,
    ) -> torch.Tensor:
        return self._compute_vectorized(
            logits,
            batch_trace_road_id,
            batch_trace_len,
            batch_candidate_road_id,
            batch_candidate_len,
        )

    def _compute_vectorized(
        self,
        logits: torch.Tensor,
        batch_trace_road_id: torch.Tensor,
        batch_trace_len: torch.Tensor,
        batch_candidate_road_id: torch.Tensor,
        batch_candidate_len: torch.Tensor,
    ) -> torch.Tensor:
        device = logits.device
        B, T, _ = logits.shape

        batch_trace_len = batch_trace_len.to(device)
        batch_candidate_len = batch_candidate_len.to(device)
        batch_trace_road_id = batch_trace_road_id.to(device)
        batch_candidate_road_id = batch_candidate_road_id.to(device)

        last_idx = batch_trace_len - 1
        valid_mask = last_idx >= 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=device)

        batch_indices = torch.arange(B, device=device)[valid_mask]
        last_idx = last_idx[valid_mask]
        candidate_len = batch_candidate_len[batch_indices, last_idx]
        valid_candidate = candidate_len > 0
        if not valid_candidate.any():
            return torch.tensor(0.0, device=device)

        batch_indices = batch_indices[valid_candidate]
        last_idx = last_idx[valid_candidate]
        candidate_len = candidate_len[valid_candidate]

        sorted_indices = torch.argsort(candidate_len, descending=True)
        batch_indices = batch_indices[sorted_indices]
        last_idx = last_idx[sorted_indices]
        candidate_len = candidate_len[sorted_indices]

        # Account for SOT token when calculating max history length
        max_trace_len = int(batch_trace_len.max().item())
        max_hist = max_trace_len + (1 if self.sot_id is not None else 0)
        history_tokens = torch.zeros((batch_indices.size(0), max_hist), dtype=torch.long, device=device)
        # Vectorized: Process all sequences without .tolist() calls
        for row in range(batch_indices.size(0)):
            b = batch_indices[row]
            t = last_idx[row]
            seq = batch_trace_road_id[b, : t + 1]
            # Optimized: GPU indexing, no CPU round-trip
            seq = self.road_to_token[seq]
            if self.sot_id is not None:
                seq = torch.cat([self.sot_tensor, seq], dim=0)
            history_tokens[row, -seq.size(0):] = seq

        # Optimized: Process all samples in single batched teacher inference with caching
        # Use cached version for better performance on repeated inputs
        teacher_logits = self.teacher.predict_next_distribution_cached(history_tokens)
        if teacher_logits.dim() == 1:
            teacher_logits = teacher_logits.unsqueeze(0)

        # Debug: Check if teacher model is using GPU efficiently (disabled for cleaner output)
        # if hasattr(teacher_logits, 'device'):
        #     print(f"[debug] Teacher logits device: {teacher_logits.device}, shape: {teacher_logits.shape}")

        kl_terms = []
        # Vectorized: Process without .tolist() to avoid GPU->CPU transfers
        for row in range(batch_indices.size(0)):
            b = batch_indices[row].item()  # Only convert single values when needed
            t = last_idx[row].item()
            cand = candidate_len[row].item()
            candidate_ids = batch_candidate_road_id[b, t, : cand]
            # Optimized: Use GPU indexing directly, no CPU round-trip
            candidate_tokens = self.road_to_token[candidate_ids]
            q_c = teacher_logits[row, candidate_tokens]

            # Handle invalid teacher probabilities
            q_c_sum = q_c.sum()
            if q_c_sum <= 0 or torch.isnan(q_c_sum) or torch.isinf(q_c_sum):
                continue  # Skip this sample if teacher gives invalid probabilities

            q_c = q_c / torch.clamp(q_c_sum, min=1e-9)

            s_logits = logits[b, t, : cand]
            T = float(self.cfg.temperature)
            p_tau = torch.softmax(s_logits / T, dim=-1)

            # Ensure student probabilities are valid
            if torch.isnan(p_tau).any() or torch.isinf(p_tau).any():
                continue  # Skip this sample if student gives invalid probabilities

            q_tau = torch.clamp(q_c, min=1e-9).pow(1.0 / T)
            q_tau_sum = q_tau.sum()
            if q_tau_sum <= 0 or torch.isnan(q_tau_sum) or torch.isinf(q_tau_sum):
                continue  # Skip if temperature scaling gives invalid probabilities

            q_tau = q_tau / q_tau_sum

            # Compute KL divergence with proper NaN handling
            log_q_tau = torch.log(torch.clamp(q_tau, min=1e-9))
            log_p_tau = torch.log(torch.clamp(p_tau, min=1e-9))

            kl = torch.sum(q_tau * (log_q_tau - log_p_tau))

            # Skip if KL is NaN or infinite
            if torch.isnan(kl) or torch.isinf(kl):
                continue

            kl_terms.append(kl)

        if not kl_terms:
            return torch.tensor(0.0, device=device)
        return torch.stack(kl_terms).mean()


