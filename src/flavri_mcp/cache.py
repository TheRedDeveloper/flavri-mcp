"""
flavri_mcp.cache
In-Memory Dense Tensor Chemosensory Universe Cache (FlavriCache).
Loads ~28,000 molecules into dense PyTorch tensors (embeddings [N, 512], profiles [N, 148], half_lives [N])
in <60 MB RAM/VRAM, enabling microsecond vector operations and property screening without disk latency.
"""

from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import os
import torch

from flavri_mcp.constants import (
    ALL_CHANNELS,
    CHANNEL_TO_IDX,
    TASTE_INDICES,
    ODOR_INDICES,
    NUM_CHANNELS,
)
from flavri_mcp.config import CACHE_PATH, get_device


class FlavriCache:
    """In-memory dense tensor chemical universe for sub-millisecond screening."""

    def __init__(self, cache_path: Optional[Path] = None, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.cache_path = cache_path or CACHE_PATH

        if self.cache_path.exists() and self.cache_path.is_file():
            data = torch.load(self.cache_path, map_location=self.device, weights_only=False)
            self.smiles: List[str] = data["canonical_smiles"]
            self.embeddings: torch.Tensor = data["embeddings"].to(self.device)         # [N, 512]
            self.flavors: torch.Tensor = data["flavor_profiles"].to(self.device)       # [N, 148]
            self.half_lives: torch.Tensor = data["half_lives"].to(self.device)         # [N]
        else:
            # Empty cache fallback
            self.smiles: List[str] = []
            self.embeddings: torch.Tensor = torch.zeros((0, 512), dtype=torch.float32, device=self.device)
            self.flavors: torch.Tensor = torch.zeros((0, NUM_CHANNELS), dtype=torch.float32, device=self.device)
            self.half_lives: torch.Tensor = torch.zeros((0,), dtype=torch.float32, device=self.device)

        self.smiles_to_idx: Dict[str, int] = {s: i for i, s in enumerate(self.smiles)}

    def __len__(self) -> int:
        return len(self.smiles)

    def get(self, canonical_smi: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
        """Returns (embedding, flavor_profile, half_life) if cached, else None."""
        idx = self.smiles_to_idx.get(canonical_smi)
        if idx is None:
            return None
        return self.embeddings[idx], self.flavors[idx], float(self.half_lives[idx])

    def append(self, canonical_smi: str, emb: torch.Tensor, flavor: torch.Tensor, half_life: float):
        """Appends a new runtime molecule to the in-memory cache."""
        if canonical_smi in self.smiles_to_idx:
            return
        idx = len(self.smiles)
        self.smiles.append(canonical_smi)
        self.smiles_to_idx[canonical_smi] = idx

        emb_row = emb.unsqueeze(0).to(self.device)
        flavor_row = flavor.unsqueeze(0).to(self.device)
        hl_elem = torch.tensor([half_life], dtype=torch.float32, device=self.device)

        self.embeddings = torch.cat([self.embeddings, emb_row], dim=0)
        self.flavors = torch.cat([self.flavors, flavor_row], dim=0)
        self.half_lives = torch.cat([self.half_lives, hl_elem], dim=0)

    def screen(
        self,
        target_descriptors: List[str],
        exclude_descriptors: Optional[List[str]] = None,
        min_half_life_hours: Optional[float] = None,
        max_half_life_hours: Optional[float] = None,
        threshold: float = 0.2,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Ultra-fast vectorized screening over all cached molecules.
        Matches molecules where target descriptors are >= threshold,
        filters out molecules where any exclude descriptor is >= threshold,
        filters by half-life window, and ranks by target descriptor score.
        """
        N = len(self.smiles)
        if N == 0:
            return []

        # Target indices
        target_idxs = [CHANNEL_TO_IDX[d] for d in target_descriptors if d in CHANNEL_TO_IDX]
        if not target_idxs:
            return []

        # Exclude indices
        exclude_idxs = (
            [CHANNEL_TO_IDX[d] for d in exclude_descriptors if d in CHANNEL_TO_IDX]
            if exclude_descriptors
            else []
        )

        # 1. Target criteria: must have positive presence for target descriptors
        # We compute score as the mean intensity across target channels
        target_matrix = self.flavors[:, target_idxs]  # [N, num_targets]
        scores = target_matrix.mean(dim=-1)            # [N]

        # Target mask: each target descriptor should ideally be >= threshold (or at least one strong match)
        target_mask = (target_matrix >= threshold).all(dim=-1)

        # 2. Exclude criteria: exclude if ANY excluded descriptor >= threshold
        if exclude_idxs:
            exclude_matrix = self.flavors[:, exclude_idxs]
            exclude_mask = (exclude_matrix >= threshold).any(dim=-1)
        else:
            exclude_mask = torch.zeros(N, dtype=torch.bool, device=self.device)

        # 3. Persistence criteria
        hl_mask = torch.ones(N, dtype=torch.bool, device=self.device)
        if min_half_life_hours is not None:
            hl_mask = hl_mask & (self.half_lives >= min_half_life_hours)
        if max_half_life_hours is not None:
            hl_mask = hl_mask & (self.half_lives <= max_half_life_hours)

        # Combined filter mask
        valid_mask = target_mask & (~exclude_mask) & hl_mask

        matching_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        if len(matching_indices) == 0:
            # Fallback: relax all-targets to any-target match
            relaxed_target_mask = (target_matrix >= threshold).any(dim=-1)
            valid_mask = relaxed_target_mask & (~exclude_mask) & hl_mask
            matching_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        if len(matching_indices) == 0:
            return []

        valid_scores = scores[matching_indices]
        sorted_order = torch.argsort(valid_scores, descending=True)
        top_indices = matching_indices[sorted_order[:top_k]].cpu().tolist()

        results = []
        for idx in top_indices:
            smi = self.smiles[idx]
            hl = float(self.half_lives[idx].item())
            prof = self.flavors[idx].cpu().numpy()

            matched = [
                d for d in target_descriptors
                if d in CHANNEL_TO_IDX and prof[CHANNEL_TO_IDX[d]] >= threshold
            ]

            taste_dict = {}
            for t_idx in TASTE_INDICES:
                v = float(prof[t_idx])
                if v >= threshold:
                    taste_dict[ALL_CHANNELS[t_idx]] = round(v, 4)
            taste_sorted = dict(sorted(taste_dict.items(), key=lambda x: -x[1]))

            odor_dict = {}
            for o_idx in ODOR_INDICES:
                v = float(prof[o_idx])
                if v >= threshold:
                    odor_dict[ALL_CHANNELS[o_idx]] = round(v, 4)
            odor_sorted = dict(sorted(odor_dict.items(), key=lambda x: -x[1]))

            results.append({
                "smiles": smi,
                "half_life_hours": round(hl, 2),
                "matched_descriptors": matched,
                "taste": taste_sorted,
                "odor": odor_sorted,
            })

        return results


_GLOBAL_CACHE: Optional[FlavriCache] = None


def get_cache() -> FlavriCache:
    """Returns singleton FlavriCache instance."""
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = FlavriCache()
    return _GLOBAL_CACHE
