"""
flavri_mcp.model
PyTorch model architecture and inference engine for FLAVRI 2.0:
- 6-layer D-MPNN Backbone (hidden_dim=512, depth=6) with Multi-Head Attentive Readout
- Morgan ECFP4 fingerprint projection and fusion
- SingleMoleculeHead (148 continuous channels)
- CommutativeSynergyHead (exact symmetry & identity neutrality)
- General mixture blending law
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from flavri_mcp.utils.chemistry import estimate_molecule_half_life

from flavri_mcp.constants import (
    NUM_CHANNELS,
    TASTE_CHANNELS,
    ODOR_CHANNELS,
    ALL_CHANNELS,
    TASTE_INDICES,
    ODOR_INDICES,
)
from flavri_mcp.config import get_checkpoint_path, get_device
from flavri_mcp.utils.featurizer import (
    ATOM_FDIM,
    BOND_FDIM,
    BatchMolGraph,
    smiles_to_graph_batch,
)
from flavri_mcp.utils.chemistry import (
    canonicalize_single_smiles,
    parse_ratio_expression,
    format_canonical_formula,
)


class MultiHeadAttentivePooling(nn.Module):
    """
    Multi-Head Attentive Readout Pooling over atoms:
    K learned queries (default 4 heads).
    Concatenates heads and projects to hidden_dim.
    """
    def __init__(self, hidden_dim: int = 512, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.w_att = nn.Linear(hidden_dim, hidden_dim)
        self.v_att = nn.Linear(hidden_dim, num_heads, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, atom_h: torch.Tensor, a2g: torch.Tensor, batch_size: int) -> torch.Tensor:
        att_scores = self.v_att(torch.tanh(self.w_att(atom_h)))  # (total_atoms, num_heads)

        max_scores = torch.full((batch_size, self.num_heads), -1e9, device=atom_h.device)
        max_scores.scatter_reduce_(0, a2g.unsqueeze(-1).expand_as(att_scores), att_scores, reduce="amax")

        exp_scores = torch.exp(att_scores - max_scores[a2g])
        sum_exp = torch.zeros((batch_size, self.num_heads), device=atom_h.device)
        sum_exp.scatter_add_(0, a2g.unsqueeze(-1).expand_as(exp_scores), exp_scores)

        weights = exp_scores / (sum_exp[a2g] + 1e-9)

        head_outputs = []
        for k in range(self.num_heads):
            w_k = weights[:, k : k + 1]
            weighted_h = atom_h * w_k
            pooled_k = torch.zeros((batch_size, self.hidden_dim), device=atom_h.device)
            pooled_k.scatter_add_(0, a2g.unsqueeze(-1).expand_as(weighted_h), weighted_h)
            head_outputs.append(pooled_k)

        multi_head = torch.cat(head_outputs, dim=-1)
        return self.proj(multi_head)


class DMPNNBackbone(nn.Module):
    """
    Directed Message Passing Neural Network (D-MPNN) Backbone.
    """
    def __init__(
        self,
        atom_fdim: int = ATOM_FDIM,
        bond_fdim: int = BOND_FDIM,
        hidden_dim: int = 512,
        depth: int = 6,
        dropout: float = 0.1,
        num_readout_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = nn.Dropout(dropout)

        self.W_i = nn.Linear(atom_fdim + bond_fdim, hidden_dim, bias=False)
        self.W_m = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_a = nn.Linear(atom_fdim + hidden_dim, hidden_dim)
        self.readout = MultiHeadAttentivePooling(hidden_dim, num_heads=num_readout_heads)

    def forward(self, batch_graph: BatchMolGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        device = batch_graph.f_atoms.device

        if batch_graph.total_bonds == 0:
            atom_h = F.relu(self.W_a(torch.cat([
                batch_graph.f_atoms,
                torch.zeros((batch_graph.total_atoms, self.hidden_dim), device=device)
            ], dim=-1)))
            z = self.readout(atom_h, batch_graph.a2g, batch_graph.batch_size)
            return z, atom_h

        u_atoms = batch_graph.b2u[1:]
        f_u_atoms = batch_graph.f_atoms[u_atoms]
        f_bonds = batch_graph.f_bonds[1:]

        init_input = torch.cat([f_u_atoms, f_bonds], dim=-1)
        m_0 = F.relu(self.W_i(init_input))

        zero_row = torch.zeros((1, self.hidden_dim), device=device)
        m = torch.cat([zero_row, m_0], dim=0)

        for _ in range(self.depth - 1):
            incoming_m = m[batch_graph.b2b]
            sum_incoming = incoming_m.sum(dim=1)
            m_update = F.relu(m_0 + self.W_m(sum_incoming[1:]))
            m_update = self.dropout(m_update)
            m = torch.cat([zero_row, m_update], dim=0)

        incoming_to_atoms = m[batch_graph.a2b]
        sum_m_v = incoming_to_atoms.sum(dim=1)

        atom_input = torch.cat([batch_graph.f_atoms, sum_m_v], dim=-1)
        atom_h = F.relu(self.W_a(atom_input))
        atom_h = self.dropout(atom_h)

        z = self.readout(atom_h, batch_graph.a2g, batch_graph.batch_size)
        return z, atom_h


class SingleMoleculeHead(nn.Module):
    """
    Single Molecule Sensory Profile Head:
    V_i = sigma( Linear(hidden_dim -> 148) ( GELU( Linear(hidden_dim -> hidden_dim) z_i ) ) )
    Output in [0, 1]^148.
    """
    def __init__(self, hidden_dim: int = 512, num_channels: int = NUM_CHANNELS, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.act(self.fc1(z)))
        logits = self.fc2(h)
        return torch.sigmoid(logits)


class CommutativeSynergyHead(nn.Module):
    """
    Symmetric Commutative Interaction Synergy Head:
    Given z_A, z_B in R^hidden_dim:
      u = z_A * z_B
      v = |z_A - z_B|
      [u || v] in R^(hidden_dim * 2)
    Delta_AB = tanh( Linear(hidden_dim -> 148) ( GELU( Linear(hidden_dim*2 -> hidden_dim) [u || v] ) ) )
    Output in [-1, +1]^148.
    Exact commutativity: Delta_AB == Delta_BA by construction.
    """
    def __init__(self, hidden_dim: int = 512, num_channels: int = NUM_CHANNELS, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_channels)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        u = z_a * z_b
        v = torch.abs(z_a - z_b)
        uv = torch.cat([u, v], dim=-1)
        h = self.dropout(self.act(self.fc1(uv)))
        return torch.tanh(self.fc2(h))


class FlavriMultimodalModel(nn.Module):
    """
    FLAVRI 2.0 Multimodal Chemosensory Foundation Architecture.
    """
    def __init__(
        self,
        hidden_dim: int = 512,
        depth: int = 6,
        num_channels: int = NUM_CHANNELS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels

        self.backbone = DMPNNBackbone(
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            num_readout_heads=4,
        )
        self.single_head = SingleMoleculeHead(
            hidden_dim=hidden_dim,
            num_channels=num_channels,
            dropout=dropout,
        )
        self.synergy_head = CommutativeSynergyHead(
            hidden_dim=hidden_dim,
            num_channels=num_channels,
            dropout=dropout,
        )

        self.fp_dim = 512
        self.fp_proj = nn.Sequential(
            nn.Linear(self.fp_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def encode(self, batch_graph: BatchMolGraph) -> torch.Tensor:
        z_graph, _ = self.backbone(batch_graph)
        if hasattr(batch_graph, "fps") and batch_graph.fps is not None:
            z_fp = self.fp_proj(batch_graph.fps)
            z = self.fusion(torch.cat([z_graph, z_fp], dim=-1))
        else:
            z = z_graph
        return z

    def predict_singles(self, batch_graph: BatchMolGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(batch_graph)
        v = self.single_head(z)
        return z, v

    def predict_synergy(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        return self.synergy_head(z_a, z_b)


class FlavriPredictor:
    """High-level singleton predictor wrapping FlavriMultimodalModel for inference."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[torch.device] = None):
        self.device = device or get_device()
        ckpt_path = checkpoint_path or get_checkpoint_path()

        checkpoint = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

        if "hidden_dim" in checkpoint:
            hidden_dim = checkpoint["hidden_dim"]
        elif "backbone.W_m.weight" in state_dict:
            hidden_dim = state_dict["backbone.W_m.weight"].shape[0]
        else:
            hidden_dim = 512

        depth = checkpoint.get("depth", 6)

        self.model = FlavriMultimodalModel(hidden_dim=hidden_dim, depth=depth, num_channels=NUM_CHANNELS)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_smiles(self, smiles_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Featurizes and runs single molecule inference for a list of canonical SMILES.
        Returns:
            z: (N, hidden_dim) latent embeddings
            v: (N, 148) single molecule predictions in [0, 1]
        """
        batch_graph = smiles_to_graph_batch(smiles_list, self.device)
        z, v = self.model.predict_singles(batch_graph)
        return z, v

    @torch.no_grad()
    def predict_components(
        self,
        smiles_list: List[str],
        weights_list: List[float],
        cache: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Computes composite blend profile for multi-component mixtures using the FLAVRI Mixture Blending Law:
        y_blend = clamp( sum_i w_i * V_i + 4 * sum_{i < j} w_i * w_j * Delta_ij, 0.0, 1.0 )
        """
        M = len(smiles_list)
        if M == 0:
            raise ValueError("No components provided.")

        z_list: List[torch.Tensor] = []
        v_list: List[torch.Tensor] = []
        comp_records: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        missing_smis = []
        missing_indices = []

        for i, s in enumerate(smiles_list):
            cached = cache.get(s) if cache is not None else None
            if cached is not None:
                z_i, v_i, _ = cached
                z_list.append(z_i.to(self.device))
                v_list.append(v_i.to(self.device))
                comp_records[s] = (z_i, v_i)
            else:
                z_list.append(None)  # placeholder
                v_list.append(None)
                missing_smis.append(s)
                missing_indices.append(i)

        if missing_smis:
            z_miss, v_miss = self.encode_smiles(missing_smis)
            for k, orig_idx in enumerate(missing_indices):
                z_k = z_miss[k]
                v_k = v_miss[k]
                z_list[orig_idx] = z_k
                v_list[orig_idx] = v_k
                comp_records[missing_smis[k]] = (z_k, v_k)

        # Single molecule case
        if M == 1:
            return v_list[0], comp_records

        w_tensor = torch.tensor(weights_list, dtype=torch.float32, device=self.device)
        V_stack = torch.stack(v_list, dim=0)  # (M, 148)

        # Linear mix
        linear_mix = torch.matmul(w_tensor, V_stack)  # (148,)

        # Non-linear synergy: sum_{i < j} 4 * w_i * w_j * Delta_ij
        synergy_sum = torch.zeros(NUM_CHANNELS, dtype=torch.float32, device=self.device)
        for i in range(M):
            for j in range(i + 1, M):
                w_pair_prod = 4.0 * weights_list[i] * weights_list[j]
                if w_pair_prod > 1e-6:
                    z_i = z_list[i].unsqueeze(0)
                    z_j = z_list[j].unsqueeze(0)
                    delta_ij = self.model.predict_synergy(z_i, z_j).squeeze(0)  # (148,)
                    synergy_sum += w_pair_prod * delta_ij

        blend = torch.clamp(linear_mix + synergy_sum, min=0.0, max=1.0)
        return blend, comp_records

    def predict_formula(
        self,
        formula: str,
        threshold: float = 0.2,
        cache: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end prediction for single molecule or ratio formula.
        Returns Option 4 schema: prioritized taste and odor dicts sorted descending.
        """
        smiles_list, weights_list = parse_ratio_expression(formula, normalize=True)
        canonical_str = format_canonical_formula(dict(zip(smiles_list, weights_list)))

        blend_vec, _ = self.predict_components(smiles_list, weights_list, cache=cache)
        blend_np = blend_vec.detach().cpu().numpy()

        taste_dict = {}
        for idx in TASTE_INDICES:
            score = float(blend_np[idx])
            if score >= threshold:
                taste_dict[ALL_CHANNELS[idx]] = round(score, 4)
        taste_sorted = dict(sorted(taste_dict.items(), key=lambda item: -item[1]))

        odor_dict = {}
        for idx in ODOR_INDICES:
            score = float(blend_np[idx])
            if score >= threshold:
                odor_dict[ALL_CHANNELS[idx]] = round(score, 4)
        odor_sorted = dict(sorted(odor_dict.items(), key=lambda item: -item[1]))

        half_lives = {}
        for s in smiles_list:
            cached = cache.get(s) if cache is not None else None
            hl = cached[2] if cached is not None else estimate_molecule_half_life(s)
            half_lives[s] = round(float(hl), 2)

        return {
            "query": {
                "formula": canonical_str,
                "threshold": threshold,
            },
            "half_lives": half_lives,
            "taste": taste_sorted,
            "odor": odor_sorted,
        }


_GLOBAL_PREDICTOR: Optional[FlavriPredictor] = None


def get_predictor() -> FlavriPredictor:
    """Returns singleton FlavriPredictor instance."""
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = FlavriPredictor()
    return _GLOBAL_PREDICTOR
