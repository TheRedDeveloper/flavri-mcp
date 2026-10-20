"""
flavri_mcp.optimizer
Continuous ratio simplex optimization engine for FLAVRI 2.0.
Solves balance_mixture, neutralize across modalities (both, taste, odor, evaporation),
and balance_perfume across evaporation timeline keyframes using exact PyTorch autograd.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import torch
import torch.nn.functional as F

from flavri_mcp.constants import (
    ALL_CHANNELS,
    CHANNEL_TO_IDX,
    TASTE_CHANNELS,
    ODOR_CHANNELS,
    TASTE_INDICES,
    ODOR_INDICES,
    NUM_CHANNELS,
)
from flavri_mcp.model import get_predictor
from flavri_mcp.cache import get_cache
from flavri_mcp.utils.chemistry import (
    canonicalize_single_smiles,
    parse_ratio_expression,
    parse_unnormalized_expression,
    format_canonical_formula,
    estimate_molecule_half_life,
    evolve_composition_at_time,
)

logger = logging.getLogger("flavri_mcp.optimizer")


class MixtureOptimizer:
    """Continuous gradient-based ratio optimizer over candidate ingredients."""

    def __init__(self):
        self.predictor = get_predictor()
        self.cache = get_cache()
        self.device = self.predictor.device

    def _parse_and_flatten_setup(
        self,
        ingredients: List[str],
        base: Optional[str] = None,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, float, List[Dict[str, Any]], Optional[str], List[str]]:
        """
        Parses base and ingredients, extracts unique canonical SMILES, and builds:
          - unique_smiles: [M] unique canonical SMILES
          - b: [M] base weight vector (raw unnormalized weights)
          - C: [M, K] ingredient composition matrix (columns sum to 1.0)
          - W_rem: float remaining budget for candidate ingredients (1.0 - sum(b))
          - ing_meta: ingredient metadata
          - canonical_base_str: canonicalized base formula or None
          - canonical_ingredients: list of canonicalized ingredient formulas
        """
        if not ingredients:
            raise ValueError("At least one candidate ingredient must be provided.")

        base_weights_map: Dict[str, float] = {}
        total_base_weight = 0.0
        canonical_base_str: Optional[str] = None
        if base and base.strip():
            smis, weights = parse_unnormalized_expression(base)
            for s, w in zip(smis, weights):
                base_weights_map[s] = base_weights_map.get(s, 0.0) + w
                total_base_weight += w
            canonical_base_str = format_canonical_formula(base_weights_map)

        W_rem = 1.0 - total_base_weight
        if W_rem <= 1e-4:
            raise ValueError(
                f"Base total weight {total_base_weight:.4g} leaves no budget (< 1e-4) for candidate ingredients."
            )

        ing_meta = []
        canonical_ingredients = []
        for idx, ing_str in enumerate(ingredients):
            smis, weights = parse_ratio_expression(ing_str, normalize=True)
            comp_dict = dict(zip(smis, weights))
            canonical_ingredients.append(format_canonical_formula(comp_dict))
            ing_meta.append({
                "raw": ing_str,
                "composition": comp_dict,
            })

        K = len(ingredients)

        all_unique: List[str] = list(base_weights_map.keys())
        for ing in ing_meta:
            for s in ing["composition"].keys():
                if s not in all_unique:
                    all_unique.append(s)

        M = len(all_unique)

        b_vec = torch.zeros(M, dtype=torch.float32, device=self.device)
        for i, s in enumerate(all_unique):
            b_vec[i] = base_weights_map.get(s, 0.0)

        C_mat = torch.zeros((M, K), dtype=torch.float32, device=self.device)
        for k, ing in enumerate(ing_meta):
            for s, frac in ing["composition"].items():
                row_idx = all_unique.index(s)
                C_mat[row_idx, k] = frac

        return all_unique, b_vec, C_mat, W_rem, ing_meta, canonical_base_str, canonical_ingredients

    @torch.no_grad()
    def _prepare_tensors(self, unique_smiles: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds constituent profiles P [M, 148] and pairwise synergy deltas Delta [M, M, 148].
        Uses in-memory cache and CommutativeSynergyHead for microsecond assembly.
        All tensors are detached constants for simplex optimization.
        """
        M = len(unique_smiles)
        P_mat = torch.zeros((M, NUM_CHANNELS), dtype=torch.float32, device=self.device)
        z_tensors: List[torch.Tensor] = []

        uncached_smis = []
        uncached_idxs = []

        for i, s in enumerate(unique_smiles):
            cached = self.cache.get(s)
            if cached is not None:
                z_i, v_i, _ = cached
                P_mat[i] = v_i.to(self.device).detach()
                z_tensors.append(z_i.to(self.device).detach())
            else:
                z_tensors.append(None)
                uncached_smis.append(s)
                uncached_idxs.append(i)

        if uncached_smis:
            z_new, v_new = self.predictor.encode_smiles(uncached_smis)
            for k, orig_idx in enumerate(uncached_idxs):
                s = uncached_smis[k]
                hl = estimate_molecule_half_life(s)
                z_k = z_new[k].detach()
                v_k = v_new[k].detach()
                self.cache.append(s, z_k, v_k, hl)
                P_mat[orig_idx] = v_k
                z_tensors[orig_idx] = z_k

        # Pairwise synergy deltas
        Delta_tensor = torch.zeros((M, M, NUM_CHANNELS), dtype=torch.float32, device=self.device)
        for i in range(M):
            for j in range(i + 1, M):
                z_i = z_tensors[i].unsqueeze(0)
                z_j = z_tensors[j].unsqueeze(0)
                delta_ij = self.predictor.model.predict_synergy(z_i, z_j).squeeze(0).detach()
                Delta_tensor[i, j] = delta_ij
                Delta_tensor[j, i] = delta_ij

        return P_mat.detach(), Delta_tensor.detach()

    def _predict_from_molecule_weights(
        self,
        w: torch.Tensor,
        P_mat: torch.Tensor,
        Delta_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        w: [..., M] molecule weights summing to 1.0.
        Returns: [..., 148] clamped in [0.0, 1.0].
        """
        linear_part = torch.matmul(w, P_mat)
        # 4 * sum_{i < j} w_i * w_j * Delta_ij = 2 * w^T Delta w
        synergy_part = 2.0 * torch.einsum("...i, ...j, ijd -> ...d", w, w, Delta_tensor)
        return torch.clamp(linear_part + synergy_part, min=0.0, max=1.0)

    def optimize_balance(
        self,
        ingredients: List[str],
        base: Optional[str] = None,
        target_descriptors: Optional[Dict[str, float]] = None,
        suppress_descriptors: Optional[List[str]] = None,
        threshold: float = 0.2,
    ) -> Dict[str, Any]:
        """Continuous ratio optimization to match target descriptor intensities and suppress off-notes."""
        unique_smis, b_vec, C_mat, W_rem, ing_meta, canonical_base_str, canonical_ingredients = self._parse_and_flatten_setup(ingredients, base)
        P_mat, Delta_tensor = self._prepare_tensors(unique_smis)
        K = len(ingredients)

        targets_dict = {}
        if target_descriptors:
            for d, val in target_descriptors.items():
                if d in CHANNEL_TO_IDX:
                    targets_dict[CHANNEL_TO_IDX[d]] = float(val)

        suppress_idxs = []
        if suppress_descriptors:
            for d in suppress_descriptors:
                if d in CHANNEL_TO_IDX:
                    suppress_idxs.append(CHANNEL_TO_IDX[d])

        if not targets_dict and not suppress_idxs:
            raise ValueError("At least one target descriptor or suppress descriptor must be provided.")

        def loss_fn(alpha: torch.Tensor) -> torch.Tensor:
            w_cand = W_rem * F.softmax(alpha, dim=0)
            w_mol = b_vec + torch.matmul(C_mat, w_cand)
            pred = self._predict_from_molecule_weights(w_mol, P_mat, Delta_tensor)

            loss = torch.tensor(0.0, device=self.device)
            for idx, target_val in targets_dict.items():
                diff = pred[idx] - target_val
                loss = loss + 10.0 * (diff ** 2)

            for s_idx in suppress_idxs:
                penalty = F.relu(pred[s_idx] - threshold)
                loss = loss + 15.0 * (penalty ** 2) + 2.0 * (pred[s_idx] ** 2)

            # Slight entropy regularizer to avoid single ingredient domination unless necessary
            p = F.softmax(alpha, dim=0)
            entropy = -torch.sum(p * torch.log(p + 1e-9))
            loss = loss - 0.01 * entropy
            return loss

        # Multi-start Adam optimization
        best_loss = float("inf")
        best_alpha: Optional[torch.Tensor] = None

        inits = [torch.zeros(K, device=self.device)]
        for _ in range(4):
            inits.append(torch.randn(K, device=self.device) * 1.5)

        for init_alpha in inits:
            alpha = init_alpha.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([alpha], lr=0.1)
            for _ in range(120):
                optimizer.zero_grad()
                loss = loss_fn(alpha)
                loss.backward()
                optimizer.step()

            final_loss = loss_fn(alpha).item()
            if final_loss < best_loss:
                best_loss = final_loss
                best_alpha = alpha.detach().clone()

        best_w_cand = W_rem * F.softmax(best_alpha, dim=0)
        best_w_mol = b_vec + torch.matmul(C_mat, best_w_cand)
        final_pred = self._predict_from_molecule_weights(best_w_mol, P_mat, Delta_tensor).detach().cpu().numpy()

        # Build canonical formula
        mol_comp = {s: float(best_w_mol[i].item()) for i, s in enumerate(unique_smis) if best_w_mol[i] > 1e-4}
        canonical_str = format_canonical_formula(mol_comp)

        # Compute MSE on targets
        mse = 0.0
        if targets_dict:
            mse = sum((final_pred[idx] - val) ** 2 for idx, val in targets_dict.items()) / len(targets_dict)

        taste_dict = {}
        for idx in TASTE_INDICES:
            v = float(final_pred[idx])
            if v >= threshold:
                taste_dict[ALL_CHANNELS[idx]] = round(v, 4)
        taste_sorted = dict(sorted(taste_dict.items(), key=lambda x: -x[1]))

        odor_dict = {}
        for idx in ODOR_INDICES:
            v = float(final_pred[idx])
            if v >= threshold:
                odor_dict[ALL_CHANNELS[idx]] = round(v, 4)
        odor_sorted = dict(sorted(odor_dict.items(), key=lambda x: -x[1]))

        return {
            "query": {
                "base": canonical_base_str,
                "ingredients": canonical_ingredients,
                "target_descriptors": target_descriptors,
                "suppress_descriptors": suppress_descriptors,
                "threshold": threshold,
            },
            "canonical_formula": canonical_str,
            "target_fit_mse": round(float(mse), 6),
            "taste": taste_sorted,
            "odor": odor_sorted,
        }

    def optimize_neutralize(
        self,
        ingredients: List[str],
        base: str,
        modality: str = "both",
        suppress_notes: Optional[List[str]] = None,
        threshold: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Solves continuous ratio optimization to suppress offensive sensations below detection threshold.
        Modalities: 'both', 'taste', 'odor', or 'evaporation'.
        """
        unique_smis, b_vec, C_mat, W_rem, ing_meta, canonical_base_str, canonical_ingredients = self._parse_and_flatten_setup(ingredients, base)
        P_mat, Delta_tensor = self._prepare_tensors(unique_smis)
        K = len(ingredients)

        # Select target channels based on modality
        if modality == "taste":
            eval_indices = TASTE_INDICES
        elif modality in ("odor", "evaporation"):
            eval_indices = ODOR_INDICES
        elif modality == "both":
            eval_indices = list(range(NUM_CHANNELS))
        else:
            raise ValueError(f"Unknown modality '{modality}'. Must be one of: 'both', 'taste', 'odor', 'evaporation'.")

        if suppress_notes:
            explicit_idxs = [CHANNEL_TO_IDX[d] for d in suppress_notes if d in CHANNEL_TO_IDX]
            if explicit_idxs:
                eval_indices = [idx for idx in eval_indices if idx in explicit_idxs]

        half_lives = {s: self.cache.get(s)[2] if self.cache.get(s) else estimate_molecule_half_life(s) for s in unique_smis}

        def loss_fn(alpha: torch.Tensor) -> torch.Tensor:
            w_cand = W_rem * F.softmax(alpha, dim=0)
            w_mol = b_vec + torch.matmul(C_mat, w_cand)

            if modality == "evaporation":
                # Quadrature over evaporation lifespan: t in [0h, 2h, 4h, 8h]
                loss = torch.tensor(0.0, device=self.device)
                timepoints = [0.0, 2.0, 4.0, 8.0]
                weights_quad = [1.0, 1.0, 1.0, 1.0]

                for t_hr, quad_w in zip(timepoints, weights_quad):
                    # compute evolved weights at time t
                    decays = torch.tensor(
                        [2.0 ** (-t_hr / max(0.01, half_lives[s])) for s in unique_smis],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    m_t = w_mol * decays
                    sum_m = m_t.sum()
                    w_t = m_t / (sum_m + 1e-9)

                    pred = self._predict_from_molecule_weights(w_t, P_mat, Delta_tensor)
                    active_above = F.relu(pred[eval_indices] - threshold)
                    loss = loss + quad_w * (torch.sum(active_above ** 2) * 20.0 + torch.sum(pred[eval_indices] ** 2) * 5.0)
                return loss
            else:
                pred = self._predict_from_molecule_weights(w_mol, P_mat, Delta_tensor)
                active_above = F.relu(pred[eval_indices] - threshold)
                loss = torch.sum(active_above ** 2) * 25.0 + torch.sum(pred[eval_indices] ** 2) * 5.0
                return loss

        best_loss = float("inf")
        best_alpha: Optional[torch.Tensor] = None

        inits = [torch.zeros(K, device=self.device)]
        for _ in range(4):
            inits.append(torch.randn(K, device=self.device) * 1.5)

        for init_alpha in inits:
            alpha = init_alpha.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([alpha], lr=0.1)
            for _ in range(120):
                optimizer.zero_grad()
                loss = loss_fn(alpha)
                loss.backward()
                optimizer.step()

            final_loss = loss_fn(alpha).item()
            if final_loss < best_loss:
                best_loss = final_loss
                best_alpha = alpha.detach().clone()

        best_w_cand = W_rem * F.softmax(best_alpha, dim=0)
        best_w_mol = b_vec + torch.matmul(C_mat, best_w_cand)
        final_pred = self._predict_from_molecule_weights(best_w_mol, P_mat, Delta_tensor).detach().cpu().numpy()

        mol_comp = {s: float(best_w_mol[i].item()) for i, s in enumerate(unique_smis) if best_w_mol[i] > 1e-4}
        canonical_str = format_canonical_formula(mol_comp)

        max_residual = float(max(final_pred[eval_indices]))
        neutralization_achieved = max_residual <= threshold

        result: Dict[str, Any] = {
            "query": {
                "base": canonical_base_str,
                "available_budget": round(W_rem, 4),
                "ingredients": canonical_ingredients,
                "modality": modality,
                "threshold": threshold,
            },
            "neutralization_achieved": neutralization_achieved,
            "max_residual_intensity": round(max_residual, 4),
            "canonical_formula": canonical_str,
        }

        # Format residual profiles according to modality omissions
        if modality in ("both", "taste"):
            residual_taste = {}
            for idx in TASTE_INDICES:
                v = float(final_pred[idx])
                if v >= threshold:
                    residual_taste[ALL_CHANNELS[idx]] = round(v, 4)
            result["residual_taste"] = dict(sorted(residual_taste.items(), key=lambda x: -x[1]))

        if modality in ("both", "odor", "evaporation"):
            residual_odor = {}
            for idx in ODOR_INDICES:
                v = float(final_pred[idx])
                if v >= threshold:
                    residual_odor[ALL_CHANNELS[idx]] = round(v, 4)
            result["residual_odor"] = dict(sorted(residual_odor.items(), key=lambda x: -x[1]))

        return result

    def optimize_perfume(
        self,
        ingredients: List[str],
        timeline_targets: Dict[float, Dict[str, float]],
        base: Optional[str] = None,
        suppress_descriptors: Optional[List[str]] = None,
        threshold: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Multi-phase fine fragrance timeline optimizer across arbitrary hours (e.g. 0h opening, 2h heart, 8h drydown).
        Purely olfactory: taste is completely omitted.
        """
        unique_smis, b_vec, C_mat, W_rem, ing_meta, canonical_base_str, canonical_ingredients = self._parse_and_flatten_setup(ingredients, base)
        P_mat, Delta_tensor = self._prepare_tensors(unique_smis)
        K = len(ingredients)

        half_lives = {s: self.cache.get(s)[2] if self.cache.get(s) else estimate_molecule_half_life(s) for s in unique_smis}
        hl_tensor = torch.tensor([max(0.01, half_lives[s]) for s in unique_smis], dtype=torch.float32, device=self.device)

        suppress_idxs = [CHANNEL_TO_IDX[d] for d in suppress_descriptors if d in CHANNEL_TO_IDX] if suppress_descriptors else []

        def loss_fn(alpha: torch.Tensor) -> torch.Tensor:
            w_cand = W_rem * F.softmax(alpha, dim=0)
            w_mol = b_vec + torch.matmul(C_mat, w_cand)

            total_loss = torch.tensor(0.0, device=self.device)

            for hr_float, target_dict in timeline_targets.items():
                decay = 2.0 ** (-float(hr_float) / hl_tensor)
                m_t = w_mol * decay
                w_t = m_t / (m_t.sum() + 1e-9)

                pred = self._predict_from_molecule_weights(w_t, P_mat, Delta_tensor)

                for d, target_val in target_dict.items():
                    if d in CHANNEL_TO_IDX:
                        idx = CHANNEL_TO_IDX[d]
                        diff = pred[idx] - target_val
                        total_loss = total_loss + 10.0 * (diff ** 2)

                for s_idx in suppress_idxs:
                    pen = F.relu(pred[s_idx] - threshold)
                    total_loss = total_loss + 15.0 * (pen ** 2)

            return total_loss

        best_loss = float("inf")
        best_alpha: Optional[torch.Tensor] = None

        inits = [torch.zeros(K, device=self.device)]
        for _ in range(4):
            inits.append(torch.randn(K, device=self.device) * 1.5)

        for init_alpha in inits:
            alpha = init_alpha.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([alpha], lr=0.1)
            for _ in range(120):
                optimizer.zero_grad()
                loss = loss_fn(alpha)
                loss.backward()
                optimizer.step()

            final_loss = loss_fn(alpha).item()
            if final_loss < best_loss:
                best_loss = final_loss
                best_alpha = alpha.detach().clone()

        best_w_cand = W_rem * F.softmax(best_alpha, dim=0)
        best_w_mol = b_vec + torch.matmul(C_mat, best_w_cand)

        initial_comp = {s: float(best_w_mol[i].item()) for i, s in enumerate(unique_smis) if best_w_mol[i] > 1e-4}
        canonical_str = format_canonical_formula(initial_comp)

        # Build timeline keyframes
        timeline_results = []
        for hr_float in sorted(timeline_targets.keys()):
            evolved_comp, rem_pct = evolve_composition_at_time(initial_comp, half_lives, float(hr_float))

            # Build weight tensor for evolved composition
            w_evolved_tensor = torch.zeros(len(unique_smis), dtype=torch.float32, device=self.device)
            for i, s in enumerate(unique_smis):
                w_evolved_tensor[i] = evolved_comp.get(s, 0.0)

            pred_t = self._predict_from_molecule_weights(w_evolved_tensor, P_mat, Delta_tensor).detach().cpu().numpy()

            odor_dict = {}
            for idx in ODOR_INDICES:
                v = float(pred_t[idx])
                if v >= threshold:
                    odor_dict[ALL_CHANNELS[idx]] = round(v, 4)
            odor_sorted = dict(sorted(odor_dict.items(), key=lambda x: -x[1]))

            timeline_results.append({
                "time_hours": float(hr_float),
                "remaining_mass_percentage": rem_pct,
                "odor": odor_sorted,
            })

        return {
            "query": {
                "base": canonical_base_str,
                "ingredients": canonical_ingredients,
                "timeline_targets": timeline_targets,
                "suppress_descriptors": suppress_descriptors,
                "threshold": threshold,
            },
            "canonical_formula": canonical_str,
            "timeline": timeline_results,
        }

    def scent_after(
        self,
        smiles: str,
        hours: List[float],
        threshold: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Simulates fragrance formula evaporation decay over arbitrary time checkpoints.
        Purely olfactory: taste is completely omitted.
        """
        comp_smiles, comp_weights = parse_ratio_expression(smiles, normalize=True)
        initial_comp = dict(zip(comp_smiles, comp_weights))

        half_lives = {s: self.cache.get(s)[2] if self.cache.get(s) else estimate_molecule_half_life(s) for s in comp_smiles}

        unique_smis = list(initial_comp.keys())
        P_mat, Delta_tensor = self._prepare_tensors(unique_smis)

        timeline_results = []
        for hr in sorted(hours):
            hr_val = float(hr)
            evolved_comp, rem_pct = evolve_composition_at_time(initial_comp, half_lives, hr_val)
            evolved_formula_str = format_canonical_formula(evolved_comp)

            w_evolved_tensor = torch.zeros(len(unique_smis), dtype=torch.float32, device=self.device)
            for i, s in enumerate(unique_smis):
                w_evolved_tensor[i] = evolved_comp.get(s, 0.0)

            pred_t = self._predict_from_molecule_weights(w_evolved_tensor, P_mat, Delta_tensor).detach().cpu().numpy()

            odor_dict = {}
            for idx in ODOR_INDICES:
                v = float(pred_t[idx])
                if v >= threshold:
                    odor_dict[ALL_CHANNELS[idx]] = round(v, 4)
            odor_sorted = dict(sorted(odor_dict.items(), key=lambda x: -x[1]))

            timeline_results.append({
                "time_hours": hr_val,
                "remaining_mass_percentage": rem_pct,
                "evolved_formula": evolved_formula_str,
                "odor": odor_sorted,
            })

        canonical_query_formula = format_canonical_formula(initial_comp)

        return {
            "query": {
                "formula": canonical_query_formula,
                "hours": hours,
                "threshold": threshold,
            },
            "timeline": timeline_results,
        }


_GLOBAL_OPTIMIZER: Optional[MixtureOptimizer] = None


def get_optimizer() -> MixtureOptimizer:
    """Returns singleton MixtureOptimizer instance."""
    global _GLOBAL_OPTIMIZER
    if _GLOBAL_OPTIMIZER is None:
        _GLOBAL_OPTIMIZER = MixtureOptimizer()
    return _GLOBAL_OPTIMIZER
