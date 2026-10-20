"""
flavri_mcp.recommender
Recommender and scent/flavor modulation engine.
Uses in-memory dense tensor cache for sub-millisecond candidate screening
and on-the-fly microsecond neural synergy for additive modulation.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch

from flavri_mcp.constants import (
    ALL_CHANNELS,
    CHANNEL_TO_IDX,
    TASTE_INDICES,
    ODOR_INDICES,
    NUM_CHANNELS,
)
from flavri_mcp.model import get_predictor
from flavri_mcp.cache import get_cache
from flavri_mcp.utils.chemistry import (
    canonicalize_single_smiles,
    parse_unnormalized_expression,
    format_canonical_formula,
    estimate_molecule_half_life,
)


class FlavriRecommender:
    """Recommender and modulation engine."""

    def __init__(self):
        self.predictor = get_predictor()
        self.cache = get_cache()
        self.device = self.predictor.device

    def recommend(
        self,
        target_descriptors: List[str],
        exclude_descriptors: Optional[List[str]] = None,
        additional_smiles: Optional[List[str]] = None,
        include_database: bool = True,
        min_half_life_hours: Optional[float] = None,
        max_half_life_hours: Optional[float] = None,
        threshold: float = 0.2,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Finds single molecules matching target chemosensory descriptors while avoiding excluded notes.
        """
        canonical_additional = None
        if additional_smiles:
            valid_additional = []
            for s in additional_smiles:
                try:
                    csmi = canonicalize_single_smiles(s)
                    valid_additional.append(csmi)
                except Exception:
                    continue
            canonical_additional = valid_additional

            uncached = [s for s in valid_additional if self.cache.get(s) is None]
            if uncached:
                z_new, v_new = self.predictor.encode_smiles(uncached)
                for i, s in enumerate(uncached):
                    hl = estimate_molecule_half_life(s)
                    self.cache.append(s, z_new[i], v_new[i], hl)

        recommendations = []
        if include_database:
            recommendations = self.cache.screen(
                target_descriptors=target_descriptors,
                exclude_descriptors=exclude_descriptors,
                min_half_life_hours=min_half_life_hours,
                max_half_life_hours=max_half_life_hours,
                threshold=threshold,
                top_k=top_k,
            )
        elif additional_smiles:
            # Inventory only mode: filter cache strictly to valid_additional
            matching = []
            for s in valid_additional:
                record = self.cache.get(s)
                if not record:
                    continue
                z, prof_t, hl = record
                prof = prof_t.cpu().numpy()

                # Targets
                t_scores = [prof[CHANNEL_TO_IDX[d]] for d in target_descriptors if d in CHANNEL_TO_IDX]
                if not t_scores or any(sc < threshold for sc in t_scores):
                    continue

                # Excludes
                if exclude_descriptors:
                    e_scores = [prof[CHANNEL_TO_IDX[d]] for d in exclude_descriptors if d in CHANNEL_TO_IDX]
                    if any(sc >= threshold for sc in e_scores):
                        continue

                # Half-life
                if min_half_life_hours is not None and hl < min_half_life_hours:
                    continue
                if max_half_life_hours is not None and hl > max_half_life_hours:
                    continue

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

                matching.append({
                    "smiles": s,
                    "half_life_hours": round(hl, 2),
                    "matched_descriptors": matched,
                    "score": sum(t_scores) / len(t_scores),
                    "taste": taste_sorted,
                    "odor": odor_sorted,
                })

            matching.sort(key=lambda x: -x["score"])
            recommendations = [
                {k: v for k, v in item.items() if k != "score"}
                for item in matching[:top_k]
            ]

        return {
            "query": {
                "target_descriptors": target_descriptors,
                "exclude_descriptors": exclude_descriptors,
                "additional_smiles": canonical_additional,
                "min_half_life_hours": min_half_life_hours,
                "max_half_life_hours": max_half_life_hours,
                "include_database": include_database,
                "threshold": threshold,
                "top_k": top_k,
            },
            "total_evaluated": len(self.cache),
            "total_matches": len(recommendations),
            "recommendations": recommendations,
        }

    def modulate_flavor(
        self,
        base: str,
        reduce_descriptors: List[str],
        boost_descriptors: Optional[List[str]] = None,
        additional_smiles: Optional[List[str]] = None,
        include_database: bool = True,
        min_half_life_hours: Optional[float] = None,
        max_half_life_hours: Optional[float] = None,
        threshold: float = 0.2,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Screens candidate additives to fill the remaining budget of an unnormalized base formula
        to suppress off-notes and enhance boost notes using on-the-fly microsecond synergy.
        """
        base_smiles, base_weights = parse_unnormalized_expression(base)
        canonical_base = format_canonical_formula(dict(zip(base_smiles, base_weights)))
        total_base_w = sum(base_weights)
        budget = 1.0 - total_base_w

        if budget <= 1e-4:
            raise ValueError(f"Base total weight {total_base_w:.4g} leaves no budget (< 1e-4) for additives.")

        # Ingest additional smiles if provided
        candidate_pool: List[str] = []
        if additional_smiles:
            for s in additional_smiles:
                try:
                    csmi = canonicalize_single_smiles(s)
                    if csmi not in candidate_pool:
                        candidate_pool.append(csmi)
                except Exception:
                    continue

        if include_database:
            # Fast filter across cache: pick top candidate additives
            # We look for molecules that have low presence of reduce_descriptors
            # and high presence of boost_descriptors (if any)
            boost_list = boost_descriptors or []
            candidates_screened = self.cache.screen(
                target_descriptors=boost_list if boost_list else ["taste_sweet", "vanilla"],
                exclude_descriptors=reduce_descriptors,
                min_half_life_hours=min_half_life_hours,
                max_half_life_hours=max_half_life_hours,
                threshold=0.1,
                top_k=min(200, len(self.cache)),
            )
            for c in candidates_screened:
                if c["smiles"] not in candidate_pool:
                    candidate_pool.append(c["smiles"])

            # If screened candidates are too few, add a diverse slice from cache
            if len(candidate_pool) < 50 and len(self.cache.smiles) > 0:
                for s in self.cache.smiles[:100]:
                    if s not in candidate_pool:
                        candidate_pool.append(s)

        if not candidate_pool:
            raise ValueError("No candidate additives available to screen.")

        # Compute baseline profile of unnormalized base (normalized to 1.0 for reference)
        base_norm_weights = [w / total_base_w for w in base_weights]
        base_blend_vec, _ = self.predictor.predict_components(base_smiles, base_norm_weights, cache=self.cache)
        base_blend_np = base_blend_vec.detach().cpu().numpy()

        reduce_indices = [CHANNEL_TO_IDX[d] for d in reduce_descriptors if d in CHANNEL_TO_IDX]
        boost_indices = (
            [CHANNEL_TO_IDX[d] for d in boost_descriptors if d in CHANNEL_TO_IDX]
            if boost_descriptors
            else []
        )

        candidate_scores: List[Tuple[float, str, Dict[str, float], Dict[str, float], float, str]] = []

        for cand_smi in candidate_pool:
            # Check half-life constraints
            cand_record = self.cache.get(cand_smi)
            hl = cand_record[2] if cand_record else estimate_molecule_half_life(cand_smi)
            if min_half_life_hours is not None and hl < min_half_life_hours:
                continue
            if max_half_life_hours is not None and hl > max_half_life_hours:
                continue

            # Formulate blended recipe: base + candidate additive
            # Merge candidate into recipe
            blended_dict: Dict[str, float] = {}
            for s, w in zip(base_smiles, base_weights):
                blended_dict[s] = blended_dict.get(s, 0.0) + w
            blended_dict[cand_smi] = blended_dict.get(cand_smi, 0.0) + budget

            # Canonicalize blended formula
            blended_formula = format_canonical_formula(blended_dict)

            # Predict blended profile
            mix_smiles = list(blended_dict.keys())
            mix_weights = list(blended_dict.values())
            mix_vec, _ = self.predictor.predict_components(mix_smiles, mix_weights, cache=self.cache)
            mix_np = mix_vec.detach().cpu().numpy()

            # Score improvement:
            # 1. Reduction of unwanted descriptors: base - new
            reduce_gain = 0.0
            for r_idx in reduce_indices:
                diff = base_blend_np[r_idx] - mix_np[r_idx]
                reduce_gain += diff

            # 2. Boost of wanted descriptors: new - base
            boost_gain = 0.0
            for b_idx in boost_indices:
                boost_gain += (mix_np[b_idx] - base_blend_np[b_idx])

            total_score = reduce_gain * 1.5 + boost_gain

            taste_dict = {}
            for t_idx in TASTE_INDICES:
                v = float(mix_np[t_idx])
                if v >= threshold:
                    taste_dict[ALL_CHANNELS[t_idx]] = round(v, 4)
            taste_sorted = dict(sorted(taste_dict.items(), key=lambda x: -x[1]))

            odor_dict = {}
            for o_idx in ODOR_INDICES:
                v = float(mix_np[o_idx])
                if v >= threshold:
                    odor_dict[ALL_CHANNELS[o_idx]] = round(v, 4)
            odor_sorted = dict(sorted(odor_dict.items(), key=lambda x: -x[1]))

            candidate_scores.append((
                float(total_score),
                cand_smi,
                taste_sorted,
                odor_sorted,
                float(hl),
                blended_formula,
            ))

        candidate_scores.sort(key=lambda x: -x[0])

        top_recommendations = []
        for score, cand_smi, taste_sorted, odor_sorted, hl, formula_str in candidate_scores[:top_k]:
            top_recommendations.append({
                "additive_smiles": cand_smi,
                "blended_formula": formula_str,
                "half_life_hours": round(hl, 2),
                "score": round(score, 4),
                "taste": taste_sorted,
                "odor": odor_sorted,
            })

        return {
            "query": {
                "base": canonical_base,
                "available_budget": round(budget, 4),
                "reduce_descriptors": reduce_descriptors,
                "boost_descriptors": boost_descriptors,
                "min_half_life_hours": min_half_life_hours,
                "max_half_life_hours": max_half_life_hours,
                "include_database": include_database,
                "threshold": threshold,
                "top_k": top_k,
            },
            "recommendations": top_recommendations,
        }


_GLOBAL_RECOMMENDER: Optional[FlavriRecommender] = None


def get_recommender() -> FlavriRecommender:
    """Returns singleton FlavriRecommender instance."""
    global _GLOBAL_RECOMMENDER
    if _GLOBAL_RECOMMENDER is None:
        _GLOBAL_RECOMMENDER = FlavriRecommender()
    return _GLOBAL_RECOMMENDER
