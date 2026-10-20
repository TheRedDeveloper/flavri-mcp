"""
tests/test_server.py
End-to-end integration and schema tests for all 8 FastMCP tools.
Verifies model inference, in-memory screening, canonical recipe sorting,
Option 4 output schemas, sensory modality omissions, and elapsed_ms timing metadata.
"""

import pytest
import torch

from flavri_mcp.server import (
    predict_flavor,
    recommend_molecules,
    modulate_flavor,
    neutralize,
    balance_mixture,
    balance_perfume,
    scent_after,
    list_descriptors,
)
from flavri_mcp.constants import ALL_CHANNELS, NUM_CHANNELS


def test_list_descriptors():
    """Tool 8: list_descriptors returns exactly 148 descriptors and omits elapsed_ms."""
    res = list_descriptors()
    assert "descriptors" in res
    assert "count" in res
    assert res["count"] == 148
    assert len(res["descriptors"]) == 148
    assert "taste_sweet" in res["descriptors"]
    assert "taste_umami" in res["descriptors"]
    assert "odor_sweet" in res["descriptors"]
    assert "vanilla" in res["descriptors"]
    assert "elapsed_ms" not in res


def test_predict_flavor_single_and_mixture():
    """Tool 1: predict_flavor predicts taste and odor with proper sorting and elapsed_ms."""
    # Single molecule
    res_single = predict_flavor("CCO", threshold=0.1)
    assert res_single["query"]["formula"] == "CCO"
    assert "canonical_formula" not in res_single
    assert "taste" in res_single
    assert "odor" in res_single
    assert "elapsed_ms" in res_single
    assert isinstance(res_single["elapsed_ms"], float)
    assert res_single["elapsed_ms"] > 0
    assert isinstance(res_single["taste"], dict)
    assert isinstance(res_single["odor"], dict)
    assert "half_lives" in res_single
    assert isinstance(res_single["half_lives"], dict)

    # Check sorting descending
    taste_vals = list(res_single["taste"].values())
    assert taste_vals == sorted(taste_vals, reverse=True)
    odor_vals = list(res_single["odor"].values())
    assert odor_vals == sorted(odor_vals, reverse=True)

    # Ratio mixture with duplicate merging
    res_mix = predict_flavor("CCN:0.7;CCO:0.2;CCO:0.1", threshold=0.1)
    assert "CCN:0.7;CCO:0.3" in res_mix["query"]["formula"]
    assert "canonical_formula" not in res_mix
    assert "elapsed_ms" in res_mix
    assert isinstance(res_mix["taste"], dict)
    assert isinstance(res_mix["odor"], dict)


def test_predict_flavor_invalid_smiles():
    """Tool 1: Non-SMILES strings are strictly rejected."""
    with pytest.raises(ValueError):
        predict_flavor("citral:0.5;limonene:0.5")


def test_recommend_molecules():
    """Tool 2: recommend_molecules screens for matching notes and includes elapsed_ms."""
    res = recommend_molecules(
        target_descriptors=["taste_sweet", "vanilla"],
        exclude_descriptors=["sulfurous"],
        include_database=True,
        threshold=0.1,
        top_k=3,
    )
    assert "recommendations" in res
    assert "elapsed_ms" in res
    assert isinstance(res["elapsed_ms"], float)
    assert res["elapsed_ms"] > 0
    assert len(res["recommendations"]) <= 3
    if res["recommendations"]:
        first = res["recommendations"][0]
        assert "smiles" in first
        assert "half_life_hours" in first
        assert "taste" in first
        assert "odor" in first
        t_vals = list(first["taste"].values())
        assert t_vals == sorted(t_vals, reverse=True)


def test_modulate_flavor():
    """Tool 3: modulate_flavor screens candidate additives to fill unnormalized base budget and includes elapsed_ms."""
    res = modulate_flavor(
        base="CSCC:0.8",  # 20% budget for candidate additives
        reduce_descriptors=["sulfurous"],
        boost_descriptors=["vanilla"],
        include_database=True,
        threshold=0.1,
        top_k=2,
    )
    assert "recommendations" in res
    assert "elapsed_ms" in res
    assert isinstance(res["elapsed_ms"], float)
    assert res["elapsed_ms"] > 0
    assert len(res["recommendations"]) <= 2
    if res["recommendations"]:
        first = res["recommendations"][0]
        assert "additive_smiles" in first
        assert "blended_formula" in first
        assert "score" in first
        assert "taste" in first
        assert "odor" in first


def test_neutralize_modalities():
    """Tool 4: neutralize enforces proper modality omissions and includes elapsed_ms."""
    base = "CCN:0.25"
    ingredients = ["CCO", "CC(=O)OCC"]

    # Modality: "both"
    res_both = neutralize(ingredients=ingredients, base=base, modality="both", threshold=0.2)
    assert "neutralization_achieved" in res_both
    assert "canonical_formula" in res_both
    assert "residual_taste" in res_both
    assert "residual_odor" in res_both
    assert "elapsed_ms" in res_both
    assert isinstance(res_both["elapsed_ms"], float)
    assert res_both["elapsed_ms"] > 0

    # Modality: "taste" -> residual_odor omitted
    res_taste = neutralize(ingredients=ingredients, base=base, modality="taste", threshold=0.2)
    assert "residual_taste" in res_taste
    assert "residual_odor" not in res_taste
    assert "elapsed_ms" in res_taste

    # Modality: "odor" -> residual_taste omitted
    res_odor = neutralize(ingredients=ingredients, base=base, modality="odor", threshold=0.2)
    assert "residual_taste" not in res_odor
    assert "residual_odor" in res_odor
    assert "elapsed_ms" in res_odor

    # Modality: "evaporation" -> residual_taste omitted
    res_evap = neutralize(ingredients=ingredients, base=base, modality="evaporation", threshold=0.2)
    assert "residual_taste" not in res_evap
    assert "residual_odor" in res_evap
    assert "elapsed_ms" in res_evap


def test_balance_mixture():
    """Tool 5: balance_mixture optimizes ratio continuous simplex and includes elapsed_ms."""
    ingredients = ["CC(=O)OCC", "CCO"]
    target_notes = {"taste_sweet": 0.5, "fruity": 0.6}
    res = balance_mixture(
        ingredients=ingredients,
        base="CN1C=NC2=C1C(=O)N(C(=O)N2C)C:0.1",
        target_descriptors=target_notes,
        threshold=0.15,
    )
    assert "canonical_formula" in res
    assert "target_fit_mse" in res
    assert "taste" in res
    assert "odor" in res
    assert "elapsed_ms" in res
    assert isinstance(res["elapsed_ms"], float)
    assert res["elapsed_ms"] > 0


def test_balance_perfume_purely_olfactory():
    """Tool 6: balance_perfume optimizes fine fragrance timeline, omits taste, and includes elapsed_ms."""
    ingredients = ["CC1=CCC(CC1)C(=C)C", "c1ccc(cc1)CCO"]
    timeline_targets = {
        0.0: {"citrus": 0.7},
        2.0: {"floral": 0.6},
    }
    res = balance_perfume(
        ingredients=ingredients,
        timeline_targets=timeline_targets,
        threshold=0.15,
    )
    assert "canonical_formula" in res
    assert "timeline" in res
    assert "elapsed_ms" in res
    assert isinstance(res["elapsed_ms"], float)
    assert res["elapsed_ms"] > 0
    assert len(res["timeline"]) == 2
    for kf in res["timeline"]:
        assert "time_hours" in kf
        assert "remaining_mass_percentage" in kf
        assert "odor" in kf
        assert "taste" not in kf, "Fragrance-only tool must omit taste block"


def test_scent_after_purely_olfactory():
    """Tool 7: scent_after simulates evaporation checkpoints, omits taste, and includes elapsed_ms."""
    formula = "CC1=CCC(CC1)C(=C)C:0.6;c1ccc(cc1)CCO:0.4"
    hours = [0.0, 4.0]
    res = scent_after(smiles=formula, hours=hours, threshold=0.15)
    assert "timeline" in res
    assert "elapsed_ms" in res
    assert isinstance(res["elapsed_ms"], float)
    assert res["elapsed_ms"] > 0
    assert len(res["timeline"]) == 2
    for kf in res["timeline"]:
        assert "time_hours" in kf
        assert "remaining_mass_percentage" in kf
        assert "evolved_formula" in kf
        assert "odor" in kf
        assert "taste" not in kf, "Fragrance-only tool must omit taste block"
