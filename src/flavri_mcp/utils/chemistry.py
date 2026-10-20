"""
flavri_mcp.utils.chemistry
Strict SMILES validation, canonicalization, ratio formula parsing,
deterministic formula formatting, and matrix-buffered evaporation physics.
"""

from typing import List, Tuple, Dict, Any, Optional
import math
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def canonicalize_single_smiles(smi: str) -> str:
    """
    Validates and canonicalizes a single SMILES string using RDKit.
    Enforces strict SMILES: arbitrary names or invalid structures raise ValueError.
    """
    smi = smi.strip()
    if not smi:
        raise ValueError("SMILES string cannot be empty.")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: '{smi}'. Arbitrary names are not permitted.")
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def parse_ratio_expression(
    expr: str,
    normalize: bool = True,
) -> Tuple[List[str], List[float]]:
    """
    Parses a formula ratio expression into canonical SMILES and normalized weights.
    Syntax examples:
      - 'CCO' (pure molecule -> weight 1.0)
      - 'CCN:0.7;CCO:0.3'
      - 'CCN;CCO' (equal parts)
      - 'CCO:0.2;CCN:0.7;CCO:0.1' (duplicates merged: CCO:0.3, CCN:0.7)
    
    Returns:
      (canonical_smiles_list, weights_list) sorted by concentration descending.
    """
    expr = expr.strip()
    if not expr:
        raise ValueError("Formula expression cannot be empty.")

    merged_weights: Dict[str, float] = {}
    tokens = [t.strip() for t in expr.split(";") if t.strip()]
    if not tokens:
        raise ValueError("No valid tokens found in formula expression.")

    for tok in tokens:
        if ":" in tok:
            parts = tok.split(":")
            if len(parts) != 2:
                raise ValueError(f"Malformed token '{tok}': expected single ':' separator.")
            smi_raw, weight_str = parts[0].strip(), parts[1].strip()
            try:
                w = float(weight_str)
                if w < 0.0 or math.isnan(w) or math.isinf(w):
                    raise ValueError
            except ValueError:
                raise ValueError(f"Invalid weight '{weight_str}' in token '{tok}'. Must be non-negative finite number.")
        else:
            smi_raw = tok
            w = 1.0

        canonical_smi = canonicalize_single_smiles(smi_raw)
        merged_weights[canonical_smi] = merged_weights.get(canonical_smi, 0.0) + w

    total_weight = sum(merged_weights.values())
    if total_weight <= 0.0:
        raise ValueError("Total weight of formula components must be strictly positive.")

    # Sort components: primary concentration descending, secondary lexicographical SMILES ascending
    sorted_items = sorted(
        merged_weights.items(),
        key=lambda item: (-item[1], item[0]),
    )

    if normalize:
        smiles_list = [item[0] for item in sorted_items]
        weights_list = [item[1] / total_weight for item in sorted_items]
    else:
        smiles_list = [item[0] for item in sorted_items]
        weights_list = [item[1] for item in sorted_items]

    return smiles_list, weights_list


def parse_unnormalized_expression(expr: str) -> Tuple[List[str], List[float]]:
    """
    Parses an unnormalized base formula (sum of weights < 1.0).
    Raises ValueError if total fraction >= 1.0.
    """
    smiles_list, weights_list = parse_ratio_expression(expr, normalize=False)
    total_w = sum(weights_list)
    if total_w >= 1.0 - 1e-6:
        raise ValueError(
            f"Base formula total weight is {total_w:.4g} (>= 1.0). "
            "Base must be an unnormalized formula leaving budget (< 1.0) for additives."
        )
    return smiles_list, weights_list


def format_canonical_formula(components: Dict[str, float]) -> str:
    """
    Formats a dictionary of {canonical_smi: weight} into a deterministic canonical string.
    - Sorted primary: concentration descending
    - Sorted secondary: lexicographical SMILES ascending
    - Numbers formatted as f"{w:.4g}"
    - Pure single molecule with weight 1.0 formatted as bare SMILES.
    """
    if not components:
        return ""

    # Check if single molecule with weight ~1.0
    if len(components) == 1:
        smi, w = next(iter(components.items()))
        if abs(w - 1.0) < 1e-4:
            return smi

    sorted_items = sorted(
        components.items(),
        key=lambda item: (-item[1], item[0]),
    )
    tokens = [f"{smi}:{w:.4g}" for smi, w in sorted_items]
    return ";".join(tokens)


def estimate_molecule_half_life(smi: str) -> float:
    """
    Matrix-buffered evaporation half-life in hours via calibrated boiling point group contribution.
    Tb (K) = 190.0 + 19.0*N_heavy + 25.0*N_rings + 0.8*TPSA + 35.0*N_HBD + 8.0*N_HBA
    t_1/2 (hours) = clamp(10^((Tb - 425.0) / 75.0), 0.15, 72.0)
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 1.0

    n_heavy = mol.GetNumHeavyAtoms()
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    tpsa = Descriptors.TPSA(mol)
    n_hbd = rdMolDescriptors.CalcNumHBD(mol)
    n_hba = rdMolDescriptors.CalcNumHBA(mol)

    tb = 190.0 + 19.0 * n_heavy + 25.0 * n_rings + 0.8 * tpsa + 35.0 * n_hbd + 8.0 * n_hba
    exponent = (tb - 425.0) / 75.0
    # Guard against math overflow
    exponent = max(-10.0, min(10.0, exponent))
    half_life = 10.0 ** exponent
    return max(0.15, min(72.0, float(half_life)))


def evolve_composition_at_time(
    comp: Dict[str, float],
    half_lives: Dict[str, float],
    t_hours: float,
) -> Tuple[Dict[str, float], float]:
    """
    Evolves liquid mixture composition at time t (hours).
    m_i(t) = w_i(0) * 2^(-t / t_half_i)
    w_i(t) = m_i(t) / sum_j m_j(t)
    
    Returns:
      (evolved_normalized_comp, remaining_mass_percentage)
    """
    if t_hours <= 0.0:
        total_initial = sum(comp.values())
        norm_comp = {s: w / total_initial for s, w in comp.items()}
        return norm_comp, 100.0

    m_t: Dict[str, float] = {}
    for s, w0 in comp.items():
        t_half = half_lives.get(s, 1.0)
        t_half = max(0.01, t_half)
        decay = 2.0 ** (-t_hours / t_half)
        m_t[s] = w0 * decay

    sum_m_t = sum(m_t.values())
    total_w0 = sum(comp.values())
    remaining_mass_percentage = (sum_m_t / total_w0 * 100.0) if total_w0 > 0 else 0.0

    if sum_m_t > 0:
        norm_evolved = {s: m / sum_m_t for s, m in m_t.items()}
    else:
        norm_evolved = {s: 1.0 / len(m_t) for s in m_t}

    return norm_evolved, round(remaining_mass_percentage, 1)
