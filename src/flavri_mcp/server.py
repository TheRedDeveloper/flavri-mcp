"""
flavri_mcp.server
FastMCP Server exposing the FLAVRI 2.0 148-Dimensional Multimodal Chemosensory Foundation Model.
Unified Multimodal Sensory Space: 11 Taste/Trigeminal Channels + 137 Canonical Odor Channels.
"""

from typing import List, Dict, Any, Optional
import sys
import time
from fastmcp import FastMCP

from flavri_mcp.constants import ALL_CHANNELS
from flavri_mcp.model import get_predictor
from flavri_mcp.cache import get_cache
from flavri_mcp.recommender import get_recommender
from flavri_mcp.optimizer import get_optimizer

mcp = FastMCP(
    "FLAVRI Chemosensory MCP Server",
    instructions="FLAVRI MCP Server provides neural multimodal chemosensory prediction (11 taste/trigeminal + 137 odor channels), on-the-fly synergy, in-memory tensor screening, and ratio optimization.",
)


@mcp.tool()
def predict_flavor(
    smiles: str,
    threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Predicts the 148D chemosensory profile for a single molecule or ratio mixture.
    
    Syntax examples:
        - 'CCO' (pure molecule, normalized to 1.0)
        - 'CCN:0.7;CCO:0.3' (ratio mixture)
        - 'CCN;CCO' (omitted ratios default to equal parts)
        - 'CCO:0.2;CCN:0.7;CCO:0.1' (duplicate SMILES are automatically merged)
    
    Args:
        smiles: Valid SMILES string or semicolon-separated ratio formula.
        threshold: Perception cutoff threshold in [0, 1] (default 0.2).
    
    Returns:
        Option 4 schema containing query, prioritized taste and odor dicts sorted descending, half_lives, and elapsed_ms.
    """
    predictor = get_predictor()
    cache = get_cache()
    start_t = time.perf_counter()
    res = predictor.predict_formula(
        formula=smiles,
        threshold=threshold,
        cache=cache,
    )
    res["elapsed_ms"] = round((time.perf_counter() - start_t) * 1000.0, 4)
    return res


@mcp.tool()
def recommend_molecules(
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
    Searches the single molecule database and user inventory for candidate molecules matching desired chemosensory descriptors and persistence constraints.
    
    Args:
        target_descriptors: List of desired taste or odor notes (e.g. ['taste_umami', 'roasted']).
        exclude_descriptors: Optional list of notes to strictly avoid (e.g. ['taste_bitter', 'sulfurous']).
        additional_smiles: Optional user-supplied SMILES to evaluate and include in the search space.
        include_database: Whether to search across the ~28,000 precomputed chemical library (default True).
        min_half_life_hours: Minimum evaporation half-life in hours.
        max_half_life_hours: Maximum evaporation half-life in hours.
        threshold: Sensory perception cutoff (default 0.2).
        top_k: Maximum candidate molecules to return (default 5).
    
    Returns:
        List of matching candidate molecules with their half-life, matched notes, prioritized taste and odor dicts, and elapsed_ms.
    """
    recommender = get_recommender()
    start_t = time.perf_counter()
    res = recommender.recommend(
        target_descriptors=target_descriptors,
        exclude_descriptors=exclude_descriptors,
        additional_smiles=additional_smiles,
        include_database=include_database,
        min_half_life_hours=min_half_life_hours,
        max_half_life_hours=max_half_life_hours,
        threshold=threshold,
        top_k=top_k,
    )
    res["elapsed_ms"] = round((time.perf_counter() - start_t) * 1000.0, 4)
    return res


@mcp.tool()
def modulate_flavor(
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
    Screens candidate additive molecules to fill the remaining budget of an unnormalized base formula
    to suppress off-notes and enhance desired boost notes using on-the-fly microsecond synergy.
    
    Args:
        base: Unnormalized base SMILES formula leaving budget (sum of weights < 1.0, e.g. 'CSCC:0.8').
        reduce_descriptors: Notes to suppress (e.g. ['taste_bitter', 'sulfurous']).
        boost_descriptors: Optional notes to accentuate (e.g. ['taste_sweet', 'vanilla']).
        additional_smiles: Optional user candidate additive SMILES.
        include_database: Whether to include the ~28,000 precomputed chemical library (default True).
        min_half_life_hours: Optional half-life lower bound.
        max_half_life_hours: Optional half-life upper bound.
        threshold: Cutoff threshold (default 0.2).
        top_k: Maximum additives to return (default 5).
    
    Returns:
        Ranked candidate additives with their blended canonical formula, score, simulated profile, and elapsed_ms.
    """
    recommender = get_recommender()
    start_t = time.perf_counter()
    res = recommender.modulate_flavor(
        base=base,
        reduce_descriptors=reduce_descriptors,
        boost_descriptors=boost_descriptors,
        additional_smiles=additional_smiles,
        include_database=include_database,
        min_half_life_hours=min_half_life_hours,
        max_half_life_hours=max_half_life_hours,
        threshold=threshold,
        top_k=top_k,
    )
    res["elapsed_ms"] = round((time.perf_counter() - start_t) * 1000.0, 4)
    return res


@mcp.tool()
def neutralize(
    ingredients: List[str],
    base: str,
    modality: str = "both",
    suppress_notes: Optional[List[str]] = None,
    threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Optimizes candidate ingredient ratios to suppress offensive sensations below detection threshold.
    
    Args:
        ingredients: Candidate neutralizing ingredients (simple SMILES or complex ratioed formulas).
        base: Base formula to neutralize (sum < 1.0').
        modality: Sensory modality to neutralize: 'both' (default), 'taste', 'odor', or 'evaporation'
        suppress_notes: Optional explicit list of notes to suppress.
        threshold: Perception cutoff threshold below which sensations are neutralized (default 0.2).
    
    Returns:
        Neutralization achieved flag, max residual intensity, optimal canonical formula, residual profiles, and elapsed_ms.
    """
    optimizer = get_optimizer()
    start_t = time.perf_counter()
    res = optimizer.optimize_neutralize(
        ingredients=ingredients,
        base=base,
        modality=modality,
        suppress_notes=suppress_notes,
        threshold=threshold,
    )
    res["elapsed_ms"] = round((time.perf_counter() - start_t) * 1000.0, 4)
    return res


@mcp.tool()
def balance_mixture(
    ingredients: List[str],
    base: Optional[str] = None,
    target_descriptors: Optional[Dict[str, float]] = None,
    suppress_descriptors: Optional[List[str]] = None,
    threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Solves continuous ratio simplex optimization over candidate ingredients (+ optional fixed base)
    to achieve target sensory notes while minimizing unwanted notes.
    
    Args:
        ingredients: List of candidate ingredients (simple SMILES or complex ratioed formulas).
        base: Optional base formula (sum < 1.0).
        target_descriptors: Dict of desired notes and target scores (e.g. {'taste_umami': 0.8, 'taste_kokumi': 0.6}).
        suppress_descriptors: Optional list of notes to minimize (e.g. ['taste_bitter']).
        threshold: Detection threshold for reporting active descriptors (default 0.2).
    
    Returns:
        Optimal canonical recipe, target fit MSE, achieved taste and odor profiles sorted descending, and elapsed_ms.
    """
    optimizer = get_optimizer()
    start_t = time.perf_counter()
    res = optimizer.optimize_balance(
        ingredients=ingredients,
        base=base,
        target_descriptors=target_descriptors,
        suppress_descriptors=suppress_descriptors,
        threshold=threshold,
    )
    res["elapsed_ms"] = round((time.perf_counter() - start_t) * 1000.0, 4)
    return res


@mcp.tool()
def balance_perfume(
    ingredients: List[str],
    timeline_targets: Dict[float, Dict[str, float]],
    base: Optional[str] = None,
    suppress_descriptors: Optional[List[str]] = None,
    threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Multi-phase fine fragrance architecture optimizer across keyframe hours
    
    Args:
        ingredients: Candidate ingredients (simple SMILES or complex ratioed formulas).
        timeline_targets: Dict of {hour: {descriptor: target_score}}.
        base: Optional base formula (sum < 1.0).
        suppress_descriptors: Notes to minimize across all timeframes.
        threshold: Perception cutoff threshold (default 0.2).
    
    Returns:
        Optimal canonical perfume formula, keyframe timeline with olfactory profiles, and elapsed_ms.
    """
    optimizer = get_optimizer()
    start_t = time.perf_counter()
    res = optimizer.optimize_perfume(
        ingredients=ingredients,
        timeline_targets=timeline_targets,
        base=base,
        suppress_descriptors=suppress_descriptors,
        threshold=threshold,
    )
    res["elapsed_ms"] = round((time.perf_counter() - start_t) * 1000.0, 4)
    return res


@mcp.tool()
def scent_after(
    smiles: str,
    hours: List[float],
    threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Simulates fragrance formula evaporation decay over arbitrary time checkpoints.
    
    Args:
        smiles: Formula or simple SMILES.
        hours: List of checkpoint hours to evaluate (e.g. [0.0, 4.0]).
        threshold: Perception cutoff threshold (default 0.2).
    
    Returns:
        Timeline of evaporation checkpoints with evolved canonical formulas, remaining mass percentages, odor notes, and elapsed_ms.
    """
    optimizer = get_optimizer()
    start_t = time.perf_counter()
    res = optimizer.scent_after(
        smiles=smiles,
        hours=hours,
        threshold=threshold,
    )
    res["elapsed_ms"] = round((time.perf_counter() - start_t) * 1000.0, 4)
    return res


@mcp.tool()
def list_descriptors() -> Dict[str, Any]:
    """
    Returns all 148 descriptors in the FLAVRI vocabulary as a simple flat list.
    
    Returns:
        Dictionary with 'descriptors' (flat list of 148 names) and 'count' (148).
    """
    return {
        "descriptors": ALL_CHANNELS,
        "count": len(ALL_CHANNELS),
    }


def main():
    """CLI entrypoint for running the FastMCP server supporting stdio and sse."""
    import argparse
    parser = argparse.ArgumentParser(description="FLAVRI FastMCP Server")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "streamable-http"],
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind for SSE / HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind for SSE / HTTP transport (default: 8000)",
    )
    args, _ = parser.parse_known_args()

    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    elif args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
