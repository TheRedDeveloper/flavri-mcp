"""
scripts/build_cache.py
Precomputes dense PyTorch tensor cache (flavri_cache.pt) from FLAVRI multimodal chemical dataset
and the FLAVRI 0.8 checkpoint.
"""

import sys
from pathlib import Path
import argparse
import csv
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flavri_mcp.constants import NUM_CHANNELS
from flavri_mcp.config import CACHE_PATH, get_checkpoint_path, get_device
from flavri_mcp.model import FlavriPredictor
from flavri_mcp.utils.chemistry import canonicalize_single_smiles, estimate_molecule_half_life


def build_cache(
    input_csv: Path,
    output_cache: Path,
    batch_size: int = 256,
):
    print(f"Loading input chemical dataset from: {input_csv}")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found at {input_csv}")

    unique_smiles_set = set()
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = "smiles" if "smiles" in reader.fieldnames else reader.fieldnames[0]
        for row in reader:
            s = row[col].strip()
            if s:
                try:
                    csmi = canonicalize_single_smiles(s)
                    unique_smiles_set.add(csmi)
                except Exception:
                    continue

    all_smiles = sorted(list(unique_smiles_set))
    N = len(all_smiles)
    print(f"Extracted {N} unique valid canonical molecules.")

    device = get_device()
    print(f"Initializing FLAVRI predictor on device: {device}")
    predictor = FlavriPredictor(device=device)

    all_embeddings = []
    all_profiles = []
    all_half_lives = []

    print(f"Running batch inference (batch_size={batch_size})...")
    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), desc="Building tensor cache"):
            batch_smis = all_smiles[i : i + batch_size]
            z_batch, v_batch = predictor.encode_smiles(batch_smis)

            all_embeddings.append(z_batch.cpu())
            all_profiles.append(v_batch.cpu())

            for s in batch_smis:
                hl = estimate_molecule_half_life(s)
                all_half_lives.append(hl)

    embeddings_cat = torch.cat(all_embeddings, dim=0)  # [N, 512]
    profiles_cat = torch.cat(all_profiles, dim=0)      # [N, 148]
    half_lives_t = torch.tensor(all_half_lives, dtype=torch.float32)  # [N]

    output_cache.parent.mkdir(parents=True, exist_ok=True)
    cache_dict = {
        "canonical_smiles": all_smiles,
        "embeddings": embeddings_cat,
        "flavor_profiles": profiles_cat,
        "half_lives": half_lives_t,
    }

    print(f"Saving tensor cache to: {output_cache}")
    torch.save(cache_dict, str(output_cache))
    file_size_mb = output_cache.stat().st_size / (1024 * 1024)
    print(f"Done! Cache file size: {file_size_mb:.2f} MB ({N} molecules).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FLAVRI dense tensor cache.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROJECT_ROOT.parent / "flavri" / "data" / "collection" / "flavri_multimodal_singles.csv",
        help="Path to singles dataset CSV.",
    )
    parser.add_argument(
        "--output-cache",
        type=Path,
        default=CACHE_PATH,
        help="Target .pt file path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Inference batch size.",
    )
    args = parser.parse_args()

    build_cache(args.input_csv, args.output_cache, args.batch_size)
