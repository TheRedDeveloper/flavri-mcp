"""
flavri_mcp.utils.featurizer
Molecular graph featurizer for small molecules.
Produces atom features (d_atom = 133), directed bond features (d_bond = 14),
and Morgan ECFP4 fingerprints (512-bit) with dense index tensors matching FLAVRI 2.0 D-MPNN.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)

MAX_ATOMIC_NUM = 100
DEGREE_CHOICES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGE_CHOICES = [-2, -1, 0, 1, 2]
CHIRAL_TAG_CHOICES = [0, 1, 2, 3]
NUM_HS_CHOICES = [0, 1, 2, 3, 4]
HYBRIDIZATION_CHOICES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

ATOM_FDIM = 101 + 7 + 6 + 5 + 6 + 6 + 1 + 1  # 133
BOND_FDIM = 4 + 1 + 1 + 7 + 1                 # 14


def one_hot_unk(val: Any, choices: List[Any]) -> List[float]:
    """One-hot encodes value with trailing slot for unknown / out-of-bounds."""
    encoding = [0.0] * (len(choices) + 1)
    if val in choices:
        encoding[choices.index(val)] = 1.0
    else:
        encoding[-1] = 1.0
    return encoding


def get_atom_features(atom: Chem.rdchem.Atom) -> List[float]:
    """Computes 133-dimensional atom feature vector."""
    atomic_num = atom.GetAtomicNum()
    at_choices = list(range(1, MAX_ATOMIC_NUM + 1))

    feats = (
        one_hot_unk(atomic_num, at_choices)
        + one_hot_unk(atom.GetTotalDegree(), DEGREE_CHOICES)
        + one_hot_unk(atom.GetFormalCharge(), FORMAL_CHARGE_CHOICES)
        + one_hot_unk(int(atom.GetChiralTag()), CHIRAL_TAG_CHOICES)
        + one_hot_unk(int(atom.GetTotalNumHs()), NUM_HS_CHOICES)
        + one_hot_unk(atom.GetHybridization(), HYBRIDIZATION_CHOICES)
        + [1.0 if atom.GetIsAromatic() else 0.0]
        + [atom.GetMass() * 0.01]
    )
    return feats


def get_bond_features(bond: Optional[Chem.rdchem.Bond]) -> List[float]:
    """Computes 14-dimensional bond feature vector."""
    if bond is None:
        return [0.0] * BOND_FDIM

    bt = bond.GetBondType()
    stereo_choices = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]

    feats = (
        [
            1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0,
            1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0,
            1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0,
            1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0,
        ]
        + [1.0 if bond.GetIsConjugated() else 0.0]
        + [1.0 if bond.IsInRing() else 0.0]
        + one_hot_unk(bond.GetStereo(), stereo_choices)
        + [1.0 if bond.GetBondDir() != Chem.rdchem.BondDir.NONE else 0.0]
    )
    return feats


class MolGraph:
    """Directed graph representation of a single molecule for D-MPNN."""

    def __init__(self, mol: Chem.rdchem.Mol, smiles: str = ""):
        self.smiles = smiles
        self.n_atoms = mol.GetNumAtoms()
        self.f_atoms: List[List[float]] = [get_atom_features(atom) for atom in mol.GetAtoms()]

        # Directed bonds: index 0 reserved for padding
        self.f_bonds: List[List[float]] = [[0.0] * BOND_FDIM]
        self.u_atoms: List[int] = [0]
        self.v_atoms: List[int] = [0]
        self.b2rev: List[int] = [0]

        self.a2b: List[List[int]] = [[] for _ in range(self.n_atoms)]
        self.b2b: List[List[int]] = [[]]

        n_bonds = 0
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            fbond = get_bond_features(bond)

            # Directed bond 1: u -> v
            n_bonds += 1
            b1 = n_bonds
            self.f_bonds.append(fbond)
            self.u_atoms.append(u)
            self.v_atoms.append(v)
            self.a2b[v].append(b1)

            # Directed bond 2: v -> u
            n_bonds += 1
            b2 = n_bonds
            self.f_bonds.append(fbond)
            self.u_atoms.append(v)
            self.v_atoms.append(u)
            self.a2b[u].append(b2)

            self.b2rev.append(b2)
            self.b2rev.append(b1)

        self.n_bonds = n_bonds

        for b in range(1, self.n_bonds + 1):
            u = self.u_atoms[b]
            rev = self.b2rev[b]
            incoming_to_u = [k for k in self.a2b[u] if k != rev]
            self.b2b.append(incoming_to_u)

        # 512-bit Morgan fingerprint
        fp_np = np.zeros((512,), dtype=np.float32)
        fp_explicit = _MORGAN_GEN.GetFingerprint(mol)
        for idx in fp_explicit.GetOnBits():
            fp_np[idx] = 1.0
        self.fp = fp_np


class BatchMolGraph:
    """
    Batched directed molecular graphs with dense index tensors for GPU execution.
    """

    def __init__(self, mol_graphs: List[MolGraph], device: torch.device = torch.device("cpu")):
        self.smiles_batch = [g.smiles for g in mol_graphs]
        self.batch_size = len(mol_graphs)
        self.device = device

        f_atoms_all = []
        f_bonds_all = [[0.0] * BOND_FDIM]

        a2g_list = []
        b2a_list = [0]
        b2u_list = [0]

        total_atoms = 0
        total_bonds = 0

        max_incoming_bonds = 1
        for g in mol_graphs:
            for incoming in g.b2b[1:]:
                if len(incoming) > max_incoming_bonds:
                    max_incoming_bonds = len(incoming)

        max_incoming_atoms = 1
        for g in mol_graphs:
            for incoming in g.a2b:
                if len(incoming) > max_incoming_atoms:
                    max_incoming_atoms = len(incoming)

        b2b_list = [[]]
        a2b_list = []

        for g_idx, g in enumerate(mol_graphs):
            f_atoms_all.extend(g.f_atoms)
            for _ in range(g.n_atoms):
                a2g_list.append(g_idx)

            for b in range(1, g.n_bonds + 1):
                f_bonds_all.append(g.f_bonds[b])
                b2a_list.append(g.v_atoms[b] + total_atoms)
                b2u_list.append(g.u_atoms[b] + total_atoms)
                incoming = [inc + total_bonds for inc in g.b2b[b]]
                b2b_list.append(incoming)

            for a in range(g.n_atoms):
                incoming_to_a = [inc + total_bonds for inc in g.a2b[a]]
                a2b_list.append(incoming_to_a)

            total_atoms += g.n_atoms
            total_bonds += g.n_bonds

        self.total_atoms = total_atoms
        self.total_bonds = total_bonds

        b2b_tensor = torch.zeros((total_bonds + 1, max_incoming_bonds), dtype=torch.long)
        for b, incoming in enumerate(b2b_list):
            if incoming:
                b2b_tensor[b, : len(incoming)] = torch.tensor(incoming, dtype=torch.long)

        a2b_tensor = torch.zeros((total_atoms, max_incoming_atoms), dtype=torch.long)
        for a, incoming in enumerate(a2b_list):
            if incoming:
                a2b_tensor[a, : len(incoming)] = torch.tensor(incoming, dtype=torch.long)

        self.f_atoms = torch.tensor(f_atoms_all, dtype=torch.float32, device=device)
        self.f_bonds = torch.tensor(f_bonds_all, dtype=torch.float32, device=device)
        self.a2g = torch.tensor(a2g_list, dtype=torch.long, device=device)
        self.b2a = torch.tensor(b2a_list, dtype=torch.long, device=device)
        self.b2u = torch.tensor(b2u_list, dtype=torch.long, device=device)
        self.b2b = b2b_tensor.to(device)
        self.a2b = a2b_tensor.to(device)
        self.fps = torch.tensor(np.stack([g.fp for g in mol_graphs]), dtype=torch.float32, device=device)


def smiles_to_graph_batch(smiles_list: List[str], device: torch.device) -> BatchMolGraph:
    """Builds a batched molecular graph from a list of SMILES."""
    mol_graphs = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"Cannot featurize invalid SMILES: '{s}'")
        mol_graphs.append(MolGraph(mol, smiles=s))
    return BatchMolGraph(mol_graphs, device)
