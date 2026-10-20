"""
flavri_mcp.constants
Unified 148-Dimensional Channel Taxonomy and Constants for FLAVRI 2.0.
11 Taste/Trigeminal Channels + 137 Odor Channels.
"""

from typing import List, Dict, Set

# 11 Canonical Taste / Trigeminal Channels
RAW_TASTE_CHANNELS: List[str] = [
    "sweet",
    "salty",
    "sour",
    "bitter",
    "umami",
    "kokumi",
    "astringent",
    "cooling",
    "heat",
    "pungency",
    "tingling",
]

TASTE_CHANNELS: List[str] = [f"taste_{t}" for t in RAW_TASTE_CHANNELS]

# 137 Canonical GoodScents-Leffingwell Odor Descriptors
CANONICAL_ODOR_DESCRIPTORS: List[str] = [
    "alcoholic",
    "aldehydic",
    "alliaceous",
    "almond",
    "amber",
    "animal",
    "anisic",
    "apple",
    "apricot",
    "aromatic",
    "balsamic",
    "banana",
    "beefy",
    "bergamot",
    "berry",
    "bitter",
    "black currant",
    "brandy",
    "burnt",
    "buttery",
    "cabbage",
    "camphoreous",
    "caramellic",
    "cedar",
    "celery",
    "chamomile",
    "cheesy",
    "cherry",
    "chocolate",
    "cinnamon",
    "citrus",
    "clean",
    "clove",
    "cocoa",
    "coconut",
    "coffee",
    "cognac",
    "cooked",
    "cooling",
    "cortex",
    "coumarinic",
    "creamy",
    "cucumber",
    "dairy",
    "dry",
    "earthy",
    "ethereal",
    "fatty",
    "fermented",
    "fishy",
    "floral",
    "fresh",
    "fruit skin",
    "fruity",
    "garlic",
    "gassy",
    "geranium",
    "grape",
    "grapefruit",
    "grassy",
    "green",
    "hawthorn",
    "hay",
    "hazelnut",
    "herbal",
    "honey",
    "hyacinth",
    "jasmin",
    "juicy",
    "ketonic",
    "lactonic",
    "lavender",
    "leafy",
    "leathery",
    "lemon",
    "lily",
    "malty",
    "meaty",
    "medicinal",
    "melon",
    "metallic",
    "milky",
    "mint",
    "muguet",
    "mushroom",
    "musk",
    "musty",
    "natural",
    "nutty",
    "oily",
    "onion",
    "orange",
    "orangeflower",
    "orris",
    "ozone",
    "peach",
    "pear",
    "phenolic",
    "pine",
    "pineapple",
    "plum",
    "popcorn",
    "potato",
    "powdery",
    "pungent",
    "radish",
    "raspberry",
    "ripe",
    "roasted",
    "rose",
    "rummy",
    "sandalwood",
    "savory",
    "sharp",
    "smoky",
    "soapy",
    "solvent",
    "sour",
    "spicy",
    "strawberry",
    "sulfurous",
    "sweaty",
    "sweet",
    "tea",
    "terpenic",
    "tobacco",
    "tomato",
    "tropical",
    "vanilla",
    "vegetable",
    "vetiver",
    "violet",
    "warm",
    "waxy",
    "weedy",
    "winey",
    "woody",
]

OVERLAP_KEYS: Set[str] = set(RAW_TASTE_CHANNELS).intersection(set(CANONICAL_ODOR_DESCRIPTORS))

ODOR_DESCRIPTOR_MAP: Dict[str, str] = {
    d: (f"odor_{d}" if d in OVERLAP_KEYS else d)
    for d in CANONICAL_ODOR_DESCRIPTORS
}

ODOR_CHANNELS: List[str] = [ODOR_DESCRIPTOR_MAP[d] for d in CANONICAL_ODOR_DESCRIPTORS]

# Unified 148 channels: 11 taste + 137 odor
ALL_CHANNELS: List[str] = TASTE_CHANNELS + ODOR_CHANNELS
assert len(ALL_CHANNELS) == 148, f"Expected 148 channels, got {len(ALL_CHANNELS)}"

CHANNEL_TO_IDX: Dict[str, int] = {ch: i for i, ch in enumerate(ALL_CHANNELS)}
IDX_TO_CHANNEL: Dict[int, str] = {i: ch for i, ch in enumerate(ALL_CHANNELS)}

NUM_CHANNELS: int = len(ALL_CHANNELS)
NUM_TASTE: int = len(TASTE_CHANNELS)
NUM_ODOR: int = len(ODOR_CHANNELS)

TASTE_INDICES: List[int] = list(range(NUM_TASTE))
ODOR_INDICES: List[int] = list(range(NUM_TASTE, NUM_CHANNELS))
