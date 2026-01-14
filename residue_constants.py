import numpy as np
import torch
from atomworks.constants import STANDARD_AA, UNKNOWN_AA, STANDARD_RNA, UNKNOWN_RNA, UNKNOWN_DNA, STANDARD_DNA, GAP
from atomworks.ml.transforms.msa._msa_constants import AMINO_ACID_ONE_LETTER_TO_INT, RNA_NUCLEOTIDE_ONE_LETTER_TO_INT

# This is the standard residue order when coding AA type as a number.
# Reproduce it by taking 3-letter AA codes and sorting them alphabetically.
restypes_single_letter = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

restypes_three_letter = [
    'ALA',
    'ARG',
    'ASN',
    'ASP',
    'CYS',
    'GLN',
    'GLU',
    'GLY',
    'HIS',
    'ILE',
    'LEU',
    'LYS',
    'MET',
    'PHE',
    'PRO',
    'SER',
    'THR',
    'TRP',
    'TYR',
    'VAL',
]

AF3_TOKENS = (
    # 20 AA + 1 unknown AA
    *STANDARD_AA, UNKNOWN_AA,
    # 1 gap
    GAP,
    # 4 RNA
    *STANDARD_RNA, UNKNOWN_RNA,
    # 4 DNA
    *STANDARD_DNA, UNKNOWN_DNA
)

AF3_TOKENS_MAP = dict(zip(AF3_TOKENS, range(len(AF3_TOKENS))))
AF3_TOKENS_MAP[UNKNOWN_RNA] = AF3_TOKENS_MAP[UNKNOWN_AA]
AF3_TOKENS_MAP[UNKNOWN_DNA] = AF3_TOKENS_MAP[UNKNOWN_AA]

restypes_one_to_three = { a: b for a,b in zip(restypes_single_letter, restypes_three_letter) }
restypes_three_to_one = { b: a for a,b in zip(restypes_single_letter, restypes_three_letter) }

_PROTEIN_TO_ID = {
    'A': 0,
    'B': 3,  # Same as D.
    'C': 4,
    'D': 3,
    'E': 6,
    'F': 13,
    'G': 7,
    'H': 8,
    'I': 9,
    'J': 20,  # Same as unknown (X).
    'K': 11,
    'L': 10,
    'M': 12,
    'N': 2,
    'O': 20,  # Same as unknown (X).
    'P': 14,
    'Q': 5,
    'R': 1,
    'S': 15,
    'T': 16,
    'U': 4,  # Same as C.
    'V': 19,
    'W': 17,
    'X': 20,
    'Y': 18,
    'Z': 6,  # Same as E.
    '-': 21,
}

_RNA_TO_ID = {
    # Map non-standard residues to UNK_NUCLEIC (N) -> 30
    **{chr(i): 30 for i in range(ord('A'), ord('Z') + 1)},
    # Continue the RNA indices from where Protein indices left off.
    '-': 21,
    'A': 22,
    'G': 23,
    'C': 24,
    'U': 25,
}

_DNA_TO_ID = {
    # Map non-standard residues to UNK_NUCLEIC (N) -> 30
    **{chr(i): 30 for i in range(ord('A'), ord('Z') + 1)},
    # Continue the DNA indices from where DNA indices left off.
    '-': 21,
    'A': 26,
    'G': 27,
    'C': 28,
    'T': 29,
}

ATOMWORKS_TO_AF3_MSA_ENCODING_LOOKUP = np.full(42, fill_value=_PROTEIN_TO_ID['X'])

for letter, i in AMINO_ACID_ONE_LETTER_TO_INT.items():
    if letter in _PROTEIN_TO_ID:
        ATOMWORKS_TO_AF3_MSA_ENCODING_LOOKUP[i] = _PROTEIN_TO_ID[letter]

for letter, i in RNA_NUCLEOTIDE_ONE_LETTER_TO_INT.items():
    if letter in _RNA_TO_ID:
        ATOMWORKS_TO_AF3_MSA_ENCODING_LOOKUP[i] = _RNA_TO_ID[letter]
