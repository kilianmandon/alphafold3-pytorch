import torch

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

restypes_one_to_three = { a: b for a,b in zip(restypes_single_letter, restypes_three_letter) }

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