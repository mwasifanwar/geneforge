import numpy as np
import re

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWYX'
AMINO_ACID_TO_ID = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
ID_TO_AMINO_ACID = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

def sequence_to_tokens(sequence):
    return [AMINO_ACID_TO_ID.get(aa, AMINO_ACID_TO_ID['X']) for aa in sequence]

def tokens_to_sequence(tokens):
    return ''.join([ID_TO_AMINO_ACID.get(token, 'X') for token in tokens])

def calculate_sequence_similarity(seq1, seq2):
    if len(seq1) != len(seq2):
        return 0.0
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1)

def validate_protein_sequence(sequence):
    return all(aa in AMINO_ACIDS for aa in sequence)

def calculate_molecular_weight(sequence):
    aa_weights = {
        'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19,
        'G': 75.07, 'H': 155.16, 'I': 131.17, 'K': 146.19, 'L': 131.17,
        'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
        'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19
    }
    return sum(aa_weights.get(aa, 110.0) for aa in sequence)

def calculate_isoelectric_point(sequence):
    aa_pkas = {
        'D': 3.9, 'E': 4.3, 'C': 8.3, 'Y': 10.1,
        'H': 6.0, 'K': 10.5, 'R': 12.5
    }
    charged_aas = [aa_pkas.get(aa) for aa in sequence if aa in aa_pkas]
    return np.median(charged_aas) if charged_aas else 7.0