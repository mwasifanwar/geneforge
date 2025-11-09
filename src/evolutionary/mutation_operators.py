import random
import numpy as np

class MutationOperators:
    def __init__(self):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    def point_mutation(self, sequence, mutation_rate=0.01):
        mutated = list(sequence)
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = random.choice(self.amino_acids)
        return ''.join(mutated)
    
    def insertion_mutation(self, sequence, insertion_rate=0.005):
        mutated = list(sequence)
        for i in range(len(mutated)):
            if random.random() < insertion_rate:
                new_aa = random.choice(self.amino_acids)
                mutated.insert(i, new_aa)
        return ''.join(mutated)
    
    def deletion_mutation(self, sequence, deletion_rate=0.005):
        if len(sequence) <= 10:
            return sequence
        
        mutated = list(sequence)
        i = 0
        while i < len(mutated):
            if random.random() < deletion_rate:
                mutated.pop(i)
            else:
                i += 1
        return ''.join(mutated)
    
    def block_mutation(self, sequence, block_size=3, mutation_rate=0.01):
        if len(sequence) < block_size:
            return sequence
        
        mutated = list(sequence)
        for i in range(0, len(mutated) - block_size + 1):
            if random.random() < mutation_rate:
                for j in range(block_size):
                    mutated[i + j] = random.choice(self.amino_acids)
        return ''.join(mutated)
    
    def homologous_recombination(self, seq1, seq2):
        min_len = min(len(seq1), len(seq2))
        if min_len < 10:
            return seq1, seq2
        
        crossover_point = random.randint(1, min_len - 1)
        
        child1 = seq1[:crossover_point] + seq2[crossover_point:]
        child2 = seq2[:crossover_point] + seq1[crossover_point:]
        
        return child1, child2
    
    def structure_guided_mutation(self, sequence, structure, mutation_strength=0.01):
        if len(sequence) != len(structure):
            return sequence
        
        mutated = list(sequence)
        for i in range(len(mutated)):
            if random.random() < mutation_strength:
                current_aa = mutated[i]
                similar_aas = self._get_similar_amino_acids(current_aa)
                mutated[i] = random.choice(similar_aas)
        
        return ''.join(mutated)
    
    def _get_similar_amino_acids(self, amino_acid):
        similarity_groups = {
            'A': ['A', 'V', 'L', 'I'],
            'C': ['C', 'S', 'T'],
            'D': ['D', 'E', 'N', 'Q'],
            'E': ['E', 'D', 'Q', 'N'],
            'F': ['F', 'Y', 'W'],
            'G': ['G', 'A'],
            'H': ['H', 'R', 'K'],
            'I': ['I', 'V', 'L', 'A'],
            'K': ['K', 'R', 'H'],
            'L': ['L', 'I', 'V', 'A'],
            'M': ['M', 'L', 'I'],
            'N': ['N', 'Q', 'D', 'E'],
            'P': ['P'],
            'Q': ['Q', 'N', 'E', 'D'],
            'R': ['R', 'K', 'H'],
            'S': ['S', 'T', 'C'],
            'T': ['T', 'S', 'C'],
            'V': ['V', 'I', 'L', 'A'],
            'W': ['W', 'F', 'Y'],
            'Y': ['Y', 'F', 'W']
        }
        return similarity_groups.get(amino_acid, [amino_acid])