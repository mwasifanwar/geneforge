import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class SequenceVisualizer:
    def __init__(self):
        self.config = Config()
    
    def plot_sequence_logo(self, sequences, title="Sequence Logo"):
        if not sequences:
            return
        
        sequence_length = len(sequences[0])
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        position_matrix = np.zeros((len(amino_acids), sequence_length))
        
        for seq in sequences:
            for i, aa in enumerate(seq):
                if aa in amino_acids:
                    j = amino_acids.index(aa)
                    position_matrix[j, i] += 1
        
        position_matrix = position_matrix / len(sequences)
        
        fig, ax = plt.subplots(figsize=(15, 4))
        
        for i in range(sequence_length):
            total_height = 0
            sorted_indices = np.argsort(position_matrix[:, i])
            
            for j in sorted_indices:
                height = position_matrix[j, i]
                if height > 0:
                    aa = amino_acids[j]
                    ax.text(i, total_height + height/2, aa, 
                           ha='center', va='center', fontsize=8,
                           fontweight='bold')
                    ax.bar(i, height, bottom=total_height, 
                          color=self._get_aa_color(aa), alpha=0.7)
                    total_height += height
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.set_xticks(range(sequence_length))
        ax.set_ylim(0, 1)
        
        plt.show()
    
    def _get_aa_color(self, amino_acid):
        color_map = {
            'A': 'gray', 'C': 'yellow', 'D': 'red', 'E': 'red',
            'F': 'blue', 'G': 'gray', 'H': 'cyan', 'I': 'green',
            'K': 'blue', 'L': 'green', 'M': 'green', 'N': 'magenta',
            'P': 'yellow', 'Q': 'magenta', 'R': 'blue', 'S': 'orange',
            'T': 'orange', 'V': 'green', 'W': 'blue', 'Y': 'cyan'
        }
        return color_map.get(amino_acid, 'black')
    
    def plot_sequence_similarity(self, sequences, title="Sequence Similarity"):
        n_seqs = len(sequences)
        similarity_matrix = np.zeros((n_seqs, n_seqs))
        
        for i in range(n_seqs):
            for j in range(n_seqs):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self._calculate_similarity(sequences[i], sequences[j])
                    similarity_matrix[i, j] = sim
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix, annot=True, cmap='viridis',
                   xticklabels=range(1, n_seqs+1),
                   yticklabels=range(1, n_seqs+1))
        plt.title(title)
        plt.xlabel('Sequence Index')
        plt.ylabel('Sequence Index')
        plt.show()
    
    def _calculate_similarity(self, seq1, seq2):
        if len(seq1) != len(seq2):
            return 0.0
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)
    
    def plot_evolution_progress(self, fitness_history, title="Evolution Progress"):
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history, 'b-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_property_distribution(self, sequences, property_name, title="Property Distribution"):
        from ..neural_networks.property_predictor import PropertyModel
        
        predictor = PropertyModel()
        properties = [predictor.predict_properties(seq)[property_name] for seq in sequences]
        
        plt.figure(figsize=(8, 6))
        plt.hist(properties, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel(property_name)
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()