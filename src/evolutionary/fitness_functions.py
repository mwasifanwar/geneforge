import numpy as np
from ..utils.bio_helpers import calculate_sequence_similarity, calculate_molecular_weight

class FitnessFunctions:
    def __init__(self):
        self.config = Config()
    
    def create_stability_fitness(self, target_stability=0.8):
        def fitness(sequence):
            from ..neural_networks.property_predictor import PropertyModel
            predictor = PropertyModel()
            properties = predictor.predict_properties(sequence)
            stability = properties.get('stability', 0.5)
            return 1.0 - abs(stability - target_stability)
        return fitness
    
    def create_binding_fitness(self, target_sequence, similarity_weight=0.7):
        def fitness(sequence):
            similarity = calculate_sequence_similarity(sequence, target_sequence)
            
            from ..neural_networks.property_predictor import PropertyModel
            predictor = PropertyModel()
            properties = predictor.predict_properties(sequence)
            stability = properties.get('stability', 0.5)
            solubility = properties.get('solubility', 0.5)
            
            return (similarity_weight * similarity + 
                   (1 - similarity_weight) * (stability + solubility) / 2)
        return fitness
    
    def create_multi_objective_fitness(self, weights=None):
        if weights is None:
            weights = {'stability': 0.3, 'solubility': 0.3, 'specificity': 0.4}
        
        def fitness(sequence):
            from ..neural_networks.property_predictor import PropertyModel
            predictor = PropertyModel()
            properties = predictor.predict_properties(sequence)
            
            score = 0
            for prop, weight in weights.items():
                score += weight * properties.get(prop, 0.5)
            
            return score
        return fitness
    
    def structure_based_fitness(self, target_structure):
        def fitness(sequence):
            from ..neural_networks.structure_predictor import StructureModel
            predictor = StructureModel()
            predicted_structure = predictor.predict_structure(sequence)
            
            if len(predicted_structure) != len(target_structure):
                return 0.0
            
            rmsd = self._calculate_rmsd(predicted_structure, target_structure)
            return 1.0 / (1.0 + rmsd)
        return fitness
    
    def _calculate_rmsd(self, coords1, coords2):
        if len(coords1) != len(coords2):
            return float('inf')
        
        squared_diff = np.sum((coords1 - coords2) ** 2, axis=1)
        return np.sqrt(np.mean(squared_diff))
    
    def create_drug_likeness_fitness(self):
        def fitness(sequence):
            mol_weight = calculate_molecular_weight(sequence)
            length = len(sequence)
            
            from ..neural_networks.property_predictor import PropertyModel
            predictor = PropertyModel()
            properties = predictor.predict_properties(sequence)
            
            toxicity_penalty = 1.0 - properties.get('toxicity', 0.0)
            stability_bonus = properties.get('stability', 0.0)
            solubility_bonus = properties.get('solubility', 0.0)
            
            size_penalty = 0.0
            if mol_weight > 5000:
                size_penalty = (mol_weight - 5000) / 10000
            
            return (stability_bonus + solubility_bonus + toxicity_penalty - size_penalty) / 3
        return fitness