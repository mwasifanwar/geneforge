import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from typing import Dict, Tuple

class StructureProcessor:
    def __init__(self):
        self.config = Config()
    
    def extract_structural_features(self, coordinates: np.ndarray) -> Dict:
        if len(coordinates) < 2:
            return self._create_dummy_features()
        
        features = {}
        
        features['radius_of_gyration'] = self._calculate_radius_of_gyration(coordinates)
        features['end_to_end_distance'] = self._calculate_end_to_end_distance(coordinates)
        features['contact_map'] = self._calculate_contact_map(coordinates)
        features['secondary_structure'] = self._predict_secondary_structure(coordinates)
        features['dihedral_angles'] = self._calculate_dihedral_angles(coordinates)
        
        return features
    
    def _calculate_radius_of_gyration(self, coordinates: np.ndarray) -> float:
        center_of_mass = np.mean(coordinates, axis=0)
        squared_distances = np.sum((coordinates - center_of_mass) ** 2, axis=1)
        return np.sqrt(np.mean(squared_distances))
    
    def _calculate_end_to_end_distance(self, coordinates: np.ndarray) -> float:
        return np.linalg.norm(coordinates[-1] - coordinates[0])
    
    def _calculate_contact_map(self, coordinates: np.ndarray, threshold: float = 8.0) -> np.ndarray:
        distances = squareform(pdist(coordinates))
        contact_map = (distances < threshold).astype(float)
        np.fill_diagonal(contact_map, 0)
        return contact_map
    
    def _predict_secondary_structure(self, coordinates: np.ndarray) -> np.ndarray:
        if len(coordinates) < 3:
            return np.array([0] * len(coordinates))
        
        ss_prediction = []
        for i in range(len(coordinates)):
            if i == 0 or i == len(coordinates) - 1:
                ss_prediction.append(0)
                continue
            
            vec1 = coordinates[i] - coordinates[i-1]
            vec2 = coordinates[i+1] - coordinates[i]
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                ss_prediction.append(0)
                continue
            
            cosine_angle = dot_product / (norm1 * norm2)
            cosine_angle = np.clip(cosine_angle, -1, 1)
            angle = np.arccos(cosine_angle)
            
            if angle > 2.0:
                ss_prediction.append(2)
            elif angle > 1.5:
                ss_prediction.append(1)
            else:
                ss_prediction.append(0)
        
        return np.array(ss_prediction)
    
    def _calculate_dihedral_angles(self, coordinates: np.ndarray) -> np.ndarray:
        if len(coordinates) < 4:
            return np.zeros(len(coordinates))
        
        dihedrals = []
        for i in range(1, len(coordinates) - 2):
            b1 = coordinates[i-1] - coordinates[i]
            b2 = coordinates[i+1] - coordinates[i]
            b3 = coordinates[i+2] - coordinates[i+1]
            
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            
            n1_norm = np.linalg.norm(n1)
            n2_norm = np.linalg.norm(n2)
            
            if n1_norm == 0 or n2_norm == 0:
                dihedrals.append(0.0)
                continue
            
            n1 = n1 / n1_norm
            n2 = n2 / n2_norm
            
            cos_angle = np.dot(n1, n2)
            cos_angle = np.clip(cos_angle, -1, 1)
            
            angle = np.arccos(cos_angle)
            dihedrals.append(angle)
        
        while len(dihedrals) < len(coordinates):
            dihedrals.append(0.0)
        
        return np.array(dihedrals[:len(coordinates)])
    
    def _create_dummy_features(self) -> Dict:
        return {
            'radius_of_gyration': 10.0,
            'end_to_end_distance': 15.0,
            'contact_map': np.zeros((50, 50)),
            'secondary_structure': np.zeros(50),
            'dihedral_angles': np.zeros(50)
        }
    
    def coordinates_to_tensor(self, coordinates: np.ndarray) -> torch.Tensor:
        if len(coordinates) == 0:
            return torch.zeros((self.config.get('data.structure_points'), 3))
        
        if len(coordinates) > self.config.get('data.structure_points'):
            indices = np.linspace(0, len(coordinates)-1, self.config.get('data.structure_points'), dtype=int)
            coordinates = coordinates[indices]
        else:
            padding = self.config.get('data.structure_points') - len(coordinates)
            coordinates = np.pad(coordinates, ((0, padding), (0, 0)), mode='constant')
        
        return torch.FloatTensor(coordinates)