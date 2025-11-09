import numpy as np
from scipy.spatial.distance import pdist, squareform

class MDAnalyzer:
    def __init__(self):
        self.config = Config()
    
    def calculate_rmsd(self, trajectory, reference=None):
        if reference is None:
            reference = trajectory[0]
        
        rmsd_values = []
        for frame in trajectory:
            if len(frame) != len(reference):
                rmsd_values.append(float('inf'))
                continue
            
            squared_diff = np.sum((frame - reference) ** 2, axis=1)
            rmsd = np.sqrt(np.mean(squared_diff))
            rmsd_values.append(rmsd)
        
        return np.array(rmsd_values)
    
    def calculate_radius_of_gyration(self, trajectory):
        rg_values = []
        for frame in trajectory:
            center_of_mass = np.mean(frame, axis=0)
            squared_distances = np.sum((frame - center_of_mass) ** 2, axis=1)
            rg = np.sqrt(np.mean(squared_distances))
            rg_values.append(rg)
        
        return np.array(rg_values)
    
    def calculate_end_to_end_distance(self, trajectory):
        e2e_values = []
        for frame in trajectory:
            if len(frame) > 1:
                distance = np.linalg.norm(frame[-1] - frame[0])
                e2e_values.append(distance)
            else:
                e2e_values.append(0.0)
        
        return np.array(e2e_values)
    
    def analyze_secondary_structure(self, trajectory):
        ss_content = []
        for frame in trajectory:
            if len(frame) < 3:
                ss_content.append({'helix': 0, 'sheet': 0, 'coil': 1})
                continue
            
            ss = self._predict_secondary_structure(frame)
            helix_count = np.sum(ss == 2)
            sheet_count = np.sum(ss == 1)
            coil_count = np.sum(ss == 0)
            
            total = len(ss)
            ss_content.append({
                'helix': helix_count / total,
                'sheet': sheet_count / total,
                'coil': coil_count / total
            })
        
        return ss_content
    
    def _predict_secondary_structure(self, coordinates):
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
    
    def calculate_contact_map_evolution(self, trajectory, threshold=8.0):
        contact_maps = []
        for frame in trajectory:
            distances = squareform(pdist(frame))
            contact_map = (distances < threshold).astype(float)
            np.fill_diagonal(contact_map, 0)
            contact_maps.append(contact_map)
        
        return np.array(contact_maps)
    
    def analyze_fluctuations(self, trajectory):
        mean_structure = np.mean(trajectory, axis=0)
        fluctuations = np.std(trajectory, axis=0)
        return {
            'mean_structure': mean_structure,
            'fluctuations': fluctuations,
            'rmsf': np.linalg.norm(fluctuations, axis=1)
        }