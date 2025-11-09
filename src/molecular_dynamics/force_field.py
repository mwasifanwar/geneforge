import numpy as np

class ForceField:
    def __init__(self):
        self.bond_force_constants = {}
        self.angle_force_constants = {}
        self.dihedral_force_constants = {}
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        self.bond_force_constants = {
            'default': 100.0
        }
        
        self.angle_force_constants = {
            'default': 10.0
        }
        
        self.dihedral_force_constants = {
            'default': 1.0
        }
    
    def calculate_bond_energy(self, coordinates, bonds):
        energy = 0.0
        for i, j, bond_type in bonds:
            r = np.linalg.norm(coordinates[i] - coordinates[j])
            r0 = 1.5
            k = self.bond_force_constants.get(bond_type, self.bond_force_constants['default'])
            energy += 0.5 * k * (r - r0)**2
        return energy
    
    def calculate_angle_energy(self, coordinates, angles):
        energy = 0.0
        for i, j, k, angle_type in angles:
            vec1 = coordinates[i] - coordinates[j]
            vec2 = coordinates[k] - coordinates[j]
            
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            
            angle0 = 2.0
            k = self.angle_force_constants.get(angle_type, self.angle_force_constants['default'])
            energy += 0.5 * k * (angle - angle0)**2
        
        return energy
    
    def calculate_dihedral_energy(self, coordinates, dihedrals):
        energy = 0.0
        for i, j, k, l, dihedral_type in dihedrals:
            b1 = coordinates[i] - coordinates[j]
            b2 = coordinates[k] - coordinates[j]
            b3 = coordinates[l] - coordinates[k]
            
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            
            n1_norm = np.linalg.norm(n1)
            n2_norm = np.linalg.norm(n2)
            
            if n1_norm == 0 or n2_norm == 0:
                continue
            
            n1 = n1 / n1_norm
            n2 = n2 / n2_norm
            
            cos_angle = np.dot(n1, n2)
            cos_angle = np.clip(cos_angle, -1, 1)
            
            k = self.dihedral_force_constants.get(dihedral_type, self.dihedral_force_constants['default'])
            energy += k * (1 + cos_angle)
        
        return energy
    
    def infer_bonds(self, coordinates, threshold=2.0):
        bonds = []
        n_particles = len(coordinates)
        
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                distance = np.linalg.norm(coordinates[i] - coordinates[j])
                if distance < threshold:
                    bonds.append((i, j, 'default'))
        
        return bonds
    
    def infer_angles(self, coordinates, bonds):
        angles = []
        bond_dict = {}
        
        for i, j, bond_type in bonds:
            if i not in bond_dict:
                bond_dict[i] = []
            if j not in bond_dict:
                bond_dict[j] = []
            bond_dict[i].append(j)
            bond_dict[j].append(i)
        
        for center in bond_dict:
            neighbors = bond_dict[center]
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    angles.append((neighbors[i], center, neighbors[j], 'default'))
        
        return angles
    
    def infer_dihedrals(self, coordinates, bonds):
        dihedrals = []
        bond_dict = {}
        
        for i, j, bond_type in bonds:
            if i not in bond_dict:
                bond_dict[i] = []
            if j not in bond_dict:
                bond_dict[j] = []
            bond_dict[i].append(j)
            bond_dict[j].append(i)
        
        for bond in bonds:
            i, j, bond_type = bond
            for k in bond_dict[i]:
                if k != j:
                    for l in bond_dict[j]:
                        if l != i:
                            dihedrals.append((k, i, j, l, 'default'))
        
        return dihedrals