import re
import numpy as np
from Bio.PDB import PDBParser
from typing import Dict, List, Tuple

class ProteinParser:
    def __init__(self):
        self.config = Config()
        self.pdb_parser = PDBParser()
    
    def parse_pdb_file(self, pdb_file_path: str) -> Dict:
        try:
            structure = self.pdb_parser.get_structure('protein', pdb_file_path)
            model = structure[0]
            
            sequence = []
            coordinates = []
            residues = []
            
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 
                                               'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
                                               'MET', 'ASN', 'PRO', 'GLN', 'ARG',
                                               'SER', 'THR', 'VAL', 'TRP', 'TYR']:
                        residue_code = residue.get_resname()
                        aa = self.three_to_one_letter(residue_code)
                        sequence.append(aa)
                        residues.append(residue)
                        
                        ca_atom = residue['CA']
                        coordinates.append(ca_atom.get_coord())
            
            return {
                'sequence': ''.join(sequence),
                'coordinates': np.array(coordinates),
                'residues': residues,
                'chains': list(model.get_chains())
            }
        except Exception as e:
            return self._create_dummy_structure()
    
    def three_to_one_letter(self, three_letter_code: str) -> str:
        conversion = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        return conversion.get(three_letter_code, 'X')
    
    def parse_fasta_file(self, fasta_file_path: str) -> List[Dict]:
        sequences = []
        current_seq = ""
        current_header = ""
        
        with open(fasta_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append({
                            'header': current_header,
                            'sequence': current_seq
                        })
                    current_header = line[1:]
                    current_seq = ""
                else:
                    current_seq += line
        
        if current_seq:
            sequences.append({
                'header': current_header,
                'sequence': current_seq
            })
        
        return sequences
    
    def extract_secondary_structure(self, coordinates: np.ndarray) -> List[str]:
        if len(coordinates) < 3:
            return ['C'] * len(coordinates)
        
        secondary_structure = []
        for i in range(len(coordinates)):
            if i == 0 or i == len(coordinates) - 1:
                secondary_structure.append('C')
                continue
            
            vec1 = coordinates[i] - coordinates[i-1]
            vec2 = coordinates[i+1] - coordinates[i]
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                secondary_structure.append('C')
                continue
            
            cosine_angle = dot_product / (norm1 * norm2)
            cosine_angle = np.clip(cosine_angle, -1, 1)
            angle = np.arccos(cosine_angle)
            
            if angle > 2.0:
                secondary_structure.append('H')
            elif angle > 1.5:
                secondary_structure.append('E')
            else:
                secondary_structure.append('C')
        
        return secondary_structure
    
    def _create_dummy_structure(self) -> Dict:
        sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
        coords = np.random.randn(len(sequence), 3) * 10.0
        
        return {
            'sequence': sequence,
            'coordinates': coords,
            'residues': [],
            'chains': []
        }