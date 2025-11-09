import numpy as np
import torch
import torch.nn as nn

class DockingPredictor(nn.Module):
    def __init__(self, input_dim=256):
        super(DockingPredictor, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.docking_head = nn.Linear(128, 1)
        self.affinity_head = nn.Linear(128, 1)
    
    def forward(self, x):
        encoded = self.encoder(x)
        docking_score = self.docking_head(encoded)
        affinity_score = self.affinity_head(encoded)
        return docking_score, affinity_score

class DockingModel:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = DockingPredictor().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def train(self, protein_features, ligand_features, docking_scores, affinity_scores, epochs=100):
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for prot_feat, lig_feat, dock_score, aff_score in zip(protein_features, ligand_features, docking_scores, affinity_scores):
                features = self._combine_features(prot_feat, lig_feat).to(self.device)
                target_dock = torch.tensor([dock_score]).float().to(self.device)
                target_aff = torch.tensor([aff_score]).float().to(self.device)
                
                self.optimizer.zero_grad()
                pred_dock, pred_aff = self.model(features)
                
                loss = self.criterion(pred_dock, target_dock) + self.criterion(pred_aff, target_aff)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"mwasifanwar Epoch {epoch}, Loss: {total_loss/len(protein_features):.4f}")
    
    def predict_docking(self, protein_sequence, ligand_smiles):
        protein_features = self._extract_protein_features(protein_sequence)
        ligand_features = self._extract_ligand_features(ligand_smiles)
        
        features = self._combine_features(protein_features, ligand_features).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            docking_score, affinity_score = self.model(features)
            
            return {
                'docking_score': float(docking_score.item()),
                'binding_affinity': float(affinity_score.item()),
                'success_probability': self._score_to_probability(float(docking_score.item()))
            }
    
    def _extract_protein_features(self, sequence):
        from ..data_processing.sequence_encoder import SequenceEncoder
        encoder = SequenceEncoder()
        features = encoder.calculate_sequence_features(sequence)
        
        feature_vector = np.array([
            features['length'] / 1000,
            features['molecular_weight'] / 10000,
            features['hydrophobicity'],
            features['charge'] / 10,
            features['aromaticity']
        ])
        
        return torch.FloatTensor(feature_vector)
    
    def _extract_ligand_features(self, smiles):
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return torch.zeros(5)
            
            features = np.array([
                Descriptors.MolWt(mol) / 1000,
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol) / 10,
                Descriptors.NumHAcceptors(mol) / 10,
                Descriptors.NumRotatableBonds(mol) / 10
            ])
            
            return torch.FloatTensor(features)
        except:
            return torch.zeros(5)
    
    def _combine_features(self, protein_features, ligand_features):
        return torch.cat([protein_features, ligand_features])
    
    def _score_to_probability(self, score):
        return 1.0 / (1.0 + np.exp(-score))