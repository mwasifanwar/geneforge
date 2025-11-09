import numpy as np
import torch
import torch.nn as nn

class BindingAffinityPredictor(nn.Module):
    def __init__(self):
        super(BindingAffinityPredictor, self).__init__()
        
        self.feature_net = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.affinity_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        features = self.feature_net(x)
        affinity = self.affinity_predictor(features)
        return affinity

class AffinityModel:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = BindingAffinityPredictor().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def train(self, complex_features, affinities, epochs=100):
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for features, affinity in zip(complex_features, affinities):
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                affinity_tensor = torch.tensor([affinity]).float().to(self.device)
                
                self.optimizer.zero_grad()
                prediction = self.model(features_tensor)
                loss = self.criterion(prediction, affinity_tensor)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"mwasifanwar Epoch {epoch}, Loss: {total_loss/len(complex_features):.4f}")
    
    def predict_affinity(self, protein_sequence, ligand_smiles):
        features = self._extract_complex_features(protein_sequence, ligand_smiles)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            affinity = self.model(features_tensor)
            return float(affinity.item())
    
    def _extract_complex_features(self, protein_sequence, ligand_smiles):
        from ..data_processing.sequence_encoder import SequenceEncoder
        encoder = SequenceEncoder()
        
        protein_features = encoder.calculate_sequence_features(protein_sequence)
        ligand_features = self._extract_ligand_descriptors(ligand_smiles)
        
        complex_features = np.concatenate([
            list(protein_features.values()),
            ligand_features
        ])
        
        if len(complex_features) < 100:
            complex_features = np.pad(complex_features, (0, 100 - len(complex_features)))
        elif len(complex_features) > 100:
            complex_features = complex_features[:100]
        
        return complex_features
    
    def _extract_ligand_descriptors(self, smiles):
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(20)
            
            descriptors = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.MolMR(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.FractionCsp3(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.NHOHCount(mol),
                Descriptors.NOCount(mol),
                Descriptors.NumHeteroatoms(mol),
                Descriptors.NumRadicalElectrons(mol),
                Descriptors.NumValenceElectrons(mol),
                Descriptors.NumSaturatedRings(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.NumAromaticHeterocycles(mol),
                Descriptors.NumSaturatedHeterocycles(mol),
                Descriptors.NumAmideBonds(mol)
            ]
            
            return np.array(descriptors)
        except:
            return np.zeros(20)