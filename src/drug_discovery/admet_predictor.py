import numpy as np
import torch
import torch.nn as nn

class ADMETPredictor(nn.Module):
    def __init__(self, input_dim=50):
        super(ADMETPredictor, self).__init__()
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.admet_heads = nn.ModuleDict({
            'absorption': nn.Linear(64, 1),
            'distribution': nn.Linear(64, 1),
            'metabolism': nn.Linear(64, 1),
            'excretion': nn.Linear(64, 1),
            'toxicity': nn.Linear(64, 1)
        })
    
    def forward(self, x):
        encoded = self.shared_encoder(x)
        
        predictions = {}
        for name, head in self.admet_heads.items():
            predictions[name] = torch.sigmoid(head(encoded))
        
        return predictions

class ADMETModel:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ADMETPredictor().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def train(self, compound_features, admet_properties, epochs=100):
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for features, properties in zip(compound_features, admet_properties):
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(features_tensor)
                
                loss = 0
                for prop_name, pred in predictions.items():
                    if prop_name in properties:
                        target = torch.tensor([properties[prop_name]]).float().to(self.device)
                        loss += nn.BCELoss()(pred, target)
                
                if loss > 0:
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"mwasifanwar Epoch {epoch}, Loss: {total_loss/len(compound_features):.4f}")
    
    def predict_admet(self, smiles):
        features = self._extract_compound_features(smiles)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features_tensor)
            
            return {name: float(pred.item()) for name, pred in predictions.items()}
    
    def _extract_compound_features(self, smiles):
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(50)
            
            descriptors = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.MolMR(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Lipinski.NumHDonors(mol),
                Lipinski.NumHAcceptors(mol),
                Lipinski.NumRotatableBonds(mol),
                Lipinski.NumHeteroatoms(mol),
                Lipinski.FractionCsp3(mol),
                Lipinski.NumAromaticRings(mol),
                Lipinski.NumSaturatedRings(mol),
                Lipinski.NumAliphaticRings(mol),
                Descriptors.NumValenceElectrons(mol),
                Descriptors.NumRadicalElectrons(mol),
                Descriptors.MaxPartialCharge(mol),
                Descriptors.MinPartialCharge(mol)
            ]
            
            features = np.array(descriptors)
            
            if len(features) < 50:
                features = np.pad(features, (0, 50 - len(features)))
            elif len(features) > 50:
                features = features[:50]
            
            return features
        except:
            return np.zeros(50)