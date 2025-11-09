import torch
import torch.nn as nn
import numpy as np

class PropertyPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=5):
        super(PropertyPredictor, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.property_heads = nn.ModuleDict({
            'stability': nn.Linear(64, 1),
            'solubility': nn.Linear(64, 1),
            'toxicity': nn.Linear(64, 1),
            'immunogenicity': nn.Linear(64, 1),
            'expressibility': nn.Linear(64, 1)
        })
    
    def forward(self, x):
        encoded = self.encoder(x)
        
        properties = {}
        for name, head in self.property_heads.items():
            properties[name] = torch.sigmoid(head(encoded))
        
        return properties

class PropertyModel:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        input_dim = self.config.get('data.amino_acid_vocab_size') * 20
        
        self.model = PropertyPredictor(input_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.encoder = SequenceEncoder()
    
    def train(self, sequences, properties_dict, epochs=200):
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for seq in sequences:
                features = self._sequence_to_features(seq).to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(features)
                
                loss = 0
                for prop_name in predictions.keys():
                    if prop_name in properties_dict and seq in properties_dict[prop_name]:
                        target = torch.tensor([properties_dict[prop_name][seq]]).float().to(self.device)
                        loss += nn.BCELoss()(predictions[prop_name], target)
                
                if loss > 0:
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"mwasifanwar Epoch {epoch}, Loss: {total_loss/len(sequences):.4f}")
    
    def predict_properties(self, sequence):
        self.model.eval()
        with torch.no_grad():
            features = self._sequence_to_features(sequence).to(self.device)
            predictions = self.model(features)
            
            return {name: float(pred.item()) for name, pred in predictions.items()}
    
    def _sequence_to_features(self, sequence):
        tokens = self.encoder.encode_sequence(sequence)
        one_hot = F.one_hot(tokens, num_classes=self.config.get('data.amino_acid_vocab_size')).float()
        
        features = one_hot.flatten()
        
        if len(features) < self.config.get('data.amino_acid_vocab_size') * 20:
            padding = self.config.get('data.amino_acid_vocab_size') * 20 - len(features)
            features = F.pad(features, (0, padding))
        elif len(features) > self.config.get('data.amino_acid_vocab_size') * 20:
            features = features[:self.config.get('data.amino_acid_vocab_size') * 20]
        
        return features.unsqueeze(0)