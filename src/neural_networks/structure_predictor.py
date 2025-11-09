import torch
import torch.nn as nn
import numpy as np

class StructurePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=None):
        super(StructurePredictor, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 512, 256]
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, 3))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class StructureModel:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        input_dim = self.config.get('data.amino_acid_vocab_size') + 50
        hidden_dims = self.config.get('neural_networks.structure_predictor.hidden_dims')
        
        self.model = StructurePredictor(input_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                         lr=self.config.get('neural_networks.structure_predictor.learning_rate'))
        self.criterion = nn.MSELoss()
        
        self.encoder = SequenceEncoder()
    
    def train(self, sequences, structures, epochs=500):
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for seq, struct in zip(sequences, structures):
                if len(seq) != len(struct):
                    continue
                
                features = self._extract_features(seq)
                target = torch.FloatTensor(struct).to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(features)
                
                loss = self.criterion(predictions, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 50 == 0:
                print(f"mwasifanwar Epoch {epoch}, Loss: {total_loss/len(sequences):.6f}")
    
    def predict_structure(self, sequence):
        self.model.eval()
        with torch.no_grad():
            features = self._extract_features(sequence)
            predictions = self.model(features)
            return predictions.cpu().numpy()
    
    def _extract_features(self, sequence):
        tokens = self.encoder.encode_sequence(sequence)
        one_hot = F.one_hot(tokens, num_classes=self.config.get('data.amino_acid_vocab_size')).float()
        
        position = torch.arange(len(sequence)).float() / len(sequence)
        position = position.unsqueeze(1)
        
        features = torch.cat([one_hot, position], dim=1)
        
        if len(features) < self.config.get('data.max_sequence_length'):
            padding = self.config.get('data.max_sequence_length') - len(features)
            features = F.pad(features, (0, 0, 0, padding))
        
        return features.unsqueeze(0).to(self.device)