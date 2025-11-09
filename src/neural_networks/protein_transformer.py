import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, hidden_dim = x.size()
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        output = self.output(context)
        return output, attention_weights

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        attn_output, attn_weights = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights

class ProteinTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, max_length, dropout=0.1):
        super(ProteinTransformer, self).__init__()
        self.config = Config()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = self._create_positional_encoding(max_length, hidden_dim)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.sequence_head = nn.Linear(hidden_dim, vocab_size)
        self.structure_head = nn.Linear(hidden_dim, 3)
        self.property_head = nn.Linear(hidden_dim, 10)
    
    def _create_positional_encoding(self, max_length, hidden_dim):
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        
        pos_encoding = torch.zeros(max_length, hidden_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)
    
    def forward(self, input_ids, mask=None):
        batch_size, seq_len = input_ids.size()
        
        embeddings = self.embedding(input_ids)
        embeddings = embeddings + self.positional_encoding[:, :seq_len, :]
        x = self.dropout(embeddings)
        
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        x = self.layer_norm(x)
        
        sequence_logits = self.sequence_head(x)
        structure_output = self.structure_head(x)
        property_output = self.property_head(x[:, 0, :])
        
        return {
            'sequence_logits': sequence_logits,
            'structure_output': structure_output,
            'property_output': property_output,
            'attention_weights': attention_weights
        }

class ProteinGenerator:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        vocab_size = self.config.get('data.amino_acid_vocab_size')
        hidden_dim = self.config.get('neural_networks.transformer.hidden_dim')
        num_layers = self.config.get('neural_networks.transformer.num_layers')
        num_heads = self.config.get('neural_networks.transformer.num_heads')
        max_length = self.config.get('data.max_sequence_length')
        dropout = self.config.get('neural_networks.transformer.dropout')
        
        self.model = ProteinTransformer(vocab_size, hidden_dim, num_layers, num_heads, max_length, dropout).to(self.device)
        self.encoder = SequenceEncoder()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()
    
    def generate_sequence(self, prompt_sequence="", max_length=100, temperature=1.0):
        self.model.eval()
        
        if prompt_sequence:
            input_tokens = self.encoder.encode_sequence(prompt_sequence).unsqueeze(0).to(self.device)
        else:
            input_tokens = torch.tensor([[self.encoder.vocab_size - 2]]).to(self.device)
        
        generated_sequence = prompt_sequence
        
        with torch.no_grad():
            for _ in range(max_length - len(prompt_sequence)):
                mask = self.encoder.create_attention_mask(input_tokens).to(self.device)
                
                outputs = self.model(input_tokens, mask)
                next_token_logits = outputs['sequence_logits'][:, -1, :] / temperature
                
                probabilities = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
                
                if next_token.item() == self.encoder.vocab_size - 1:
                    break
                
                generated_sequence += self.encoder.amino_acids[next_token.item()]
                input_tokens = torch.cat([input_tokens, next_token], dim=1)
                
                if input_tokens.size(1) >= self.config.get('data.max_sequence_length'):
                    break
        
        return generated_sequence
    
    def train(self, sequences, epochs=100):
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for sequence in sequences:
                if len(sequence) < 2:
                    continue
                
                input_tokens = self.encoder.encode_sequence(sequence[:-1]).unsqueeze(0).to(self.device)
                target_tokens = self.encoder.encode_sequence(sequence[1:]).to(self.device)
                
                mask = self.encoder.create_attention_mask(input_tokens).to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(input_tokens, mask)
                
                loss = self.criterion(outputs['sequence_logits'].squeeze(0), target_tokens)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"mwasifanwar Epoch {epoch}, Loss: {total_loss/len(sequences):.4f}")
    
    def predict_structure(self, sequence):
        self.model.eval()
        with torch.no_grad():
            tokens = self.encoder.encode_sequence(sequence).unsqueeze(0).to(self.device)
            mask = self.encoder.create_attention_mask(tokens).to(self.device)
            
            outputs = self.model(tokens, mask)
            coordinates = outputs['structure_output'].squeeze(0).cpu().numpy()
            
            return coordinates
    
    def predict_properties(self, sequence):
        self.model.eval()
        with torch.no_grad():
            tokens = self.encoder.encode_sequence(sequence).unsqueeze(0).to(self.device)
            mask = self.encoder.create_attention_mask(tokens).to(self.device)
            
            outputs = self.model(tokens, mask)
            properties = outputs['property_output'].squeeze(0).cpu().numpy()
            
            return {
                'stability': float(properties[0]),
                'solubility': float(properties[1]),
                'hydrophobicity': float(properties[2]),
                'flexibility': float(properties[3])
            }