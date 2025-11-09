import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.neural_networks.protein_transformer import ProteinGenerator
from src.data_processing.sequence_encoder import SequenceEncoder

class TestTransformer(unittest.TestCase):
    def test_sequence_encoding(self):
        encoder = SequenceEncoder()
        sequence = "MVLSPADKTNVK"
        tokens = encoder.encode_sequence(sequence)
        decoded = encoder.decode_sequence(tokens)
        self.assertEqual(sequence, decoded.replace('X', ''))
    
    def test_protein_generation(self):
        generator = ProteinGenerator()
        sequence = generator.generate_sequence(max_length=20)
        self.assertGreater(len(sequence), 10)
        self.assertTrue(all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence))
    
    def test_property_prediction(self):
        generator = ProteinGenerator()
        sequence = "MVLSPADKTNVKAAWGKVGA"
        properties = generator.predict_properties(sequence)
        self.assertIn('stability', properties)
        self.assertIn('solubility', properties)

if __name__ == '__main__':
    unittest.main()