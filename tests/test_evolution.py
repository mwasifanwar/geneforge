import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.evolutionary.genetic_algorithm import GeneticOptimizer
from src.evolutionary.fitness_functions import FitnessFunctions

class TestEvolution(unittest.TestCase):
    def test_genetic_optimizer(self):
        fitness_funcs = FitnessFunctions()
        fitness_function = fitness_funcs.create_stability_fitness()
        
        optimizer = GeneticOptimizer()
        best_sequence, best_fitness, history = optimizer.optimize(
            fitness_function, target_length=30, generations=5
        )
        
        self.assertGreater(len(best_sequence), 10)
        self.assertIsInstance(best_fitness, float)
    
    def test_fitness_functions(self):
        fitness_funcs = FitnessFunctions()
        
        stability_fitness = fitness_funcs.create_stability_fitness()
        score = stability_fitness("MVLSPADKTNVKAAWGKVGA")
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        
        drug_fitness = fitness_funcs.create_drug_likeness_fitness()
        score = drug_fitness("MVLSPADKTNVKAAWGKVGA")
        self.assertGreaterEqual(score, -1)
        self.assertLessEqual(score, 1)

if __name__ == '__main__':
    unittest.main()