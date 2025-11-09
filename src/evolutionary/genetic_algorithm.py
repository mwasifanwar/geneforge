import numpy as np
import random
from typing import List, Dict, Tuple

class GeneticOptimizer:
    def __init__(self):
        self.config = Config()
        
        self.population_size = self.config.get('evolutionary.population_size')
        self.generations = self.config.get('evolutionary.generations')
        self.mutation_rate = self.config.get('evolutionary.mutation_rate')
        self.crossover_rate = self.config.get('evolutionary.crossover_rate')
        self.elite_size = self.config.get('evolutionary.elite_size')
    
    def optimize(self, fitness_function, initial_sequence=None, target_length=100):
        population = self._initialize_population(target_length, initial_sequence)
        
        best_individual = None
        best_fitness = -float('inf')
        fitness_history = []
        
        for generation in range(self.generations):
            fitness_scores = self._evaluate_population(population, fitness_function)
            
            new_population = []
            
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx]
            
            fitness_history.append(best_fitness)
            
            if generation % 10 == 0:
                print(f"mwasifanwar Generation {generation}, Best Fitness: {best_fitness:.4f}")
        
        return best_individual, best_fitness, fitness_history
    
    def _initialize_population(self, target_length, initial_sequence=None):
        population = []
        
        if initial_sequence:
            population.append(initial_sequence)
        
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        for _ in range(self.population_size - len(population)):
            if initial_sequence and random.random() < 0.3:
                sequence = self._mutate_sequence(initial_sequence)
            else:
                sequence = ''.join(random.choices(amino_acids, k=target_length))
            population.append(sequence)
        
        return population
    
    def _evaluate_population(self, population, fitness_function):
        return np.array([fitness_function(seq) for seq in population])
    
    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        selected = random.sample(range(len(population)), tournament_size)
        best_idx = selected[np.argmax(fitness_scores[selected])]
        return population[best_idx]
    
    def _crossover(self, parent1, parent2):
        min_len = min(len(parent1), len(parent2))
        crossover_point = random.randint(1, min_len - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def _mutate(self, sequence):
        if random.random() < self.mutation_rate:
            return self._mutate_sequence(sequence)
        return sequence
    
    def _mutate_sequence(self, sequence):
        mutation_type = random.choice(['substitute', 'insert', 'delete'])
        
        if mutation_type == 'substitute' and len(sequence) > 0:
            pos = random.randint(0, len(sequence) - 1)
            new_aa = random.choice('ACDEFGHIKLMNPQRSTVWY')
            return sequence[:pos] + new_aa + sequence[pos+1:]
        
        elif mutation_type == 'insert' and len(sequence) < self.config.get('data.max_sequence_length'):
            pos = random.randint(0, len(sequence))
            new_aa = random.choice('ACDEFGHIKLMNPQRSTVWY')
            return sequence[:pos] + new_aa + sequence[pos:]
        
        elif mutation_type == 'delete' and len(sequence) > 10:
            pos = random.randint(0, len(sequence) - 1)
            return sequence[:pos] + sequence[pos+1:]
        
        return sequence