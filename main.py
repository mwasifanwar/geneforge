import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.neural_networks.protein_transformer import ProteinGenerator
from src.evolutionary.genetic_algorithm import GeneticOptimizer
from src.evolutionary.fitness_functions import FitnessFunctions
from src.drug_discovery.docking_predictor import DockingModel
from src.visualization.structure_viz import StructureVisualizer
from src.api.server import app
import uvicorn

def run_demo():
    print("Running GeneForge Demo...")
    
    generator = ProteinGenerator()
    
    print("Generating novel protein sequences...")
    sequence = generator.generate_sequence(max_length=50)
    print(f"Generated sequence: {sequence}")
    
    properties = generator.predict_properties(sequence)
    print(f"Predicted properties: {properties}")
    
    structure = generator.predict_structure(sequence)
    print(f"Predicted structure with {len(structure)} coordinates")
    
    viz = StructureVisualizer()
    viz.plot_protein_structure(structure, sequence, "Generated Protein Structure")
    
    return sequence, properties, structure

def run_optimization():
    print("Running protein optimization...")
    
    fitness_funcs = FitnessFunctions()
    fitness_function = fitness_funcs.create_stability_fitness(target_stability=0.9)
    
    optimizer = GeneticOptimizer()
    best_sequence, best_fitness, history = optimizer.optimize(fitness_function, target_length=80)
    
    print(f"Optimized sequence: {best_sequence}")
    print(f"Best fitness: {best_fitness:.4f}")
    
    return best_sequence, best_fitness

def run_api():
    from src.utils.config import Config
    config = Config()
    print(f"Starting GeneForge API server on {config.get('api.host')}:{config.get('api.port')}")
    uvicorn.run(app, host=config.get('api.host'), port=config.get('api.port'))

def main():
    parser = argparse.ArgumentParser(description='GeneForge: AI-Powered Protein Design')
    parser.add_argument('--mode', choices=['api', 'demo', 'optimize', 'design'], default='demo', help='Operation mode')
    parser.add_argument('--sequence', type=str, help='Input protein sequence')
    parser.add_argument('--target', type=str, help='Target sequence for design')
    parser.add_argument('--length', type=int, default=100, help='Sequence length for generation')
    
    args = parser.parse_args()
    
    if args.mode == 'api':
        run_api()
    elif args.mode == 'demo':
        run_demo()
    elif args.mode == 'optimize':
        run_optimization()
    elif args.mode == 'design':
        if args.sequence:
            generator = ProteinGenerator()
            designed = generator.generate_sequence(prompt_sequence=args.sequence, max_length=args.length)
            print(f"Designed sequence: {designed}")
        else:
            print("Please provide a sequence with --sequence argument")
    else:
        print("GeneForge system ready - mwasifanwar")

if __name__ == "__main__":
    main()