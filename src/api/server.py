from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

app = FastAPI(title="GeneForge API", version="1.0.0")

class ProteinDesignRequest(BaseModel):
    target_sequence: Optional[str] = ""
    target_structure: Optional[List[List[float]]] = None
    design_objective: str = "stability"
    sequence_length: int = 100
    num_designs: int = 5

class OptimizationRequest(BaseModel):
    initial_sequence: str
    fitness_function: str
    generations: int = 100
    population_size: int = 50

class PropertyPredictionRequest(BaseModel):
    sequence: str
    properties: List[str] = ["stability", "solubility", "toxicity"]

class DockingRequest(BaseModel):
    protein_sequence: str
    ligand_smiles: str

@app.post("/design_protein")
async def design_protein(request: ProteinDesignRequest):
    try:
        from src.neural_networks.protein_transformer import ProteinGenerator
        from src.evolutionary.genetic_algorithm import GeneticOptimizer
        from src.evolutionary.fitness_functions import FitnessFunctions
        
        generator = ProteinGenerator()
        
        if request.target_sequence:
            designed_sequences = []
            for _ in range(request.num_designs):
                sequence = generator.generate_sequence(
                    prompt_sequence=request.target_sequence[:20],
                    max_length=request.sequence_length
                )
                designed_sequences.append(sequence)
        else:
            fitness_funcs = FitnessFunctions()
            
            if request.design_objective == "stability":
                fitness_function = fitness_funcs.create_stability_fitness()
            elif request.design_objective == "binding":
                fitness_function = fitness_funcs.create_binding_fitness("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK")
            else:
                fitness_function = fitness_funcs.create_multi_objective_fitness()
            
            optimizer = GeneticOptimizer()
            best_sequence, best_fitness, history = optimizer.optimize(
                fitness_function, 
                target_length=request.sequence_length
            )
            designed_sequences = [best_sequence]
        
        return {
            "status": "success",
            "designed_sequences": designed_sequences,
            "design_objective": request.design_objective,
            "num_designs": len(designed_sequences)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize_protein")
async def optimize_protein(request: OptimizationRequest):
    try:
        from src.evolutionary.genetic_algorithm import GeneticOptimizer
        from src.evolutionary.fitness_functions import FitnessFunctions
        
        fitness_funcs = FitnessFunctions()
        
        if request.fitness_function == "stability":
            fitness_function = fitness_funcs.create_stability_fitness()
        elif request.fitness_function == "drug_likeness":
            fitness_function = fitness_funcs.create_drug_likeness_fitness()
        else:
            fitness_function = fitness_funcs.create_multi_objective_fitness()
        
        optimizer = GeneticOptimizer()
        best_sequence, best_fitness, history = optimizer.optimize(
            fitness_function,
            initial_sequence=request.initial_sequence,
            target_length=len(request.initial_sequence)
        )
        
        return {
            "status": "success",
            "optimized_sequence": best_sequence,
            "best_fitness": best_fitness,
            "generations": request.generations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_properties")
async def predict_properties(request: PropertyPredictionRequest):
    try:
        from src.neural_networks.property_predictor import PropertyModel
        
        predictor = PropertyModel()
        properties = predictor.predict_properties(request.sequence)
        
        filtered_properties = {
            prop: value for prop, value in properties.items() 
            if prop in request.properties
        }
        
        return {
            "status": "success",
            "sequence": request.sequence,
            "properties": filtered_properties
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_docking")
async def predict_docking(request: DockingRequest):
    try:
        from src.drug_discovery.docking_predictor import DockingModel
        
        docking_model = DockingModel()
        results = docking_model.predict_docking(request.protein_sequence, request.ligand_smiles)
        
        return {
            "status": "success",
            "docking_results": results,
            "protein_length": len(request.protein_sequence)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_structure")
async def predict_structure(sequence: str):
    try:
        from src.neural_networks.protein_transformer import ProteinGenerator
        
        generator = ProteinGenerator()
        structure = generator.predict_structure(sequence)
        
        return {
            "status": "success",
            "sequence": sequence,
            "structure": structure.tolist(),
            "sequence_length": len(sequence)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "GeneForge"}

if __name__ == "__main__":
    import uvicorn
    from src.utils.config import Config
    config = Config()
    uvicorn.run(app, host=config.get('api.host'), port=config.get('api.port'))