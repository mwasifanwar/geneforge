import numpy as np
from typing import Dict, List

class MDSimulator:
    def __init__(self):
        self.config = Config()
        self.time_step = self.config.get('molecular_dynamics.time_step')
        self.temperature = self.config.get('molecular_dynamics.temperature')
    
    def simulate(self, coordinates, sequence, simulation_time=None):
        if simulation_time is None:
            simulation_time = self.config.get('molecular_dynamics.simulation_time')
        
        n_steps = int(simulation_time / self.time_step)
        n_particles = len(coordinates)
        
        trajectory = [coordinates.copy()]
        velocities = np.random.normal(0, np.sqrt(self.temperature), coordinates.shape)
        
        for step in range(n_steps):
            forces = self._calculate_forces(coordinates, sequence)
            
            velocities += forces * self.time_step
            coordinates += velocities * self.time_step
            
            if step % 100 == 0:
                velocities = self._thermostat(velocities)
                trajectory.append(coordinates.copy())
            
            if step % 1000 == 0:
                energy = self._calculate_energy(coordinates, velocities, sequence)
                print(f"mwasifanwar MD Step {step}, Energy: {energy:.4f}")
        
        return np.array(trajectory)
    
    def _calculate_forces(self, coordinates, sequence):
        n_particles = len(coordinates)
        forces = np.zeros_like(coordinates)
        
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                r_vec = coordinates[j] - coordinates[i]
                r = np.linalg.norm(r_vec)
                
                if r > 0:
                    force_magnitude = self._lennard_jones_force(r)
                    force_dir = r_vec / r
                    forces[i] += force_magnitude * force_dir
                    forces[j] -= force_magnitude * force_dir
        
        return forces
    
    def _lennard_jones_force(self, r, epsilon=1.0, sigma=3.8):
        if r == 0:
            return 0
        return 24 * epsilon * (2 * (sigma/r)**13 - (sigma/r)**7) / r
    
    def _thermostat(self, velocities, target_temperature=None):
        if target_temperature is None:
            target_temperature = self.temperature
        
        current_temperature = np.mean(np.sum(velocities**2, axis=1)) / 3
        scale_factor = np.sqrt(target_temperature / current_temperature)
        
        return velocities * scale_factor
    
    def _calculate_energy(self, coordinates, velocities, sequence):
        kinetic_energy = 0.5 * np.sum(velocities**2)
        potential_energy = self._calculate_potential_energy(coordinates, sequence)
        return kinetic_energy + potential_energy
    
    def _calculate_potential_energy(self, coordinates, sequence):
        energy = 0.0
        n_particles = len(coordinates)
        
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                r = np.linalg.norm(coordinates[j] - coordinates[i])
                energy += self._lennard_jones_potential(r)
        
        return energy
    
    def _lennard_jones_potential(self, r, epsilon=1.0, sigma=3.8):
        if r == 0:
            return float('inf')
        return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)