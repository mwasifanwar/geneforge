import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class StructureVisualizer:
    def __init__(self):
        self.config = Config()
    
    def plot_protein_structure(self, coordinates, sequence=None, title="Protein Structure"):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if len(coordinates) == 0:
            return
        
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        z = coordinates[:, 2]
        
        ax.plot(x, y, z, 'b-', linewidth=2, alpha=0.8)
        ax.scatter(x, y, z, c=range(len(coordinates)), cmap='viridis', s=50)
        
        if sequence and len(sequence) == len(coordinates):
            for i, (xi, yi, zi, aa) in enumerate(zip(x, y, z, sequence)):
                ax.text(xi, yi, zi, aa, fontsize=8)
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(title)
        
        plt.show()
    
    def plot_interactive_structure(self, coordinates, sequence=None, title="Protein Structure"):
        if len(coordinates) == 0:
            return
        
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        z = coordinates[:, 2]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            line=dict(width=6, color='blue'),
            marker=dict(size=4, color=list(range(len(coordinates))),
                       colorscale='Viridis'),
            name='Protein Backbone'
        ))
        
        if sequence and len(sequence) == len(coordinates):
            annotations = []
            for i, (xi, yi, zi, aa) in enumerate(zip(x, y, z, sequence)):
                annotations.append(dict(
                    x=xi, y=yi, z=zi,
                    text=aa,
                    showarrow=False,
                    font=dict(color='black', size=10),
                    bgcolor='white',
                    opacity=0.8
                ))
            fig.update_layout(scene_annotations=annotations)
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)'
            )
        )
        
        fig.show()
    
    def plot_contact_map(self, contact_map, title="Contact Map"):
        plt.figure(figsize=(8, 6))
        plt.imshow(contact_map, cmap='viridis', origin='lower')
        plt.colorbar(label='Contact Probability')
        plt.title(title)
        plt.xlabel('Residue Index')
        plt.ylabel('Residue Index')
        plt.show()
    
    def plot_secondary_structure(self, secondary_structure, sequence=None, title="Secondary Structure"):
        ss_map = {'H': 2, 'E': 1, 'C': 0}
        if isinstance(secondary_structure[0], str):
            ss_numeric = [ss_map.get(ss, 0) for ss in secondary_structure]
        else:
            ss_numeric = secondary_structure
        
        plt.figure(figsize=(12, 4))
        plt.plot(ss_numeric, 'r-', linewidth=2)
        plt.fill_between(range(len(ss_numeric)), ss_numeric, alpha=0.3)
        
        if sequence:
            for i, aa in enumerate(sequence):
                plt.text(i, -0.5, aa, ha='center', va='center', fontsize=8)
        
        plt.yticks([0, 1, 2], ['Coil', 'Sheet', 'Helix'])
        plt.xlabel('Residue Index')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()