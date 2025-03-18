# Mol2D/basis.py
import numpy as np
from .integrals import overlap

class BasisFunction:
    def __init__(self, origin, shell, exps, coefs):
        self.origin = np.array(origin)
        self.shell = np.array(shell)
        self.exps = np.array(exps)
        self.coefs = np.array(coefs)
        self.norm = np.zeros_like(coefs)
        self.normalize()
    
    def normalize(self):
        # Normalize primitive Gaussians
        for i in range(len(self.exps)):
            S = overlap(self.exps[i], self.shell, self.origin,
                        self.exps[i], self.shell, self.origin)
            self.norm[i] = 1 / np.sqrt(S)
        
        # Normalize contracted combination
        N = 0.0
        for i in range(len(self.exps)):
            for j in range(len(self.exps)):
                S = overlap(self.exps[i], self.shell, self.origin,
                            self.exps[j], self.shell, self.origin)
                N += self.norm[i] * self.norm[j] * self.coefs[i] * self.coefs[j] * S
        self.coefs /= np.sqrt(N)

'''def create_basis(origin, shell_type, n_primitives, start_exp=0.1, factor=2.0):
    shell_map = {
        's': [[0, 0]],
        'p': [[1, 0], [0, 1]],
        'd': [[2, 0], [1, 1], [0, 2]],
        'f': [[3, 0], [2, 1], [1, 2], [0, 3]]
    }
    exponents = [start_exp * (factor**i) for i in range(n_primitives)]
    basis = []
    for exp in exponents:
        for angular in shell_map[shell_type]:
            basis.append(BasisFunction(origin, angular, [exp], [1.0]))
    return basis'''
    
def create_basis(origin, shell_type, n_primitives, start_exp, factor=2.0):
    shell_map = {
        's': [[0, 0]],          # 1 component
        'p': [[1, 0], [0, 1]],  # 2 components
        'd': [[2, 0], [1, 1], [0, 2]],          # 3 components
        'f': [[3, 0], [2, 1], [1, 2], [0, 3]]   # 4 components
    }
    basis = []
    
    # Generate exponents first
    exponents = [start_exp * (factor ** i) for i in range(n_primitives)]
    
    # Create basis functions: all angular components for each exponent
    for exp in exponents:
        for angular in shell_map[shell_type]:
            basis.append(
                BasisFunction(
                    origin,angular,[exp],[1.0]))
            
    
    
    return basis

