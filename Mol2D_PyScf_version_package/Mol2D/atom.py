# Mol2D/atom.py
import numpy as np

class Atom:
    _atomic_data = {
        'H': {'Z': 1}, 'He': {'Z': 2}, 'Li': {'Z': 3}, 'Be': {'Z': 4},
        'B': {'Z': 5}, 'C': {'Z': 6}, 'N': {'Z': 7}, 'O': {'Z': 8},
        'F': {'Z': 9}, 'Ne': {'Z': 10}
    }
    
    def __init__(self, symbol, position, charge=0):
        self.symbol = symbol
        self.position = np.array(position)
        self.Z = self._get_atomic_number(charge)
        
    def _get_atomic_number(self, charge):
        base_z = self._atomic_data[self.symbol]['Z']
        return base_z - charge
