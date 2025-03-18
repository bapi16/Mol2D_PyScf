# Mol2D/molecule.py
import numpy as np
from multiprocessing import Pool, cpu_count, RawArray
from .atom import Atom
from .basis import create_basis

class Molecule:
    def __init__(self, geom, basis_str, charge=0, mult=1):
        self.atoms = []
        self.basis = []
        self.units = 'bohr'
        self.charge = charge
        self.mult = mult
        self.parse_geom(geom)
        self.parse_basis(basis_str)
        self.nuclear_repulsion = self.calculate_nuclear_repulsion()
        
    def parse_geom(self, geom):
        lines = [line.strip() for line in geom.split('\n') if line.strip()]
        for line in lines:
            if line.startswith('units'):
                self.units = line.split()[1].lower()
            else:
                parts = line.split()
                if len(parts) >= 3:
                    symbol, x, y = parts[:3]
                    pos = self.convert_units([float(x), float(y)])
                    self.atoms.append(Atom(symbol, pos))
    
    def convert_units(self, coords):
        if self.units == 'angstrom':
            return [x * 1.8897259886 for x in coords]
        return coords
    
    def parse_basis(self, basis_str):
        basis_specs = {}
        for spec in basis_str.split(';'):
            spec = spec.strip()
            if not spec:
               continue
            elem, basis = spec.split(':')
            elem = elem.strip()
            basis_specs[elem] = {}
            for part in basis.strip().split():
                n = int(''.join(filter(str.isdigit, part)))
                shell = ''.join(filter(str.isalpha, part))
                basis_specs[elem][shell] = n
         
        # Check all atoms have basis defined
        for atom in self.atoms:
            symbol = atom.symbol
            if symbol not in basis_specs:
               raise ValueError(f"Basis set not defined for element {symbol}")
            if not basis_specs[symbol]:
               raise ValueError(f"No basis functions specified for element {symbol}")
               
        for atom in self.atoms:
            for shell, n in basis_specs[atom.symbol].items():
                self.basis.extend(create_basis(
                     atom.position,
                     shell_type=shell,
                     n_primitives=n,
                     start_exp=self._get_default_exp(atom.symbol, shell)
                ))        
    
    def _get_default_exp(self, symbol, shell):
        defaults = {
            'H': {'s': 0.006, 'p': 0.0005},
            'He': {'s': 0.003, 'p': 0.0005},
            'Li': {'s': 0.0005, 'p': 0.0005},
            'Be': {'s': 0.0005, 'p': 0.0005},
            'B' : {'s': 0.0005, 'p': 0.0005},
            'N' : {'s': 0.0005, 'p': 0.0005},
            'F' : {'s': 0.0005, 'p': 0.0005},
            'Ne' : {'s': 0.0005, 'p': 0.0005},
            'Na' : {'s': 0.0005, 'p': 0.0005},
            'Mg' : {'s': 0.0005, 'p': 0.0005},
            'Al' : {'s': 0.0005, 'p': 0.0005},
            'P' : {'s': 0.0005, 'p': 0.0005},
            'Cl' : {'s': 0.0005, 'p': 0.0005},
            'Ar' : {'s': 0.0005, 'p': 0.0005}
            
        }
        if symbol not in defaults:
            raise ValueError(f"Element '{symbol}' not found in default exponent parameters")
        if shell not in defaults[symbol]:
            raise ValueError(f"Shell type '{shell}' for element '{symbol}' not found in default parameters")
        return defaults[symbol][shell]
    
    def calculate_nuclear_repulsion(self):
        E_nuc = 0.0
        n_atoms = len(self.atoms)
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                R = np.linalg.norm(self.atoms[i].position - self.atoms[j].position)
                E_nuc += self.atoms[i].Z * self.atoms[j].Z / R
        return E_nuc
    
    '''def scf(self, method='rhf',verbose=0):
        """Perform SCF calculation
        Args:
            method: 'rhf' (Restricted) or 'uhf' (Unrestricted)
        """
        from .scf import RHF, UHF
        method = method.lower()
        if method == 'rhf':
            if self.mult != 1:
                raise ValueError("RHF requires singlet state (mult=1)")
            return RHF(self).run()
        elif method == 'uhf':
            return UHF(self).run()
        raise ValueError(f"Unsupported method: {method}")'''
        
    def scf(self, method='rhf', verbose=0):
        """Perform SCF calculation
    
        Args:
            method (str): 'rhf' (Restricted) or 'uhf' (Unrestricted)
            verbose (int): Verbosity level (0-5)
        """
        from .scf import RHF, UHF
        method = method.lower()
    
        if method == 'rhf':
           if self.mult != 1:
              raise ValueError("RHF requires singlet state (mult=1)")
           return RHF(self, verbose=verbose).run()  # Pass verbose to RHF
        elif method == 'uhf':
             return UHF(self, verbose=verbose).run()  # Pass verbose to UHF
        raise ValueError(f"Unsupported method: {method}")
