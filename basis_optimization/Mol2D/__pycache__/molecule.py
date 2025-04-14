# Mol2D/molecule.py
import numpy as np
import re
from multiprocessing import Pool, cpu_count, RawArray
from .atom import Atom
from .basis import BasisFunction
from pyscf import gto, scf, ao2mo
from .scf import RHF, UHF

class Molecule:
    def __init__(self, geom, basis_str, charge=0, mult=1):
        self.atoms = []
        self.basis = []
        self.units = 'bohr'
        self.charge = charge
        self.mult = mult
        self.parse_geom(geom)
        self.parse_basis(basis_str)
        self.n_bf = len(self.basis) 
        self.nuclear_repulsion = self.calculate_nuclear_repulsion()
        
        if self.n_bf == 0:
           raise ValueError("No basis functions! Check basis set definitions.")
        
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

        try:
            elem_part, basis_part = spec.split(':', 1)
        except ValueError:
            raise ValueError(f"Invalid basis specification: '{spec}' (missing colon)")

        elem = elem_part.strip()
        basis_specs[elem] = []

        # Process contracted basis specifications with explicit coefficients
        contracted_matches = re.findall(r'(\w+)\s+(\[.*?\])\s+(\[.*?\])', basis_part)
        for match in contracted_matches:
            shell_type, exps_str, coefs_str = match
            try:
                exps = list(map(float, exps_str.strip('[]').split(',')))
                coefs = list(map(float, coefs_str.strip('[]').split(',')))
            except ValueError as e:
                raise ValueError(f"Invalid number format in '{match}': {e}")

            if len(exps) != len(coefs):
                raise ValueError(
                    f"Mismatched exponents/coefficients in '{match}'\n"
                    f"Found {len(exps)} exponents but {len(coefs)} coefficients"
                )
            basis_specs[elem].append((shell_type, exps, coefs))

        # Process uncontracted basis specifications (number-prefixed notation)
        remaining = re.sub(r'\w+\s+\[.*?\]\s+\[.*?\]', '', basis_part)
        for shell_spec in remaining.split(','):
            shell_spec = shell_spec.strip()
            if not shell_spec:
                continue

            # Normalize by removing all whitespace
            normalized_spec = re.sub(r'\s+', '', shell_spec)
            
            # Match parameterized basis (e.g., 5s(0.003,2.0))
            param_match = re.match(
                r'^(\d*)([spdf])(\d*)\(([\d.]+),([\d.]+)\)$', 
                normalized_spec
            )
            if param_match:
                prefix_num, shell_type, suffix_num, start_exp_str, factor_str = param_match.groups()
                start_exp = float(start_exp_str)
                factor = float(factor_str)
                
                # Determine number of primitives
                if prefix_num and suffix_num:
                    raise ValueError(f"Ambiguous basis specification: '{shell_spec}'")
                n_primitives = int(prefix_num or suffix_num or 1)
                
                # Generate exponents
                for i in range(n_primitives):
                    exp = start_exp * (factor ** i)
                    basis_specs[elem].append((shell_type, [exp], [1.0]))
                continue

            # Match simple basis (e.g., 5s, s5, 3p)
            simple_match = re.match(r'^(\d*)([spdf])(\d*)$', normalized_spec)
            if simple_match:
                prefix_num, shell_type, suffix_num = simple_match.groups()
                
                if prefix_num and suffix_num:
                    raise ValueError(f"Ambiguous basis specification: '{shell_spec}'")
                
                n_primitives = int(prefix_num or suffix_num or 1)
                if n_primitives <= 0:
                    raise ValueError(f"Invalid number of primitives: {n_primitives}")

                # Get default exponent for element and shell
                try:
                    start_exp = self._get_default_exp(elem, shell_type)
                except KeyError:
                    start_exp = 0.2  # Fallback default

                factor = 2.0
                # Generate exponents
                for i in range(n_primitives):
                    exp = start_exp * (factor ** i)
                    basis_specs[elem].append((shell_type, [exp], [1.0]))
            else:
                raise ValueError(f"Invalid basis specification: '{shell_spec}'")

    # Create BasisFunction objects
    shell_map = {
        's': [[0, 0]],
        'p': [[1, 0], [0, 1]],
        'd': [[2, 0], [1, 1], [0, 2]],
        'f': [[3, 0], [2, 1], [1, 2], [0, 3]]
    }

    for atom in self.atoms:
        elem = atom.symbol
        if elem not in basis_specs:
            raise ValueError(f"No basis set specified for element {elem}")

        for shell_type, exps, coefs in basis_specs[elem]:
            if shell_type not in shell_map:
                raise ValueError(
                    f"Unsupported shell type '{shell_type}' for {elem}. "
                    f"Valid types: {list(shell_map.keys())}"
                )

            for angular in shell_map[shell_type]:
                self.basis.append(BasisFunction(
                    origin=atom.position,
                    shell=angular,
                    exps=exps,
                    coefs=coefs
                ))
    def _get_default_exp(self, symbol, shell):
        # Default exponents for common elements/shells
        defaults = {
            'H': {'s': 0.006},
            'He': {'s': 0.003},
            'Li': {'s': 0.0005, 'p': 0.0005},
            'Be': {'s': 0.0005, 'p': 0.0005},
            'B': {'s': 0.0005, 'p': 0.0005},
            'C': {'s': 0.0005, 'p': 0.0005},
            'N': {'s': 0.0005, 'p': 0.0005},
            'O': {'s': 0.0005, 'p': 0.0005},
            'F': {'s': 0.0005, 'p': 0.0005},
            'Ne': {'s': 0.0005, 'p': 0.0005}
        }
        return defaults.get(symbol, {}).get(shell, 0.2)
    
    def calculate_nuclear_repulsion(self):
        E_nuc = 0.0
        n_atoms = len(self.atoms)
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                R = np.linalg.norm(self.atoms[i].position - self.atoms[j].position)
                E_nuc += self.atoms[i].Z * self.atoms[j].Z / R
        return E_nuc
    
   
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
        
    
    # In Mol2D/molecule.py (updated pyscf method)
    def pyscf(self, method='rhf'):
        from pyscf import gto, scf, ao2mo
        import numpy as np

        # --- Integral Computation ---
        method = method.lower()
        if method == 'rhf':
           scf_instance = RHF(self, verbose=0)
        elif method == 'uhf':
           scf_instance = UHF(self, verbose=0)
        else:
           raise ValueError(f"Unsupported method: {method}")

        H = scf_instance.H
        S = scf_instance.S
        G = scf_instance.G

        # --- Basis Validation ---
        if self.n_bf == 0:
           raise RuntimeError("No basis functions generated. Check basis set syntax.")
        print(f"\n[DEBUG] Basis functions: {self.n_bf}")

        # --- PySCF Molecule Setup ---
        mol = gto.M()
        mol.nelectron = sum(a.Z for a in self.atoms) - self.charge
        mol.charge = self.charge
        mol.spin = self.mult - 1

        # --- Electron Validation ---
        print(f"[DEBUG] Electrons: {mol.nelectron}")
        if mol.nelectron <= 0:
           raise ValueError(f"Invalid electron count: {mol.nelectron}")

        # --- ERI Conversion ---
        eri_pyscf = ao2mo.restore(1, G, S.shape[0])

        # --- SCF Object Configuration ---
        mf = scf.RHF(mol) if method == 'rhf' else scf.UHF(mol)
        mf.get_hcore = lambda *args: H
        mf.get_ovlp = lambda *args: S
        mf._eri = eri_pyscf

        # --- Debug: Print Critical Matrices ---
        print("[DEBUG] H diagonal (eV):", np.diag(H)[:5] * 27.2114)
        print("[DEBUG] S diagonal:", np.diag(S)[:5])

        # --- MO Initialization with Fallback ---
        try:
           mf.kernel()  # Try standard SCF
        except Exception as e:
           print(f"[WARNING] SCF failed: {str(e)}")
           print("Attempting manual MO initialization...")
        
           # Canonical orthogonalization
           s_eval, s_evec = np.linalg.eigh(S)
           keep = s_eval > 1e-6
           X = s_evec[:, keep] / np.sqrt(s_eval[keep])
           H_ortho = X.T @ H @ X
           mo_energy, mo_coeff_ortho = np.linalg.eigh(H_ortho)
           mo_coeff = X @ mo_coeff_ortho
        
           # Sort MOs by energy
           idx = np.argsort(mo_energy)
           mf.mo_energy = mo_energy[idx]
           mf.mo_coeff = mo_coeff[:, idx]
           mf.mo_occ = np.zeros_like(mf.mo_energy)
           mf.mo_occ[:mol.nelectron//2] = 2  # RHF

        # --- Final Occupation Check ---
        nocc = np.sum(mf.mo_occ > 0)
        print(f"[DEBUG] Occupied orbitals: {nocc}")
        if nocc == 0 and mol.nelectron > 0:
           print("[EMERGENCY] Forcing lowest MO occupation")
           mf.mo_occ[:mol.nelectron//2] = 2
           nocc = mol.nelectron//2

        if nocc == 0:
           raise RuntimeError("Electrons exist but no orbitals occupied. Check H matrix.")

        return mf
