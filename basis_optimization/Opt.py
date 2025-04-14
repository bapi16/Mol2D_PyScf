# Mol2D/optimizer.py
from .molecule import Molecule  # Critical import
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pyscf import ci, cc, mp

class FixedBetaOptimizer:
    _method_registry = {
        'ccsd': (cc, 'CCSD'),
        'cisd': (ci, 'CISD'),
        'mp2': (mp, 'MP2'),
        'ccsd(t)': (cc, 'CCSD_T')
    }

    def __init__(self, geom, initial_params, target_element='H', 
                 beta=2.0, max_n=6, method='ccsd', cc_settings=None):
        self.geom = geom
        self.target_element = target_element
        self.beta = beta
        self.max_n = max_n
        self.method = method.lower()
        self.cc_settings = cc_settings or {}
        
        # Initialize with validated parameters
        self.initial_params = [
            initial_params[0], 
            max(1, min(int(round(initial_params[1])), self.max_n))
        ]
        self.params_history = [self.initial_params.copy()]
        self.energy_history = []
        self.iteration = 0
        self.best_energy = np.inf
        self.best_params = self.initial_params.copy()
        self._validate_method()

    def _validate_method(self):
        if self.method not in self._method_registry:
            raise ValueError(f"Unsupported method: {self.method}. "
                             f"Available: {list(self._method_registry.keys())}")

    def create_basis_string(self, params):
        """Generate basis set string with geometric sequence notation"""
        alpha0, n = params
        n_int = int(round(n))
        return f"{self.target_element}: s{n_int}({alpha0:.6f},{self.beta:.1f})"

    def _run_post_hf(self, pyscf_mol):
        """Execute post-HF calculation"""
        mod, cls_name = self._method_registry[self.method]
        calculator = getattr(mod, cls_name)(pyscf_mol)
        
        # Apply custom settings
        for key, value in self.cc_settings.items():
            setattr(calculator, key, value)
            
        calculator.kernel()
        return calculator.e_tot

    def objective(self, params):
        """Calculate energy for given parameters"""
        alpha0, n = params
        n_int = max(1, min(int(round(n)), self.max_n))
        
        try:
            basis_str = self.create_basis_string([alpha0, n_int])
            mol = Molecule(self.geom, basis_str, charge=0, mult=1)
            
            # Run HF
            hf_result = mol.scf(method='rhf', verbose=0)
            if not hf_result['converged']:
                return np.inf
                
            # Run post-HF
            pyscf_mol = mol.pyscf('rhf')
            energy = self._run_post_hf(pyscf_mol)
            
        except Exception as e:
            print(f"Calculation failed at α0={alpha0:.4f}, n={n_int}: {str(e)}")
            return np.inf

        # Track progress
        self.iteration += 1
        self.params_history.append([alpha0, n_int])
        self.energy_history.append(energy)
        
        # Update best parameters
        if energy < self.best_energy:
            self.best_energy = energy
            self.best_params = [alpha0, n_int]

        print(f"Iter {self.iteration:3d} | α0={alpha0:.4f} n={n_int:2d} | "
              f"{self.method.upper()} E={energy:.8f}")
        return energy

    def optimize(self, alpha_bounds=(0.1, 1.0), n_bounds=(2, 6), 
                maxiter=50, opt_method='Nelder-Mead'):
        """Run optimization"""
        bounds = [
            alpha_bounds,
            (n_bounds[0]-0.499, n_bounds[1]+0.499)
        ]
        
        result = minimize(
            self.objective,
            x0=self.initial_params,
            method=opt_method,
            bounds=bounds,
            options={'maxiter': maxiter}
        )
        
        if self.best_params is None:
            raise RuntimeError("Optimization failed to find valid parameters")
            
        return {
            'success': result.success,
            'alpha0': self.best_params[0],
            'n': self.best_params[1],
            'energy': self.best_energy
        }

    def plot_results(self):
        """Visualize optimization progress"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # Energy plot
        ax1.plot(self.energy_history, 'o-')
        ax1.set_title('Energy Convergence')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Energy (Hartree)')
        
        # Alpha0 plot
        ax2.plot([p[0] for p in self.params_history], 'o-')
        ax2.set_title('α₀ Optimization')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Initial Exponent (α₀)')
        
        # Basis size plot
        ax3.plot([p[1] for p in self.params_history], 'o-')
        ax3.set_title('Basis Size Optimization')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Number of Basis Functions')
        
        plt.tight_layout()
        plt.show()
