# Mol2D/optimizer.py (updated)
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
from pyscf import ci, cc, mp
from .molecule import Molecule

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class BasisOptimizer:
    def __init__(self, molecule, element, shell_types, n_primitives, 
                 beta=2.0, method='CCSD', opt_method='Nelder-Mead',initial_alphas=None):
        """
        element: str (e.g., 'Li')
        shell_types: list (e.g., ['s', 'p'])
        n_primitives: list of ints matching shell_types (e.g., [3, 2])
        beta: float or list (contraction factor for each shell)
        """
        self.original_mol = molecule
        self.element = element
        self.shell_types = shell_types
        self.n_primitives = n_primitives
        self.beta = beta if isinstance(beta, list) else [beta]*len(shell_types)
        self.method = method.upper()
        self.opt_method = opt_method
        
        # Get default alphas from molecule class
        self.initial_alphas = [
            molecule._get_default_exp(element, shell) 
            for shell in shell_types
        ]
        
        # History tracking
        self.alpha_history = []
        self.energy_history = []
       
        # Handle initial alphas
        if initial_alphas is not None:
            if len(initial_alphas) != len(shell_types):
                raise ValueError("initial_alphas length must match shell_types")
            
            # Replace None entries with defaults
            self.initial_alphas = [
                a if a is not None 
                else molecule._get_default_exp(element, shell)
                for a, shell in zip(initial_alphas, shell_types)
            ]
        else:
            # All defaults
            self.initial_alphas = [
                molecule._get_default_exp(element, shell)
                for shell in shell_types
            ] 
       
    def compute_energy(self, alphas):
        """alphas: list of exponents for each shell type"""
        try:
            # Generate basis string for all shells
            basis_parts = []
            for i, (shell, n_prim) in enumerate(zip(self.shell_types, self.n_primitives)):
                basis_parts.append(
                    f"{shell}{n_prim}({alphas[i]},{self.beta[i]})"
                )
            
            basis_str = f"{self.element}:" + ",".join(basis_parts)
            
            # Create new molecule with optimized basis
            new_mol = Molecule(
                self.original_mol.geom_str,
                basis_str,
                charge=self.original_mol.charge,
                mult=self.original_mol.mult
            )
            
            # Compute energy using specified method
            mf = new_mol.pyscf("rhf")
            mf.kernel()
            
            if self.method == 'HF':
                energy = mf.e_tot
            elif self.method == 'MP2':
                mp2 = mp.MP2(mf).run()
                energy = mp2.e_tot
            elif self.method == 'CISD':
                cisolver = ci.CISD(mf).run()
                energy = cisolver.e_tot
            elif self.method == 'CCSD':
                mycc = cc.CCSD(mf).run()
                energy = mycc.e_tot
            elif self.method == 'CCSD(T)':
                mycc = cc.CCSD(mf).run()
                energy = cc.ccsd_t(mycc).e_tot
            else:
                raise ValueError(f"Unsupported method: {self.method}")
            
            # Store results
            self.alpha_history.append(alphas)
            self.energy_history.append(energy if energy is not None else float('inf'))
            
            return energy if energy is not None else float('inf')
            
        except Exception as e:
            print(f"Error at alphas={alphas}: {str(e)}")
            self.alpha_history.append(alphas)
            self.energy_history.append(float('inf'))
            return float('inf')

    def optimize(self, initial_alphas=None, tol=1e-4, maxiter=100, plot=True):
        """Optimize multiple exponents simultaneously"""
        initial_alphas = initial_alphas or self.initial_alphas
        
        # Add bounds: alpha > 1e-9 to avoid numerical issues
        bounds = [(1e-6, None) for _ in initial_alphas]
        
        res = minimize(
            self.compute_energy,
            initial_alphas,
            method=self.opt_method,
            bounds=bounds,  # Add bounds parameter
            options={'maxiter': maxiter, 'xatol': tol}   #maxfev
        )
        
        if plot:
            self._plot_convergence()
            
        # Print optimization history
        print("\nOptimization History:")
        print(f"{'Iter':<6}{'Alphas':<30}{'Energy (Hartree)':<20}")
        for i, (alpha, energy) in enumerate(zip(self.alpha_history, self.energy_history)):
            alpha_str = np.array2string(np.round(alpha, 6), precision=6, suppress_small=True)
            energy_str = f"{energy:.8f}" if np.isfinite(energy) else "FAILED"
            print(f"{i+1:<6}{alpha_str:<30}{energy_str:<20}")
            
        return {
            'optimal_alphas': res.x,
            'optimal_energy': res.fun,
            'history': list(zip(self.alpha_history, self.energy_history)),
            'success': res.success
        }

    def _plot_convergence(self):
        """Plot convergence for multiple parameters"""
        if plt is None:
            return
            
        plt.figure(figsize=(10, 6))
        
        # Plot energy convergence
        plt.subplot(2, 1, 1)
        plt.plot(self.energy_history, 'o-')
        plt.ylabel('Energy (Hartree)')
        
        # Plot alpha convergence
        plt.subplot(2, 1, 2)
        alphas = np.array(self.alpha_history)
        for i, shell in enumerate(self.shell_types):
            plt.plot(alphas[:, i], 'o-', label=f'{shell} orbital')
        plt.xlabel('Iteration')
        plt.ylabel('Alpha values')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('basis_optimization.png')
        plt.close()
