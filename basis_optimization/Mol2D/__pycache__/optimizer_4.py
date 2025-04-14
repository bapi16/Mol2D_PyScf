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
        self._cache = {}  # Cache for energy evaluations
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
        
        key = tuple(alphas)
        if key in self._cache:
            return self._cache[key]
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
            self.alpha_history.append(list(key))
            #self.alpha_history.append(alphas)
            self.energy_history.append(energy if energy is not None else float('inf'))
            self._cache[key] = energy if energy is not None else np.inf
            
            return self._cache[key]
            #return energy if energy is not None else float('inf')
            
        except Exception as e:
            print(f"Error at alphas={alphas}: {str(e)}")
            self.alpha_history.append(alphas)
            self.energy_history.append(float('inf'))
            self._cache[key] = np.inf
            return float('inf')

    def optimize(self, initial_alphas=None, tol=1e-4, maxiter=100, plot=True):
        """Optimize multiple exponents simultaneously"""
        initial_alphas = initial_alphas or self.initial_alphas
        
        # Variables for tracking convergence in callback
        prev_energy = None
        stop_flag = [False]  # Mutable to bypass nonlocal in Python 2.x
        
        def callback(xk):
            current_energy = self.compute_energy(xk)
            if prev_energy is not None:
                delta = abs(current_energy - prev_energy)
                print(f"  ΔE = {delta:.2e}", end='')
                if delta < tol:
                    print("\n*** Energy change below tolerance - stopping optimization")
                    stop_flag[0] = True
                    return True  # Stop minimization
            else:
                delta = np.inf
                print(f"  ΔE = ---", end='')
            
            # For Python 3, use nonlocal instead of this hack
            #nonlocal prev_energy
            #prev_energy = current_energy
            #return False
        
        print("\nStarting optimization:")
        print(f"{'Iter':<6}{'Energy (Hartree)':<20}{'ΔE':<12}{'Alphas'}")
        
        res = minimize(
            self.compute_energy,
            initial_alphas,
            method=self.opt_method,
            callback=callback,
            options={'maxiter': maxiter, 'xatol': tol}
        )
        
        # Handle early stopping
        if stop_flag[0]:
            res.nit = len(self.energy_history)  # Approximate iteration count
        
        
        if plot:
            self._plot_convergence()
            
        # Print optimization history
        print("\nOptimization History:")
        print(f"{'Iter':<6}{'Energy (Hartree)':<20}{'ΔE':<12}{'Alphas'}")
        prev_e = None
        
        #print(f"{'Iter':<6}{'Alphas':<30}{'Energy (Hartree)':<20}")
        
        for i, (a, e) in enumerate(zip(self.alpha_history, self.energy_history)):
            # Handle energy formatting
            energy_str = f"{e:.8f}" if np.isfinite(e) else "FAILED"
        
            # Handle delta calculation
            delta = abs(e - prev_e) if prev_e is not None else np.inf
            delta_str = f"{delta:.2e}" if prev_e is not None else "---"
        
            # Format alphas
            a_str = np.array2string(np.array(a), precision=4, suppress_small=True)
        
            print(f"{i+1:<6}{energy_str:<20}{delta_str:<12}{a_str}")
            prev_e = e
            
        return {
            'optimal_alphas': res.x,
            'optimal_energy': res.fun,
            'history': list(zip(self.alpha_history, self.energy_history)),
            'success': res.success or stop_flag[0]
        }

    def _plot_convergence(self):
        """Plot convergence for multiple parameters"""
        if plt is None or len(self.energy_history) < 2:
            return
            
        plt.figure(figsize=(10, 6))
        
        # Plot energy convergence
        plt.subplot(2, 1, 1)
        valid_energies = [e for e in self.energy_history if np.isfinite(e)]
        #plt.plot(self.energy_history, 'o-')
        plt.plot(valid_energies, 'o-')
        plt.ylabel('Energy (Hartree)')
        plt.title(f'{self.method} Basis Optimization Convergence')
        
        # Plot alpha convergence
        plt.subplot(2, 1, 2)
        alphas = np.array([a for a, e in zip(self.alpha_history, self.energy_history) if np.isfinite(e)])
        #alphas = np.array(self.alpha_history)
        for i in range(alphas.shape[1]):
            plt.plot(alphas[:,i], 'o-', label=f'{self.shell_types[i]} orbital')
        plt.xlabel('Evaluation')
        plt.ylabel('Exponent (alpha)')
        plt.legend()
        
        
        plt.tight_layout()
        plt.savefig('basis_optimization.png')
        plt.close()
