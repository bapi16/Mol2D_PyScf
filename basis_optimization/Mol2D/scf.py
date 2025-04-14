# Mol2D/scf.py
import numpy as np
import time
from scipy.linalg import eigh
from multiprocessing import Pool, cpu_count, RawArray
from .integrals import overlap, kinetic, nuclear_attraction, electron_repulsion

# Global variables for shared memory
_global_S_flat = None
_global_T_flat = None
_global_V_flat = None
_global_n = None
_global_basis = None
_global_atoms = None

_global_G_flat = None
_global_G_n = None
_global_G_basis = None


class RHF:
    def __init__(self, molecule, verbose=0):
        self.molecule = molecule
        self.verbose = verbose
        self.n_bf = len(molecule.basis)
        
        if self.verbose >= 1:
            print("\n=== Integral Computation ===")
            print(f"Number of basis functions: {self.n_bf}")
            print(f"Number of electrons: {sum(a.Z for a in molecule.atoms) - molecule.charge}")
            print(f"Nuclear repulsion energy: {molecule.nuclear_repulsion:.8f} Hartree") 
    
       
        # Time integral computations
        start_time = time.time()
        self.S, self.T, self.V = self._compute_STV_parallel()
        self.time_stv = time.time() - start_time
        
        start_time = time.time()
        self.G = self._compute_G_parallel()
        self.time_g = time.time() - start_time
        
        self.S, self.T, self.V = self._compute_STV_parallel()
        self.G = self._compute_G_parallel()
        self.H = self.T + self.V
        self.n_electrons = sum(a.Z for a in molecule.atoms) - molecule.charge
        self.n_occ = self.n_electrons // 2
        
        if self.verbose >= 1:
            print(f"\nIntegral computation times:")
            print(f"- STV integrals: {self.time_stv:.2f}s")
            print(f"- G integrals: {self.time_g:.2f}s")
        
    def _print_matrix(self, name, matrix, threshold=1e-4):
        if self.verbose >= 5:
           print(f"\n{name}:")
           print(matrix)
        elif self.verbose >= 4:
             print(f"\n{name} (non-zero elements):")
             for i in range(matrix.shape[0]):
                 for j in range(matrix.shape[1]):
                     if abs(matrix[i,j]) > threshold:
                        print(f"{i+1:3d}{j+1:3d} {matrix[i,j]:12.4e}")     
        
    def _compute_STV_parallel(self):
        global _global_S_flat, _global_T_flat, _global_V_flat, _global_n, _global_basis, _global_atoms
        n = self.n_bf
        basis = self.molecule.basis
        atoms = self.molecule.atoms

        _global_S_flat = RawArray('d', n * n)
        _global_T_flat = RawArray('d', n * n)
        _global_V_flat = RawArray('d', n * n)
        _global_n = n
        _global_basis = basis
        _global_atoms = atoms

        S = np.frombuffer(_global_S_flat, dtype=np.float64).reshape(n, n)
        T = np.frombuffer(_global_T_flat, dtype=np.float64).reshape(n, n)
        V = np.frombuffer(_global_V_flat, dtype=np.float64).reshape(n, n)
        S[:] = 0.0
        T[:] = 0.0
        V[:] = 0.0

        total = n * n
        chunk_size = 1000
        chunks = [
            (i, min(i + chunk_size, total))
            for i in range(0, total, chunk_size)
        ]

        with Pool(cpu_count()) as pool:
            pool.map(self.compute_STV_chunk, chunks)

        return S, T, V

    @staticmethod
    def compute_STV_chunk(args):
        start, end = args
        global _global_S_flat, _global_T_flat, _global_V_flat, _global_n, _global_basis, _global_atoms

        n = _global_n
        basis = _global_basis
        atoms = _global_atoms

        S_matrix = np.frombuffer(_global_S_flat, dtype=np.float64).reshape(n, n)
        T_matrix = np.frombuffer(_global_T_flat, dtype=np.float64).reshape(n, n)
        V_matrix = np.frombuffer(_global_V_flat, dtype=np.float64).reshape(n, n)

        for idx in range(start, end):
            i, j = divmod(idx, n)
            S_val = T_val = V_val = 0.0

            for p in range(len(basis[i].exps)):
                for q in range(len(basis[j].exps)):
                    a = basis[i].exps[p]
                    b = basis[j].exps[q]

                    coeff = (
                        basis[i].norm[p] * basis[j].norm[q] *
                        basis[i].coefs[p] * basis[j].coefs[q]
                    )

                    S_val += coeff * overlap(
                        a, basis[i].shell, basis[i].origin,
                        b, basis[j].shell, basis[j].origin
                    )
                    T_val += coeff * kinetic(
                        a, basis[i].shell, basis[i].origin,
                        b, basis[j].shell, basis[j].origin
                    )
                    for atom in atoms:
                        V_val -= atom.Z * coeff * nuclear_attraction(
                            a, basis[i].shell, basis[i].origin,
                            b, basis[j].shell, basis[j].origin,
                            atom.position
                        )

            S_matrix[i, j] = S_val
            T_matrix[i, j] = T_val
            V_matrix[i, j] = V_val

    def _compute_G_parallel(self):
        global _global_G_flat, _global_G_n, _global_G_basis
        n = self.n_bf
        basis = self.molecule.basis

        _global_G_flat = RawArray('d', n**4)
        _global_G_n = n
        _global_G_basis = basis

        G = np.frombuffer(_global_G_flat, dtype=np.float64).reshape(n, n, n, n)
        G[:] = 0.0

        total = n**4
        chunk_size = 1000
        chunks = [
            (i, min(i + chunk_size, total))
            for i in range(0, total, chunk_size)
        ]

        with Pool(cpu_count()) as pool:
            pool.map(self.compute_G_chunk, chunks)

        return G

    @staticmethod
    def compute_G_chunk(args):
        start, end = args
        global _global_G_flat, _global_G_n, _global_G_basis

        n = _global_G_n
        basis = _global_G_basis
        G_matrix = np.frombuffer(_global_G_flat, dtype=np.float64).reshape(n, n, n, n)

        for idx in range(start, end):
            i, j, k, l = np.unravel_index(idx, (n, n, n, n))
            integral = 0.0

            for p in range(len(basis[i].exps)):
                for q in range(len(basis[j].exps)):
                    for r in range(len(basis[k].exps)):
                        for s in range(len(basis[l].exps)):
                            coeff = (
                                basis[i].norm[p] * basis[j].norm[q] *
                                basis[k].norm[r] * basis[l].norm[s] *
                                basis[i].coefs[p] * basis[j].coefs[q] *
                                basis[k].coefs[r] * basis[l].coefs[s]
                            )

                            integral += coeff * electron_repulsion(
                                basis[i].exps[p], basis[i].shell, basis[i].origin,
                                basis[j].exps[q], basis[j].shell, basis[j].origin,
                                basis[k].exps[r], basis[k].shell, basis[k].origin,
                                basis[l].exps[s], basis[l].shell, basis[l].origin
                            )

            G_matrix[i, j, k, l] = integral

    def run(self, max_iter=100, tol=1e-8):
        D = np.zeros_like(self.S)
        scf_start = time.time()
        
        if self.verbose >= 1:
            print("\n=== SCF Iterations ===")
            print("Iter   Energy (Hartree)   ΔD         Time")
            
        
            
            
        for iter in range(max_iter):
            iter_start = time.time()
            J = np.einsum('ijkl,kl->ij', self.G, D)
            K = np.einsum('ikjl,kl->ij', self.G, D)
            F = self.H + J - 0.5*K
            
            if self.verbose >= 3:
               self._print_matrix("Fock Matrix", F)
               self._print_matrix("Density Matrix", D)
               
            _, C = eigh(F, self.S)
            D_new = 2 * C[:, :self.n_occ] @ C[:, :self.n_occ].T
            
            # Calculate metrics
            delta_d = np.max(np.abs(D_new - D))
            iter_time = time.time() - iter_start
            
            if self.verbose >= 1:
                energy = np.sum(D * (self.H + 0.5*J - 0.25*K)) + self.molecule.nuclear_repulsion
                print(f"{iter:3d}   {energy:16.8f}  {delta_d:.2e}  {iter_time:.2f}s")

            if np.max(np.abs(D_new - D)) < tol:
                break
            D = D_new
        time_scf = time.time() - scf_start  # End SCF timing
        

        E_elec = np.sum(D * (self.H + 0.5*J - 0.25*K))
        
        if self.verbose >= 1:
            print(f"\nFinal nuclear repulsion: {self.molecule.nuclear_repulsion:.8f} Hartree")
            print(f"Total energy breakdown:")
            print(f"- Electronic: {E_elec:.8f}")
            print(f"- Nuclear:    {self.molecule.nuclear_repulsion:.8f}")
            print(f"- Total:      {E_elec + self.molecule.nuclear_repulsion:.8f}")
            print(f"- Total_Time:{self.time_stv + self.time_g + time_scf:.2f}s")
        return {
            'energy': E_elec + self.molecule.nuclear_repulsion,
            #'nuclear_repulsion': self.molecule.nuclear_repulsion,
            'density': D,
            'converged': iter < max_iter-1,
            'time_integrals_stv': self.time_stv,
            'time_integrals_g': self.time_g,
            'time_scf': time_scf,
            'time_total': self.time_stv + self.time_g + time_scf
        }


class UHF(RHF):
    def __init__(self, molecule, verbose=0):
        super().__init__(molecule,verbose)
        self.n_alpha = (self.n_electrons + (molecule.mult - 1)) // 2
        self.n_beta = self.n_electrons - self.n_alpha
        
        if self.verbose >= 1:
            print(f"Unrestricted calculation with:")
            print(f"- Alpha electrons: {self.n_alpha}")
            print(f"- Beta electrons: {self.n_beta}")

    def run(self, max_iter=100, tol=1e-8):
        D_a = D_b = np.zeros_like(self.S)
        scf_start = time.time()  # Start SCF timing
        
        if self.verbose >= 1:
            print("\n=== UHF SCF Iterations ===")
            print("Iter   Energy (Hartree)   ΔDα        ΔDβ        Time")
            
        for iter in range(max_iter):
            iter_start = time.time()
            J = np.einsum('ijkl,kl->ij', self.G, D_a + D_b)
            K_a = np.einsum('ikjl,kl->ij', self.G, D_a)
            K_b = np.einsum('ikjl,kl->ij', self.G, D_b)

            F_a = self.H + J - K_a
            F_b = self.H + J - K_b
            
            if self.verbose >= 3:
                self._print_matrix("Alpha Fock Matrix", F_a)
                self._print_matrix("Beta Fock Matrix", F_b)
                self._print_matrix("Alpha Density", D_a)
                self._print_matrix("Beta Density", D_b)


            _, C_a = eigh(F_a, self.S)
            _, C_b = eigh(F_b, self.S)

            D_a_new = C_a[:, :self.n_alpha] @ C_a[:, :self.n_alpha].T
            D_b_new = C_b[:, :self.n_beta] @ C_b[:, :self.n_beta].T
            
            # Calculate metrics
            delta_a = np.max(np.abs(D_a_new - D_a))
            delta_b = np.max(np.abs(D_b_new - D_b))
            delta = max(delta_a, delta_b)
            iter_time = time.time() - iter_start
            
            if self.verbose >= 1:
                energy = (np.sum(D_a * self.H) + np.sum(D_b * self.H) + 
                        0.5*np.einsum('ij,ij', D_a+D_b, J) - 
                        0.5*(np.einsum('ij,ij', D_a, K_a) + np.einsum('ij,ij', D_b, K_b)) + 
                        self.molecule.nuclear_repulsion)
                print(f"{iter:3d}   {energy:16.8f}  {delta_a:.2e}  {delta_b:.2e}  {iter_time:.2f}s")

            delta = max(np.max(np.abs(D_a_new - D_a)),
                        np.max(np.abs(D_b_new - D_b)))
            if delta < tol:
                break

            D_a, D_b = D_a_new, D_b_new
        time_scf = time.time() - scf_start  # End SCF timing
        E_elec = (np.sum(D_a * self.H) + np.sum(D_b * self.H) +
                  0.5*np.einsum('ij,ij', D_a+D_b, J) -
                  0.5*(np.einsum('ij,ij', D_a, K_a) + np.einsum('ij,ij', D_b, K_b)))
        # Final output
        if self.verbose >= 1:
            print(f"\nFinal nuclear repulsion: {self.molecule.nuclear_repulsion:.8f} Hartree")
            print(f"Total energy breakdown:")
            print(f"- Electronic: {E_elec:.8f}")
            print(f"- Nuclear:    {self.molecule.nuclear_repulsion:.8f}")
            print(f"- Total:      {E_elec + self.molecule.nuclear_repulsion:.8f}")
            print(f"- Total_Time:{self.time_stv + self.time_g + time_scf:.2f}s")
        return {
            'energy': E_elec + self.molecule.nuclear_repulsion,
            'alpha_density': D_a,
            'beta_density': D_b,
            'nuclear_repulsion': self.molecule.nuclear_repulsion,
            'alpha_density': D_a,
            'beta_density': D_b,
            'spin_contamination': self._calculate_spin_contamination(D_a, D_b),
            'converged': iter < max_iter-1,
            'n_alpha': self.n_alpha,
            'n_beta': self.n_beta, 
            'time_integrals_stv': self.time_stv,
            'time_integrals_g': self.time_g,
            'time_scf': time_scf,
            'time_total': self.time_stv + self.time_g + time_scf
        }

    def _calculate_spin_contamination(self, D_a, D_b):
        S = np.einsum('ijkl,ji,lk->', self.G, D_a - D_b, D_a - D_b)
        return 0.5*(self.n_alpha - self.n_beta) * (0.5*(self.n_alpha - self.n_beta) + 1) - S
