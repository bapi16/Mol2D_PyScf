import Mol2D as m2d
from Mol2D.optimizer import BasisOptimizer

geom = '''
Li 0.0 0.0
units bohr
'''

initial_alpha = 0.0003
mol = m2d.Molecule(geom, f"Li:s10({initial_alpha},2.0)", charge=0, mult=2)

optimizer = BasisOptimizer(
    molecule=mol,
    element='Li',
    shell_type='s',
    n_primitives=20,
    beta=2.0,
    method='CCSD'
)

# Run with expanded bracket handling
result = optimizer.optimize(
    initial_alpha=initial_alpha,
    bracket_factor=5,  # More conservative expansion
    maxiter=50
)

print(f"Optimal alpha: {result['optimal_alpha']:.6f}")
print(f"Final energy: {result['optimal_energy']:.8f} Hartree")
