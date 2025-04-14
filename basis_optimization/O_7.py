import Mol2D as m2d
from Mol2D.molecule import Molecule
from Mol2D.optimizer import BasisOptimizer
from Mol2D.molecule import Molecule
from pyscf import cc, mp, ci

geom = '''
Li 0.0 0.0
units bohr
'''


inial = 0.0005		
# Now works with parenthesized parameters
mol = m2d.Molecule(geom, f"Li:s10({inial},2.0)", charge=0, mult=2)
print(f"Created molecule with {len(mol.basis)} basis functions")

# Initialize optimizer for Li's s-orbital with 3 primitives
optimizer = BasisOptimizer(
    molecule=mol,
    element='Li',
    shell_type='s',
    n_primitives=10,
    beta=2.0,
    method='CCSD'
)

# Run optimization
result = optimizer.optimize(initial_alpha=inial)

print(f"Optimal alpha: {result['optimal_alpha']:.4f}")
print(f"Optimal energy: {result['optimal_energy']:.8f} Hartree")
print(f"Converged: {result['success']} ({result['message']})")
print(f"Evaluations: {result['n_iter']}")
