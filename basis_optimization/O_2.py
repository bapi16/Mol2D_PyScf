#from molecule import Molecule
#import Mol2D as m2d
#from pyscf import cc
#from Mol2D import MethodAwareOptimizer

from Mol2D.molecule import Molecule
from Mol2D.optimizer import MethodAwareOptimizer

# Define molecular geometry
geom = '''
H 0.0 0.0

'''

# Initialize molecule with minimal basis
mol = Molecule(geom, "H: s1(0.5,2.0)", charge=0, mult=1)

# Set up optimizer
optimizer = MethodAwareOptimizer(
    molecule=mol,
    target_element='H',
    shells=['s', 'p'],
    initial_params={'s': [0.5, 3], 'p': [0.3, 2]},
    max_n={'s': 10, 'p': 5}
)

# Run optimization with CCSD(T)
result = optimizer.optimize(method='ccsd(t)', maxiter=20)

# Extract optimized parameters
optimized_params = result['parameters']
optimized_basis = result['basis']

print("Optimized parameters:")
for shell, (alpha0, n) in optimized_params.items():
    print(f"{shell}-shell: α₀ = {alpha0:.6f}, n = {n}")

print("Optimized basis set:", optimized_basis)

# Plot convergence
optimizer.plot_convergence()
