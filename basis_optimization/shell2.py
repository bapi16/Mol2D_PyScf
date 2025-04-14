import Mol2D as m2d

geom = '''
Li 0.0 0.0
units bohr
'''

# Create initial molecule
mol2d = m2d.Molecule(geom, "Li:6s,2p", charge=0, mult=2)

# Create optimizer with CUSTOM initial alphas
optimizer = m2d.BasisOptimizer(
    molecule=mol2d,
    element='Li',
    shell_types=['s', 'p'],
    n_primitives=[6,2],
    beta=[2.0, 2.0],
    method='CCSD',
    initial_alphas=[0.005, 0.008]  # Your custom values here
)

# Run optimization (no need to specify initial_alphas again)
result = optimizer.optimize(tol=1e-5, maxiter=50)

print(f"Optimal alphas: {result['optimal_alphas']}")
print(f"Final CCSD energy: {result['optimal_energy']:.8f}")
