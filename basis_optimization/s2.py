import Mol2D as m2d

geom = '''
Li 0.0 0.0
units bohr
'''

# Create initial molecule
mol2d = m2d.Molecule(geom, "Li:6s", charge=0, mult=2)

# Create optimizer with CUSTOM initial alphas
optimizer = m2d.BasisOptimizer(
    molecule=mol2d,
    element='Li',
    shell_types=['s'],
    n_primitives=[6],
    beta=[2.0],
    method='CISD',
    initial_alphas=[0.001]  # Your custom values here
)

# Run optimization (no need to specify initial_alphas again)
result = optimizer.optimize(tol=1e-8, maxiter=200)

print(f"Optimal alphas: {result['optimal_alphas']}")
print(f"Final CISD energy: {result['optimal_energy']:.8f}")
