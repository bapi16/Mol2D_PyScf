import Mol2D as m2d

geom = '''
Li 0.0 0.0
units bohr
'''

# Create initial molecule
mol2d = m2d.Molecule(geom, "Li:6s,5p", charge=0, mult=2)

# Create optimizer for s and p orbitals
optimizer = m2d.BasisOptimizer(
    molecule=mol2d,
    element='Li',
    shell_types=['s', 'p'],
    n_primitives=[6, 5],  # 6 s-type primitives, 5 p-type
    beta=[2.0, 2.0],      # Same contraction factor for both
    method='CCSD'
)

# Get default alphas from molecule
initial_alphas = [
    mol2d._get_default_exp('Li', 's'),  # 0.0005 from defaults
    mol2d._get_default_exp('Li', 'p')   # 0.0005 from defaults
]

# Run optimization
result = optimizer.optimize(
    initial_alphas=initial_alphas,
    tol=1e-5,
    maxiter=50
)

print(f"Optimal alphas: {result['optimal_alphas']}")
print(f"Final CCSD energy: {result['optimal_energy']:.8f}")
