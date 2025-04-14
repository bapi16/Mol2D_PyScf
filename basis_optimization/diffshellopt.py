import Mol2D as m2d
from Mol2D.optimizer import BasisOptimizer

geom = '''
Li 0.0 0.0
units bohr
'''

# Define orbital configurations with initial parameters
orbital_configs = [
    {'shell': 's', 'n_primitives': 10, 'initial_alpha': 0.0005},
    {'shell': 'p', 'n_primitives': 8, 'initial_alpha': 0.0005},
    #{'shell': 'd', 'n_primitives': 6, 'initial_alpha': 0.005},
    #{'shell': 'f', 'n_primitives': 4, 'initial_alpha': 0.01}
]

# Create initial molecule with default basis
mol = m2d.Molecule(
    geom,
    "Li:s10(0.0005,2.0);p8(0.001,2.0);d6(0.005,2.0);f4(0.01,2.0)",
    charge=0,
    mult=2
)

# Initialize optimizer
optimizer = BasisOptimizer(
    molecule=mol,
    element='Li',
    orbital_configs=orbital_configs,
    beta=2.0,
    method='CCSD',
    opt_method='L-BFGS-B'
)

# Extract initial alphas from config
initial_alphas = [cfg['initial_alpha'] for cfg in orbital_configs]

# Run optimization
result = optimizer.optimize(
    initial_alphas=initial_alphas,
    plot=True
)

# Print results
print("\nOptimization Results:")
for i, config in enumerate(orbital_configs):
    shell = config['shell']
    opt_alpha = result['optimal_alphas'][i]
    print(f"Optimized {shell}-orbital α: {opt_alpha:.5e}")
print(f"Final Energy: {result['optimal_energy']:.8f} Hartree")
print(f"Total Iterations: {result['n_iter']}")
