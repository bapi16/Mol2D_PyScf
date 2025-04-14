import Mol2D as m2d
from Mol2D.optimizer import BasisOptimizer

geom = '''
Li 0.0 0.0
units bohr
'''

# Configuration for each orbital type
orbital_configs = {
    's': {'n_primitives': 10, 'initial_alpha': 0.0005},
    'p': {'n_primitives': 8, 'initial_alpha': 0.001},
    'd': {'n_primitives': 6, 'initial_alpha': 0.005},
    'f': {'n_primitives': 4, 'initial_alpha': 0.01}
}

beta = 2.0
optimized_params = {}

# Sequentially optimize each orbital type
for shell_type, config in orbital_configs.items():
    print(f"\nOptimizing {shell_type}-orbital:")
    
    # Create basis string for current orbital only
    basis_str = f"Li:{shell_type}{config['n_primitives']}({config['initial_alpha']},{beta})"
    
    # Create temporary molecule
    temp_mol = m2d.Molecule(geom, basis_str, charge=0, mult=2)
    
    # Optimize current orbital
    optimizer = BasisOptimizer(
        molecule=temp_mol,
        element='Li',
        shell_type=shell_type,
        n_primitives=config['n_primitives'],
        beta=beta,
        method='CCSD'
    )
    
    result = optimizer.optimize(initial_alpha=config['initial_alpha'], plot=True)
    
    # Store optimized parameters
    optimized_params[shell_type] = {
        'alpha': result['optimal_alpha'],
        'n_primitives': config['n_primitives']
    }

# Create combined basis string
basis_parts = [
    f"{shell}{cfg['n_primitives']}({cfg['alpha']:.5e},{beta})"
    for shell, cfg in optimized_params.items()
]
combined_basis = "Li:" + "+".join(basis_parts)  # Use '+' instead of ';'

# Create final molecule with combined basis
final_mol = m2d.Molecule(geom, combined_basis, charge=0, mult=2)

# Verify combined basis calculation
final_energy = final_mol.scf(method='CCSD')['energy']

print("\nFinal Combined Results:")
for shell, params in optimized_params.items():
    print(f"{shell}-orbital: α={params['alpha']:.5e}, primitives={params['n_primitives']}")
print(f"Combined Basis Energy: {final_energy:.8f} Hartree")
