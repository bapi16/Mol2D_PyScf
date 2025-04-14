import Mol2D as m2d
import os

geom = '''
Li 0.0 0.0
units bohr
'''

# Create initial molecule
mol2d = m2d.Molecule(geom, "Li:6s", charge=0, mult=2)

# Generate plot name based on current script's name
script_name = os.path.splitext(os.path.basename(__file__))[0]
plot_filename = f"{script_name}_basis_opt.png"

# Create optimizer
optimizer = m2d.BasisOptimizer(
    molecule=mol2d,
    element='Li',
    shell_types=['s'],
    n_primitives=[6],
    beta=[2.0],
    method='CISD',
    initial_alphas=[0.001]
)

# Run optimization with custom plot name
result = optimizer.optimize(tol=1e-8, maxiter=200, plot_name=plot_filename)

print(f"Optimal alphas: {result['optimal_alphas']}")
print(f"Final CISD energy: {result['optimal_energy']:.8f}")
