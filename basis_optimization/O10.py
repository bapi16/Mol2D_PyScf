import Mol2D as m2d
from Mol2D.optimizer import BasisOptimizer

geom = '''
Li 0.0 0.0
units bohr
'''

initial_alpha = 0.0005
mol = m2d.Molecule(geom, f"Li:s10({initial_alpha},2.0)", charge=0, mult=2)

optimizer = BasisOptimizer(
    molecule=mol,
    element='Li',
    shell_type='s',
    n_primitives=10,
    beta=2.0,
    method='CCSD'
)

# Explicitly enable plotting
result = optimizer.optimize(initial_alpha=initial_alpha, plot=True)

print("\nOptimization Results:")
print(f"Total evaluations: {len(result['alphas'])}")
print(f"Final energy: {result['optimal_energy']:.8f} Hartree")

print("\nOptimization Results:")
print(f"Optimal alpha: {result['optimal_alpha']:.5e}")
print(f"Final energy: {result['optimal_energy']:.8f} Hartree")
print(f"Total evaluations: {len(result['alphas'])}")

print("\nEvaluation History:")
for a, e in zip(result['alphas'], result['energies']):
    print(f"α = {a:.5e} \t Energy = {e:.8f}")
