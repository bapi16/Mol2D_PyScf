from Mol2D import MethodAwareOptimizer

# H2 molecule geometry (in bohr)
geom = '''
H 0.0 0.0

'''

# Initialize and run optimization
optimizer = MethodAwareOptimizer(
    geom=geom,
    initial_params=[0.5, 36],  # Start with α₀=0.5, n=3
    target_element='H',
    beta=2.0,
    max_n=36,
    method='ccsd',
    cc_settings={'conv_tol': 1e-8}
)

result = optimizer.optimize(
    alpha_bounds=(0.006, 1.0),
    n_bounds=(20, 36),
    maxiter=300
)

print(f"\nOptimized Parameters:")
print(f"α₀ = {result['alpha0']:.6f}")
print(f"n = {result['n']}")
print(f"Final Energy: {result['energy']:.8f} Hartree")

# Visualize optimization progress
optimizer.plot_results()
