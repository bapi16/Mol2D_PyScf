import Mol2D as m2d

geom = '''
He 0.0 0.0
units bohr
'''

# Create initial molecule with minimal basis for faster initial calculations
mol2d = m2d.Molecule(geom, "He:6s", charge=0, mult=1)  # Start with smaller basis

# Create optimizer with gradual basis expansion
optimizer = m2d.BasisOptimizer(
    molecule=mol2d,
    element='He',
    shell_types=['s'],        # Optimize both s and p orbitals
    n_primitives=[6],           # Start with fewer primitives
    beta=[2.0],               # Different contraction factors
    method='CCSD(T)',              # More accurate method
    opt_method='L-BFGS-B',         # Better for smooth optimization
    initial_alphas=[mol2d._get_default_exp('He', 's'),
    #mol2d._get_default_exp('He', 'p'), # 0.0005 from default
    ]    # Better starting guesses
)

# Optimization parameters
result = optimizer.optimize(
    tol=1e-6,        # Looser tolerance for initial runs
    maxiter=30,      # Fewer iterations for testing
    plot=True        # Visualize convergence
)

# Refine with tighter tolerance
if result['success']:
    final_optimizer = m2d.BasisOptimizer(
        molecule=mol2d,
        element='He',
        shell_types=['s'],
        n_primitives=[6],       # Final target basis size
        beta=[2.0],
        method='CCSD(T)',
        initial_alphas=result['optimal_alphas']
    )
    
    final_result = final_optimizer.optimize(
        tol=1e-8,
        maxiter=50,
        plot=True
    )

    print(f"\nFinal optimal exponents: {final_result['optimal_alphas']}")
    print(f"CCSD(T) energy: {final_result['optimal_energy']:.10f} Hartree")
else:
    print("Initial optimization failed. Check basis set parameters.")
