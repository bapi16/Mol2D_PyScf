from Mol2D.molecule import Molecule
from Mol2D.optimizer import MethodAwareOptimizer

# Define the molecule geometry (in Bohr or Angstrom)
geom = """
units bohr
H  0.0  0.0
H
"""

# Initialize the molecule with the correct basis_str format
molecule = Molecule(geom, basis_str="H: 10s", charge=0, mult=1)

# Define the target element and basis set parameters
target_element = "H"
shells = ["s"]  # Shells to optimize (e.g., 's', 'p', 'd')
n_primitives = {"s": 2}  # Number of primitives for each shell
initial_alpha = {"s": 0.5}  # Initial alpha values for each shell
beta = 2.0  # Fixed beta value

# Initialize the optimizer
optimizer = MethodAwareOptimizer(
    molecule=molecule,
    target_element=target_element,
    shells=shells,
    n_primitives=n_primitives,
    initial_alpha=initial_alpha,
    beta=beta
)

# Run optimization for a specific method (e.g., CCSD(T))
result_ccsd = optimizer.optimize(method="cisd", maxiter=50, tol=1e-5)
print("CCSD(T) Optimization Result:")
print(f"Optimal Alpha: {result_ccsd['optimal_alpha']}")
print(f"Final Energy: {result_ccsd['energy']} Hartree")

# Compare different basis sets
#basis_combinations = {
    #"5s": {"s": 5},
    #"7s": {"s": 7},
    #"20s": {"s": 20}
#}
#results = optimizer.compare_basis_sets(basis_combinations, method="ccsd(t)")
#print("\nBasis Set Comparison Results:")
#for basis, result in results.items():
#print(f"{basis}: Energy = {result['energy']} Hartree")

# Plot convergence for a specific optimization
#optimizer.plot_convergence(results)
