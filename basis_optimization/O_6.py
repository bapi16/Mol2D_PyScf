from Mol2D.molecule import Molecule
from Mol2D.optimizer import MethodAwareOptimizer
import numpy as np

# Single hydrogen atom
geom = """
units bohr
H  0.0  0.0
"""

# Basis configuration
basis_config = {
    "target_element": "H",
    "shells": ["s", "p"],
    "n_primitives": {"s": 10, "p": 5},
    "initial_alpha": {"s": 0.5, "p": 0.3},
    "beta": 2.0
}

# Initialize molecule with PROPER basis string
basis_str = f"H: {basis_config['n_primitives']['s']}s{''.join([str(basis_config['n_primitives'][s])+s for s in basis_config['shells'] if s != 's']}"
molecule = Molecule(geom, basis_str=basis_str, charge=0, mult=1)

# Initialize optimizer with SAFEGUARDS
optimizer = MethodAwareOptimizer(
    molecule=molecule,
    target_element=basis_config["target_element"],
    shells=basis_config["shells"],
    n_primitives=basis_config["n_primitives"],
    initial_alpha=basis_config["initial_alpha"],
    beta=basis_config["beta"],
    alpha_bounds=[(0.1, 5.0)]*len(basis_config["shells"])  # Constrain alpha
)

# Run optimization with STABILITY CHECKS
result = optimizer.optimize(
    method="ccsd(t)",
    maxiter=100,
    tol=1e-6,
    callback=lambda x: print(f"Current alphas: {x}")  # Monitor progress
)

print(f"Final result: {result}")
