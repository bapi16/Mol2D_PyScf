import Mol2D as m2d
from pyscf import cc 

geom = '''
Be 0.0 0.0
units bohr
'''

# Create Mol2D molecule
mol2d = m2d.Molecule(geom, "Be:36s", charge=0, mult=1)

# Run Mol2D's SCF (optional)
# mol2d.scf("hf")

# Generate PySCF SCF object with Mol2D integrals
mf = mol2d.pyscf("uhf")

# Run PySCF's SCF calculation
mf.kernel()

# Run CCSD using PySCF
mycc = cc.CCSD(mf)
mycc.kernel()
print("CCSD energy:", mycc.e_tot)
