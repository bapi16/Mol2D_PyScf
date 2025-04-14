# File: run_ccsd.py
import Mol2D as m2d
from pyscf import cc

geom = '''
H   0.0  0.0

units bohr
'''

mol2d = m2d.Molecule(
    geom,
    basis_str="H:2s",
    charge=0,
    mult=2  # Doublet state (required for odd electrons)
)

mf = mol2d.pyscf("uhf")  # Must use UHF
mf.kernel()
mycc = cc.CCSD(mf)


