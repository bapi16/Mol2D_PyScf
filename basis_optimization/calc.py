
# calc.py
from Qchem_2D import Molecule

geom = '''
H  0.0 0.0
Li 0.0 0.1
 
units bohr
'''

mol = Molecule(geom, "H: 6s; Li: 6s 3p", charge=0, mult=2)
result = mol.scf(method='uhf',verbose=2)
'''print(f"Total Energy: {result['energy']:.12f} Hartree")
print(f"Nuclear Repulsion: {result['nuclear_repulsion']:.12f} Hartree")
print(f"Alpha electrons: {result['n_alpha']}")
print(f"Beta electrons: {result['n_beta']}")
print(f"Timing Breakdown:")
print(f"- STV integrals: {result['time_integrals_stv']:.2f}s")
print(f"- G integrals: {result['time_integrals_g']:.2f}s")
print(f"- SCF iterations: {result['time_scf']:.2f}s")
print(f"- Total time: {result['time_total']:.2f}s")'''
