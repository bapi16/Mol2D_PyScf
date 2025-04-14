from Mol2D import Molecule


geom = '''
Li 0.0 0.0


units bohr
'''
basis_str = "Li:36s"
mol = Molecule(geom, basis_str, charge=0, mult=2)
result = mol.scf(method='uhf',verbose=3)
    
#print("\nFinal Results:")
#print(f"Total energy: {result['energy']:.6f} Hartree")
#print(f"Alpha electrons: {result['n_alpha']}")
#print(f"Beta electrons: {result['n_beta']}")
