from Qchem_2D import Molecule


geom_contr = '''
H 0.0 0.0
units bohr
'''
basis_contr = "H: s [8.264000E+01, 1.241000E+01, 2.824000E+00, 7.977000E-01, 2.581000E-01, 8.989000E-02] [2.006000E-03, 1.534300E-02, 7.557900E-02, 2.568750E-01, 4.973680E-01, 2.961330E-01]"
    
mol = Molecule(geom_contr, basis_contr, charge=0, mult=2)
result = mol.scf(method='uhf',verbose=3)
