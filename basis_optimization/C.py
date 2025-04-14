# Example usage
if __name__ == "__main__":
    # Hydrogen atom calculation
    geom = '''
    H 0.0 0.0
    units bohr
    '''
    basis_str = "H: 10s"
    mol = Molecule(geom, basis_str, charge=0, mult=2)
    result = mol.scf(method='uhf')
    print(f"Hydrogen atom energy: {result['energy']:.6f} Hartree")
    print(f"Spin contamination: {result['spin_contamination']:.4f}")
