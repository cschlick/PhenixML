from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
import copy
import numpy as np

periodic_table_symbol_keys = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5,
                            "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
                            "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
                            "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
                            "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
                            "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
                            "Zn": 30, "Ga": 31, "Ge": 32, "As": 33,
                            "Se": 34, "Br": 35, "Kr": 36, "Rb": 37,
                            "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
                            "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46,
                            "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
                            "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55,
                            "Ba": 56, "La": 57, "Ce": 58, "Pr": 59,
                            "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63,
                            "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67,
                            "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
                            "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76,
                            "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
                            "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84,
                            "At": 85, "Rn": 86, "Fr": 87, "Ra": 88,
                            "Ac": 89, "Th": 90, "Pa": 91, "U": 92, "Np": 93,
                            "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97,
                            "Cf": 98, "Es": 99, "Fm": 100, "Md": 101,
                            "No": 102, "Lr": 103, "Rf": 104, "Db": 105,
                            "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109,
                            "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113,
                            "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117,
                            "Og": 118}


periodic_table_number_keys = {value: key for key, value in periodic_table_symbol_keys.items()}


def mol2d(mol,print_indices=False,rmH=True,size=600):
  IPythonConsole.molSize = size,size
  mol_copy = copy.deepcopy(mol)
  if rmH:
    mol_copy, idx_dict = remove_H(mol_copy)
  ret = Chem.rdDepictor.Compute2DCoords(mol_copy)
  
  if print_indices:
    for atom in mol_copy.GetAtoms():
      atom.SetProp("atomNote", str(atom.GetIdx()))
  return mol_copy



def remove_H(rdmol):

  atoms = rdmol.GetAtoms()
  mol = Chem.Mol()
  rwmol = Chem.RWMol(mol)

  idx_map = {} # new_idx:old_idx
  for i,atom in enumerate(rdmol.GetAtoms()):
    old_idx = atom.GetIdx()
    atomic_number = atom.GetAtomicNum()
    if atomic_number !=1:
      to_delete = False
    else:
      to_delete = True
      nbrs = atom.GetNeighbors()
      if len(nbrs)!=1:
        to_delete = False
      else:
        nbr = nbrs[0]
        if nbr.GetSymbol() != "C":
          to_delete = False
    if not to_delete:
      new_idx = rwmol.AddAtom(Chem.Atom(atomic_number))
      idx_map[new_idx]=old_idx
  idx_map_rev = {value:key for key,value in idx_map.items()}
  for i,bond in enumerate(rdmol.GetBonds()):
    start,end = bond.GetBeginAtom(), bond.GetEndAtom()
    start_idx, end_idx = start.GetIdx(), end.GetIdx()
    if start_idx in idx_map.values() and end_idx in idx_map.values():

      new_start = idx_map_rev[start_idx]
      new_end = idx_map_rev[end_idx]
      bond_idx = rwmol.AddBond(new_start,new_end,bond.GetBondType())

  mol = rwmol.GetMol()
  return mol, idx_map




def elbow_to_rdkit(elbow_mol):
  """
  A simple conversion using atoms and bonds. 
  Presumably a lot of info is lost that could also
  be transfered.
  """

  # elbow bond order to rdkit bond orders
  bond_order_elbowkey = {
    1.5:Chem.rdchem.BondType.AROMATIC,
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
  }
  bond_order_rdkitkey = {value:key for key,value in bond_order_elbowkey.items()}


  atoms = list(elbow_mol)

  mol = Chem.Mol()
  rwmol = Chem.RWMol(mol)
  conformer = Chem.Conformer(len(atoms)) 

  for i,atom in enumerate(atoms):
    xyz = atom.xyz
    atomic_number = atom.number
    rdatom = rwmol.AddAtom(Chem.Atom(int(atomic_number)))
    conformer.SetAtomPosition(rdatom,xyz)

  for i,bond in enumerate(elbow_mol.bonds):
    bond_atoms = list(bond)
    start,end = atoms.index(bond_atoms[0]), atoms.index(bond_atoms[1])
    order = bond_order_elbowkey[bond.order]
    rwmol.AddBond(int(start),int(end),order)

  rwmol.AddConformer(conformer)
  mol = rwmol.GetMol()
  return mol


def cctbx_model_to_rdkit(model,iselection=None):
  if iselection is not None:
    from cctbx.array_family import flex
    isel = flex.size_t(iselection)
    sel_model = model.select(isel)
  else:
    sel_model = model
  # probably should do this through the GRM, but this
  # works
  m = Chem.MolFromPDBBlock(sel_model.model_as_pdb())
  return m



def enumerate_bonds(mol):
  idx_set = set()
  for atom in mol.GetAtoms():
    for neigh1 in atom.GetNeighbors():
      idx1,idx2 = atom.GetIdx(), neigh1.GetIdx()
      s = frozenset([idx1,idx2])
      if len(s)==2:
        idx_set.add(s)
  # check that it matches a more direct approach
  check = {frozenset((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())) for bond in mol.GetBonds()}
  assert idx_set == check
  return np.array(sorted([list(s) for s in idx_set]))

def enumerate_angles(mol):
  idx_set = set()
  for atom in mol.GetAtoms():
    for neigh1 in atom.GetNeighbors():
      for neigh2 in neigh1.GetNeighbors():
        idx0,idx1,idx2 = atom.GetIdx(), neigh1.GetIdx(),neigh2.GetIdx()
        s = (idx0,idx1,idx2)
        if len(set(s))==3:
          if idx0>idx2:
            idx0,idx2 = idx2,idx0
          idx_set.add((idx0,idx1,idx2))
  return np.array([list(s) for s in idx_set])

def enumerate_torsions(mol):
  idx_set = set()
  for atom in mol.GetAtoms():
    for neigh1 in atom.GetNeighbors():
      for neigh2 in neigh1.GetNeighbors():
        for neigh3 in neigh2.GetNeighbors():
          idx0,idx1,idx2,idx3 = atom.GetIdx(), neigh1.GetIdx(),neigh2.GetIdx(), neigh3.GetIdx()
          s = (idx0,idx1,idx2,idx3)
          if len(set(s))==4:
            if idx0>idx3:
              idx0,idx3 = idx3,idx0
            idx_set.add((idx0,idx1,idx2,idx3))
  return np.array([list(s) for s in idx_set])