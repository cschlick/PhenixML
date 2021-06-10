from rdkit import Chem

from phenixml.featurizers.featurizer_base import MolFeaturizer


def bond_order(bond_type):

  if bond_type==Chem.rdchem.BondType.SINGLE:
    bond_order = 1
  elif bond_type==Chem.rdchem.BondType.DOUBLE:
    bond_order = 2
  elif bond_type==Chem.rdchem.BondType.TRIPLE:
    bond_order = 3
  elif bond_type==Chem.rdchem.BondType.AROMATIC:
    bond_order = 4
  else:
    bond_order = -1
  return bond_order

class BondTokenizer(MolFeaturizer):
  """
  A simple featurizer that returns the bond symbols
  This is a simpler alternative to smiles which may contain additional information.
  
  
  CC for a carbon-carbon bond.
  if ignore bond_type=False, a number for bond order is inserted between the atom symbols
  
  """
  
  def __init__(self,ignore_bond_type=False):
    self.ignore_bond_type = ignore_bond_type
  
  
  def featurize(self,fragment):
      assert len(fragment) ==2, "Cannot use this featurizer for fragments larger than one bond"
      return self.bond_symbol(fragment,ignore_bond_type=self.ignore_bond_type)

    
  @staticmethod
  def bond_symbol(frag,ignore_bond_type=True):
    assert(len(frag)==2)
    i,j = frag.atom_indices
    if not ignore_bond_type:
      bond = frag.rdmol.GetBondBetweenAtoms(i,j)
      bond_type = bond.GetBondType()
      bond_ord = str(bond_order(bond_type))
      atom_symbols = frag.atom_symbols
      symbols = [atom_symbols[0],bond_ord,atom_symbols[1]]

    else:
      symbols = frag.atom_symbols
    a,b = frag.atom_numbers
    if a<b:
      symbol = symbols
    else:
      if not ignore_bond_type:
        symbol = [symbols[2],symbols[1],symbols[0]]
      else:
        symbol = [symbols[1],symbols[0]]
    return symbol


class AngleTokenizer(MolFeaturizer):
  """
  A simple featurizer that returns symbols for an angle fragment.
  This is a simpler alternative to smiles which may contain additional information
  
  CCN for a carbon-carbon-nitrogen bond
  if ignore bond_type=False, a number for bond order is inserted between the atom symbols
  """

  def __init__(self,ignore_bond_type=False):
    self.ignore_bond_type = ignore_bond_type


  def featurize(self,fragment):
      assert len(fragment) ==3, "To use this featurizer, provide a fragment with exactly 3 atoms"
      return self.angle_symbol(fragment,ignore_bond_type=self.ignore_bond_type)



  @staticmethod
  def angle_symbol(frag,ignore_bond_type=True):
    assert(len(frag)==3)
    i,j,k = frag.atom_indices
    if not ignore_bond_type:
      bonds = [frag.rdmol.GetBondBetweenAtoms(i,j),frag.rdmol.GetBondBetweenAtoms(j,k)]
      bond_types = [bond.GetBondType() for bond in bonds]
      bond_orders = [str(bond_order(bond_type)) for bond_type in bond_types]
      atom_symbols = frag.atom_symbols
      symbols = [atom_symbols[0],bond_orders[0],atom_symbols[1],bond_orders[1],atom_symbols[2]]

    else:
      symbols = frag.atom_symbols
    a,b,c = frag.atom_numbers
    if a<c:
      symbol = symbols
    else:
      if not ignore_bond_type:
        symbol = [symbols[4],symbols[3],symbols[2],symbols[1],symbols[0]]
      else:
        symbol = [symbols[2],symbols[1],symbols[0]]
    return symbol

