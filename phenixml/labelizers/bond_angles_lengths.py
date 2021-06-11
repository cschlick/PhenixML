import numpy as np
from rdkit import Chem

class BondFragLabeler:
  def __init__(self,assert_2_atoms=True,atom_indices=None):
    self.assert_2_atoms = assert_2_atoms
    self.atom_indices = atom_indices
    if self.atom_indices is None:
      self.atom_indices = [0,1]
    
      
  def labelize(self,frag):
    if self.assert_2_atoms:
      assert len(frag)==2
    assert(len(frag.rdmol.GetConformers())>0)
    conf = frag.rdmol.GetConformer()
    i,j = [frag.atom_indices[i] for i in self.atom_indices]
    bond_length = Chem.rdMolTransforms.GetBondLength(conf,i,j)
    return bond_length
    
    
class AngleFragLabeler:
  def __init__(self,assert_3_atoms=True,atom_indices=None,degrees=True):
    self.assert_3_atoms = assert_3_atoms
    self.atom_indices = atom_indices
    self.degrees= True
    if self.atom_indices is None:
      self.atom_indices = [0,1,2]
    
      
  def labelize(self,frag):
    if self.assert_3_atoms:
      assert len(frag)==3
    assert(len(frag.rdmol.GetConformers())>0)
    conf = frag.rdmol.GetConformer()
    i,j,k = [frag.atom_indices[i] for i in self.atom_indices]
    if self.degrees:
      angle = Chem.rdMolTransforms.GetAngleDeg(conf,i,j,k)
    else:
      angle = Chem.rdMolTransforms.GetAngleRad(conf,i,j,k)
    return angle
    