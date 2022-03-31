from rdkit import Chem
from rdkit.Chem import rdMolTransforms
import numpy as np

class FragLabelerBase:
  def __init__(self,name="label"):
    self.name = label
  
  def __call__(self,obj):
    return self.label(obj)
  
  def label(self,obj):
    raise NotImplementedError


class AngleFragLabeler(FragLabelerBase):
  """
  Return the angle of three atoms in a fragment.
  """
  
  def __init__(self,degrees=False,assert_bonded=True,name="angle"):
    self.degrees=degrees
    self.assert_bonded = True
    self.name = name
    
  def label(self,fragment):
    
    assert len(fragment)==3, "Cannot calculate angle for fragment if not containing 3 atoms"
    
    # rdkit
    rdkit_mol = fragment.rdkit_mol
    
    # first make sure the middle atom is the middle of the selection. For now we don't try to fix.
    idx0,idx1,idx2 = fragment.atom_selection
    idx0,idx1,idx2 = int(idx0), int(idx1), int(idx2)

    atom0,atom1,atom2 = rdkit_mol.GetAtomWithIdx(idx0), rdkit_mol.GetAtomWithIdx(idx1), rdkit_mol.GetAtomWithIdx(idx2)
    nbrs = [nbr.GetIdx() for nbr in atom1.GetNeighbors()]
    assert atom0.GetIdx() in nbrs and atom2.GetIdx() in nbrs, "This angle fragment is malformed. Atom0 or Atom2 are not bonded to Atom 1"
    conf = rdkit_mol.GetConformer()
    angle_rad = rdMolTransforms.GetAngleRad(conf,idx0,idx1,idx2)
    if self.degrees:
      angle = np.degrees(angle_rad)
    else:
      angle = angle_rad
    return angle
    