import numpy as np
from multiprocessing import Pool
from contextlib import closing
from tqdm.notebook import tqdm

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from phenixml.fragmentation.fragments import MolContainer, Fragment

class FragmentLabelerBase:
    """
    Base class to generate labels for fragment objects


    To make a new labeler, subclass this object and write a custom
    label method which takes a fragment as an 
    argument, and returns a numpy label vector

    def _label(self,fragment):

      # custom code

      return label_vector

    """
    @staticmethod
    def _label_fragment_list(instance,fragments,**kwargs):
      
      labels = [instance.label(frag) for frag in fragments]      
      return np.vstack(labels)
    
    def __call__(self,fragment,**kwargs):
        return self.label(fragment,**kwargs)
    
    def label(self,fragment,**kwargs):
        if isinstance(fragment,list):
            return self._label_fragment_list(self,fragment,**kwargs)
        if isinstance(fragment,MolContainer):
            fragment = fragment.full_fragment
        
        assert isinstance(fragment,Fragment), "Pass a Fragment object"
        
        return self._label(fragment,**kwargs)
    
    def _label(self,obj,**kwargs):
        raise NotImplementedError

class BondFragmentLabeler(FragmentLabelerBase):
  """
  Return the bond length of two atoms in a fragment.
  """
  
  def __init__(self,assert_bonded=True,name="bond"):
    self.assert_bonded = True
    self.name = name
    
  def _label(self,fragment):
    
    assert len(fragment)==2, "Cannot calculate length for bond fragment if not containing 2 atoms"
    
    # rdkit
    rdkit_mol = fragment.rdkit_mol
    
    # first make sure the atoms are bonded
    idx0,idx1 = fragment.atom_selection
    idx0,idx1 = int(idx0), int(idx1)

    atom0,atom1= rdkit_mol.GetAtomWithIdx(idx0), rdkit_mol.GetAtomWithIdx(idx1)
    nbrs = [nbr.GetIdx() for nbr in atom0.GetNeighbors()]
    assert atom1.GetIdx() in nbrs, "This angle fragment is malformed to be a bond fragment. Atom0 and Atom1 are not bonded"
    conf = rdkit_mol.GetConformer()
    length = rdMolTransforms.GetBondLength(conf,idx0,idx1)
    return np.array(length)


class AngleFragmentLabeler(FragmentLabelerBase):
  """
  Return the angle of three atoms in a fragment.
  """
  
  @staticmethod
  def _featurize_fragment_list(instance,fragments,disable_progress=True,nproc=1,**kwargs):
    worker = instance.featurize
    work = fragments
    results = []
    with closing(Pool(processes=nproc)) as pool:
      for result in tqdm(pool.map(worker, work), total=len(work)):
          results.append(result)
      pool.terminate()
  
  def __init__(self,degrees=False,assert_bonded=True,name="angle"):
    self.degrees=degrees
    self.assert_bonded = True
    self.name = name
    
  def _label(self,fragment):
    
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
    
    
    
  