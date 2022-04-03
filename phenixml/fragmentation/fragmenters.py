import itertools
import numpy as np
import tqdm

from rdkit import Chem

from phenixml.utils.rdkit_utils import enumerate_angles, enumerate_bonds, enumerate_torsions
from phenixml.fragments.fragments import Fragment, MolContainer
from phenixml.utils.mp_utils import pool_with_progress

class FragmenterBase:
    
  """
  Base class to generate molecular fragments from a molecule container
  
  
  To make a new fragmenter, subclass this object and write a custom
  fragmentation method which takes a MolContainer object as an 
  argument, and returns a list of Fragment objects
  
  def _fragment(self,mol_container):
    
    # custom code
    
    return fragment_list
    
  """
  @classmethod
  def from_container_list(cls,containers,disable_progress=False,**kwargs):
    fragmenter = cls(**kwargs)
    fragments = pool_with_progress(fragmenter,containers,**kwargs)
    return list(itertools.chain.from_iterable(fragments))
  
  def __init__(self,exclude_elements=[],**kwargs):
    self.exclude_elements = exclude_elements
  
  def __call__(self,obj,**kwargs):
    return self.fragment(obj,**kwargs)
  
  def fragment(self,container,**kwargs):
    if isinstance(container,list):
        return self.from_container_list(container,exclude_elements=self.exclude_elements,**kwargs)
    
    assert isinstance(container,MolContainer), "Pass a MolContainer instance"
    return self._fragment(container,**kwargs)

  def _fragment(self,container,**kwargs):
    raise NotImplementedError
  

  def _return_fragments(self,fragments):
    if len(self.exclude_elements)>0:
      # print("trimming")
      # trimmed_fragments = []
      # for fragment in fragments:
      #   failed = False
      #   for e in exclude_elements:
      #     if e in fragment.elements:
      #       failed = True
      #   if not failed:
      #     trimmed_fragments.append(fragment)
      # fragments = trimmed_fragments
          
      for e in self.exclude_elements:
        fragments = [frag for frag in fragments if e not in frag.elements]
     
        
    return fragments

class BondFragmenter(FragmenterBase):
  """
  Return the bonded pair fragments for a molecule
  """

      
  def _fragment(self,container):
    assert isinstance(container,MolContainer)
    
    
    # rdkit
    idxs = enumerate_bonds(container.rdkit_mol)
    fragments = [Fragment(container,atom_selection=idx) for idx in idxs]
    return self._return_fragments(fragments)
    
class AngleFragmenter(FragmenterBase):
  """
  Generate the angle fragments for a molecule
  """
  

    
  def _fragment(self,container):
    
    # rdkit
    angle_idxs = enumerate_angles(container.rdkit_mol)
    fragments = [Fragment(container,atom_selection=angle_idx) for angle_idx in angle_idxs]
    return self._return_fragments(fragments)
  

class SmallMoleculeFragmenter(FragmenterBase):
  """
  Return a list of disconnected small molecules
  
  Will ignore macromolecules and water.
  """

  def _fragment(self,container):
    assert isinstance(container,MolContainer)
    
    
    # cctbx
    ligand_sel = container.cctbx_model.selection("not protein and not nucleotide and not water").as_numpy_array()
    ligand_sel = np.where(ligand_sel==True)[0]
    

    ligand_fragments = [Fragment(container,atom_selection=ligand_sel)]
    
    return self.return_fragments(fragments)
    
    
class MoleculeFragmenter(FragmenterBase):
    """
    Split a molecule into unbonded fragments
    """
    
    
    def _fragment(self,container):
        assert isinstance(container,MolContainer)
        
        # rdkit
        frag_indices = Chem.GetMolFrags(container.rdkit_mol, asMols=False,sanitizeFrags=False)
        fragments = [Fragment(container,atom_selection=inds) for inds in frag_indices]
        return self.return_fragments(fragments)
