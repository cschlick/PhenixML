from rdkit import Chem
import numpy as np
from itertools import combinations

from phenixml.utils.rdkit_utils import enumerate_angles

from phenixml.fragmentation.fragments_base import Fragment

  
class BondFragmenter:
  def __init__(self,exclude_symbols=[]):
    self.exclude_symbols = exclude_symbols
  
  def fragment(self,rdmol):
    indices = []
    for bond in rdmol.GetBonds():
      i,j = bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
      si,sj = rdmol.GetAtomWithIdx(i).GetSymbol() ,rdmol.GetAtomWithIdx(j).GetSymbol()
      if set([si,sj]).issubset(set(self.exclude_symbols)):
        pass
      else:
        indices.append([i,j])
    frags = [Fragment(rdmol,atom_indices=ind) for ind in indices]
    return frags


class AngleFragmenter:
  def __init__(self,exclude_symbols=[]):
    self.exclude_symbols = exclude_symbols

  def fragment(self,rdmol):
    
    angle_idxs = enumerate_angles(rdmol)
    keep_indices = []
    for i,j,k in angle_idxs:
      i,j,k = int(i), int(j) ,int(k)
      atoms = [rdmol.GetAtomWithIdx(i), rdmol.GetAtomWithIdx(j), rdmol.GetAtomWithIdx(k)]
      symbols = [atom.GetSymbol() for atom in atoms]
      reject = False
      for s in symbols:
        if s in self.exclude_symbols:
          reject = True
      if not reject:
        keep_indices.append([i,j,k])
    frags = [Fragment(rdmol,atom_indices=ind) for ind in keep_indices]
    return frags
