from rdkit import Chem
import numpy as np
from itertools import combinations


from phenixml.fragmentation.fragments_base import Fragment

  
class BondFragmenter:
  def __init__(self):
    pass
  
  def fragment(self,rdmol):
    indices = [[bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()] for bond in rdmol.GetBonds()]
    frags = [Fragment(rdmol,atom_indices=ind) for ind in indices]
    return frags


class AngleFragmenter(Fragmenter):
  def __init__(self):
    pass

  def fragment(self,rdmol):
    
    angle_idxs = set()
    for atom in rdmol.GetAtoms():
      i = atom.GetIdx()
      nbrs = atom.GetNeighbors()

      if len(nbrs)>1:
        nbr_idxs = [atom.GetIdx() for atom in nbrs]
        atom_triples = [(j,i,k) for j,k in list(combinations(nbr_idxs,2))]
        atom_triples_atomic_number = [[rdmol.GetAtomWithIdx(idx).GetAtomicNum() for idx in triple] for triple in atom_triples]
        atom_triples_sorted = []
        for i,(a,b,c) in enumerate(atom_triples_atomic_number):
          if a<c:
            atom_triples_sorted.append(atom_triples[i])
          else:
            triple = atom_triples[i]
            k,j,i = triple
            atom_triples_sorted.append([i,j,k])
        atom_triples = set([tuple(triple) for triple in atom_triples_sorted])
        angle_idxs |= atom_triples
        
    frags = [Fragment(rdmol,atom_indices=ind) for ind in angle_idxs]
    return frags
