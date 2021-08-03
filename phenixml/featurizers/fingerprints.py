import numpy as np
from rdkit import Chem

class MorganFeaturizer:
  def __init__(self,radius=2,nBits=2048,useChirality=False,useBondTypes=True,useFeatures=False):
    self.radius = radius
    self.nBits = nBits
    self.useChirality = useChirality
    self.useBondTypes = useBondTypes
    self.useFeatures = useFeatures
    
  def featurize(self,fragment):
    if len(fragment.atom_indices) ==0 or len(fragment.atom_indices) == fragment.rdmol.GetNumAtoms():
      fromAtoms = []
    else:
      fromAtoms = fragment.atom_indices
    fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
          fragment.rdmol,
          self.radius,
          fromAtoms = fromAtoms,
          nBits=self.nBits,
          useChirality=self.useChirality,
          useBondTypes=self.useBondTypes,
          useFeatures=self.useFeatures)
    return np.array(fp)
  
  
  
class RDKFeaturizer:
  def __init__(self,fpSize=2048,radius=1):
    self.fpSize = fpSize
    self.radius = radius
    
  def featurize(self,fragment):
    if len(fragment.atom_indices) ==0 or len(fragment.atom_indices) == fragment.rdmol.GetNumAtoms():
      fromAtoms = []
    else:
      if self.radius >1:
        m,inds = fragment.extract_fragment(addDummies=False,radius=self.radius,return_inds=True)
        fromAtoms = inds
      else:
        fromAtoms = fragment.atom_indices
    fp = Chem.RDKFingerprint(fragment.rdmol,fromAtoms = fromAtoms)
    
    return np.array(fp)