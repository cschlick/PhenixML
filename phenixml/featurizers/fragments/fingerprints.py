import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from phenixml.featurizers.fragments.base import FragmentFeaturizerBase




class MorganFeaturizer(FragmentFeaturizerBase):
  def __init__(self,radius=1,nBits=1024,useChirality=True,useBondTypes=True,useFeatures=True):
    self.radius = radius
    self.nBits = nBits
    self.useChirality = useChirality
    self.useBondTypes = useBondTypes
    self.useFeatures = useFeatures
    
  def _featurize(self,fragment,**kwargs):
    
    if isinstance(fragment,list):
        return self._featurize_fragment_list(self,fragment,**kwargs)
    

    fromAtoms = fragment.atom_selection.tolist()
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
          fragment.rdkit_mol,
          self.radius,
          fromAtoms = fromAtoms,
          nBits=self.nBits,
          useChirality=self.useChirality,
          useBondTypes=self.useBondTypes,
          useFeatures=self.useFeatures)
    return np.array(fp)


class RDKFingerprint(FragmentFeaturizerBase):

    def _featurize(self,fragment,**kwargs):

        if isinstance(fragment,list):
            return self._featurize_fragment_list(self,fragment,**kwargs)

        if len(fragment.atom_selection) == len(fragment):
          fromAtoms = []
        else:
          fromAtoms = fragment.atom_selection.tolist()
        mol = fragment.rdkit_mol
        fp = Chem.RDKFingerprint(mol,fromAtoms=fromAtoms)
        return np.array(fp)
    
    
