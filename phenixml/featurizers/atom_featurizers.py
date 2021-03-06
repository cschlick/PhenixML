import numpy as np
from rdkit import Chem
pt  = Chem.GetPeriodicTable()

try:
  import cctbx
  from iotbx.pdb.hierarchy import atom as cctbx_atom
  from mmtbx.model.model import manager as model_manager
except:
  pass # no cctbx

# type checks
def is_rdkit_atom(atom):
  return isinstance(atom,Chem.rdchem.Atom)
def is_rdkit_mol(mol):
  return isinstance(mol,Chem.rdchem.Mol)

def is_cctbx_atom(atom):
  return isinstance(atom,cctbx_atom)
def is_cctbx_mol(mol):
  return isinstance(mol,model_manager)


class AtomFeaturizer_Element:
  """
    One-hot encode the Element symbol of an atom.
  
  Usage:
  
  featurizer = AtomFeaturizer_Element()
  featurizer = AtomFeaturizer_Element(elements=["C","H","X"] # or use a restricted set
  
  feat = featurizer.featurize_atom(atom)      # where atom is either an RDkit or CCTBX atom object
  feat = featurizer.featurize_molecule(mol)   # where mol is either an RDkit Chem.Mol or CCTBX mmtbx.model.model.manager object
  Note: 
  If encode_unknown=True and "X" is not present in element list, "X" will be added to encode unknown elements.
  """
  elements_default = ["C","H","N","O","S","X"] # X is other
  elements_full = [pt.GetElementSymbol(i) for i in range(117)]
  
  def __init__(self,elements=[],dtype=np.float32,encode_unknown=True):
    self.dtype=dtype
    if len(elements)==0:
      self.elements = self.elements_default
    else:
      self.elements = elements
    
    if encode_unknown:
      if "X" not in self.elements:
        self.elements.append("X")
    else:
      if "X" in self.elements:
        self.elements.remove("X")
   
  def __len__(self):
    return len(self.elements)
  
  def __call__(self,atom):
    return self.featurize_atom(atom)
  
  def featurize_element(self,element):
    if element in self.elements:
      index = self.elements.index(element)
    else:
      assert "X" in self.elements, "Unknown element: "+element
      index = self.elements.index("X")
    
    feat = np.zeros(len(self.elements)).astype(self.dtype)
    feat[index]=1
    return feat
  
  def featurize_atom(self,atom):
    if is_rdkit_atom(atom):
      element = pt.GetElementSymbol(atom.GetAtomicNum())
    elif is_cctbx_atom(atom):
      element = atom.element.strip()
    
    return self.featurize_element(element)
  
  def featurize_molecule(self,mol):
    feats = []
    if is_rdkit_mol(mol):
      rdmol = mol
      for atom in rdmol.GetAtoms():
        element = pt.GetElementSymbol(atom.GetAtomicNum())
        feat = self.featurize_element(element)
        feats.append(feat)
        
    elif is_cctbx_mol(mol):
      model = mol
      for atom in model.get_atoms():
        element = atom.element.strip()
        feat = self.featurize_element(element)
        feats.append(feat)
      
    return np.vstack(feats)
  
  def invert_feature(self,feat):
    single = False
    if feat.ndim==1:
      feats = feat[np.newaxis,:]
      single = True
    else:
      feats = feat
    ret_vals = []
    for feat in feats:
      index = np.argwhere(feat>0).flatten()
      assert len(index)==1, "One hot feature vector should have only one nonzero value."
      index = index[0]
      ret = self.elements[index]
      ret_vals.append(ret)
    if single:
      return ret_vals[0]
    else:
      return ret_vals


class AtomFeaturizer_Residue:
  """
  One-hot encode the residue name of an atom. Intented for protein.
  
  Usage:
  
  featurizer = AtomFeaturizer_Residues()
  featurizer = AtomFeaturizer_Residues(residues=['ALA', 'ARG', 'ASN', 'ASP','UNK']) # or use a restricted set
  
  feat = featurizer.featurize_atom(atom)    # where atom is either an RDkit or CCTBX atom object
  feat = featurizer.featurize_molecule(mol) # where mol is either an RDkit Chem.Mol or CCTBX mmtbx.model.model.manager object
  
  Note: 
  If encode_unknown=True and "UNK" is not present in residue list, "UNK" will be added to encode unknown residues.
  
  """
  residues_default = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL','UNK']
  def __init__(self,residues=[],encode_unknown=True,dtype=np.float32):
    self.dtype=dtype
    if len(residues)==0:
      self.residues = self.residues_default
    else:
      self.residues = residues
    
    if encode_unknown:
      if "UNK" not in self.residues:
        self.residues.append("UNK")
    else:
      if "UNK" in self.residues:
        self.residues.remove("UNK")

  def __len__(self):
    return len(self.residues)

  def __call__(self,atom):
    return self.featurize_atom(atom)
  
  def featurize_resname(self,resname):
    
    resname = resname.strip()
    resname = resname.upper()
    if resname in self.residues:
      index = self.residues.index(resname)
    else:
      assert "UNK" in self.residues, "Unknown residue: "+resname
      index = self.residues.index("UNK")
      
    feat = np.zeros(len(self.residues)).astype(self.dtype)
    feat[index]=1
    return feat
  
  def featurize_atom(self,atom):
    if is_rdkit_atom(atom):
      resinfo = atom.GetPDBResidueInfo()
      assert resinfo is not None, "Trying to featurize an atom using PDB residue information, but atom.PDBResidueInfo() is None"
      resname = resinfo.GetResidueName()
      resname = resname.strip()
    elif is_cctbx_atom(atom):
      resname = atom.parent().resname
    
    return self.featurize_resname(resname)
  
  def featurize_molecule(self,mol):
    feats = []
    if is_rdkit_mol(mol):
      rdmol = mol
      for atom in rdmol.GetAtoms():
        resinfo = atom.GetPDBResidueInfo()
        assert resinfo is not None, "Trying to featurize an atom using PDB residue information, but atom.PDBResidueInfo() is None"
        resname = resinfo.GetResidueName()
        resname = resname.strip()
        feat = self.featurize_resname(resname)
        feats.append(feat)
        
    elif is_cctbx_mol(mol):
      model = mol
      for atom in model.get_atoms():
        resname = atom.parent().resname
        resname = resname.strip()
        feat = self.featurize_resname(resname)
        feats.append(feat)
      
    return np.vstack(feats)
  
  
  def invert_feature(self,feat):
    single = False
    if feat.ndim==1:
      feats = feat[np.newaxis,:]
      single = True
    else:
      feats = feat
    ret_vals = []
    for feat in feats:
      index = np.argwhere(feat>0).flatten()
      assert len(index)==1, "One hot feature vector should have only one nonzero value."
      index = index[0]
      ret = self.residues[index]
      ret_vals.append(ret)
    if single:
      return ret_vals[0]
    else:
      return ret_vals

  

class ConcatAtomFeaturizer:
  """
  Concatenate multiple atom featurizers together
  
  Usage:
  
  f1 = AtomFeaturizer_Element()
  f2 = AtomFeaturizer_Residue()
  featurizer = ConcatAtomFeaturizer(featurizers=[f1,f2])
  
  feat = featurizer.featurize_atom(atom)      # where atom is either an RDkit or CCTBX atom object
  feat = featurizer.featurize_molecule(mol)   # where mol is either an RDkit Chem.Mol or CCTBX mmtbx.model.model.manager object
  
  Notes:
  feature_inputs = featurizer.invert_feature(feat) 
  
  """
  def __init__(self,featurizers=[]):
    assert len(featurizers)>0, "Must provide a list of Atom featurizer instances"
    self.featurizers = featurizers
    assert len(set([f.dtype for f in self.featurizers]))==1, "Cannot concatenate featurizers of different dtypes"
    
  def __call__(self,atom):
    return self.featurize_atom(atom)
  
  def featurize_atom(self,atom):
    feats = [featurizer(atom) for featurizer in self.featurizers]
    return np.concatenate(feats)

  def featurize_molecule(self,mol):
    if is_rdkit_mol(mol):
      rdmol = mol
      atoms = rdmol.GetAtoms()
    elif is_cctbx_mol(mol):
      model = mol  
      atoms = model.get_atoms()
    feats = [np.concatenate([featurizer.featurize_atom(atom) for featurizer in self.featurizers]) for atom in atoms]
    return np.vstack(feats)
    
  def invert_feature(self,feat):


    single = False
    if feat.ndim==1:
      feats = feat[np.newaxis,:]
      single = True
    else:
      feats = feat
    ret_vals = []
    for feat in feats:
      cum_i = 0
      inverted_features = []
      for featurizer in self.featurizers:
        feat_i = feat[cum_i:cum_i+len(featurizer)]
        inverted_feature = featurizer.invert_feature(feat_i)
        inverted_features.append(inverted_feature)
        cum_i+=len(featurizer)
      ret_vals.append(inverted_features)
    if single:
      return ret_vals[0]
    else:
      return ret_vals



class RDKIT_Fingerprint:
    # based on the Espaloma rdkit fingerprint
    
    
    def __call__(self,atom):
        return self.featurize_atom(atom)

    def featurize_atom(self,rdkit_atom):
        atom = rdkit_atom
        HYBRIDIZATION_RDKIT = {
          Chem.rdchem.HybridizationType.SP: np.array(
              [1, 0, 0, 0, 0],
          ),
          Chem.rdchem.HybridizationType.SP2: np.array(
              [0, 1, 0, 0, 0],
          ),
          Chem.rdchem.HybridizationType.SP3: np.array(
              [0, 0, 1, 0, 0],
          ),
          Chem.rdchem.HybridizationType.SP3D: np.array(
              [0, 0, 0, 1, 0],
          ),
          Chem.rdchem.HybridizationType.SP3D2: np.array(
              [0, 0, 0, 0, 1],
          ),
          Chem.rdchem.HybridizationType.S: np.array(
              [0, 0, 0, 0, 0],
          ),
        }

        return np.concatenate(
              [
                  np.array(
                      [
                          atom.GetTotalDegree(),
                          atom.GetTotalNumHs(),
                          atom.GetTotalValence(),
                          atom.GetExplicitValence(),
                          atom.GetFormalCharge(),
                          atom.GetIsAromatic() * 1.0,
                          atom.GetMass(),
                          atom.IsInRingSize(3) * 1.0,
                          atom.IsInRingSize(4) * 1.0,
                          atom.IsInRingSize(5) * 1.0,
                          atom.IsInRingSize(6) * 1.0,
                          atom.IsInRingSize(7) * 1.0,
                          atom.IsInRingSize(8) * 1.0,
                      ],
                  ),
                  HYBRIDIZATION_RDKIT[atom.GetHybridization()],
              ],
          )

