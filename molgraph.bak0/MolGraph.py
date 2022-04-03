from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import numpy as np
import dgl
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

from molgraph.build_graphs import build_heterograph_from_homo_mol, build_homograph
from pathlib import Path

try:
  import cctbx
  from iotbx.data_manager import DataManager
  from mmtbx.model.model import manager as model_manager
except:
  pass # no cctbx



class AtomMolgraph:
  """
  A container object to store molecules and the resulting atom graphs
  
  """
  @classmethod
  def from_binary_file(cls,filepath):
    obj = torch.load(filepath)
    return obj
  
  @classmethod
  def from_model_file(cls,filepath):
    filepath = Path(filepath)
    suffixes = set(filepath.suffixes) # may include .gz,.zip, but will likely fail later for now
    macromol_suffixes = [".pdb",".cif",".mmcif"]
    micromol_suffixes = [".mol",".mol2"]
    
    mdata = {"filepath":str(filepath)}

    if len(suffixes.intersection(set(macromol_suffixes))) >0:
      # is macromol
      dm = DataManager()
      dm.process_model_file(str(filepath))
      model = dm.get_model()
      if model.crystal_symmetry()==None: # add crystal symmetry if missing (common for cryoem)
        from cctbx.maptbx.box import shift_and_box_model
        model = shift_and_box_model(model,shift_model=False)
      
      return cls.from_cctbx_model(model,mdata=mdata)
    
    elif len(suffixes.intersection(set(micromol_suffixes))) >0:
      # is micromol
      if ".mol2" in suffixes:
        rdmol = Chem.MolFromMol2File(str(filepath))
      elif ".mol" in suffixes:
        rdmol = Chem.MolFromMolFile(str(filepath))
        
      return cls.from_rdkit(rdmol,mdata=mdata)
    else:
      assert False, "File format not recognized. Supported suffixes: "+str(macromol_suffixes+micromol_suffixes)
      
    
  @classmethod
  def from_cctbx_model(cls,model,mdata={}):
    """
    Pending a functional geometry restraints manager on Colabs, 
    use the PDB string as an intermediate to generate a rdkit mol.
    
    Because of an issue where the number of atoms can differ between
    cctbx/rdkit when reading pdb files, the cctbx model is recreated from
    the rdkit mol. So the original model object is not retained.
    """
    
    rdmol = Chem.MolFromPDBBlock(model.model_as_pdb())
    
    dm = DataManager()
    dm.process_model_str("input",Chem.MolToPDBBlock(rdmol))
    model_new = dm.get_model(filename="input")
    
    # if model.crystal_symmetry()==None: # add crystal symmetry if missing (common for cryoem)
    #   from cctbx.maptbx.box import shift_and_box_model
    #   model = shift_and_box_model(model,shift_model=False)
    mdata["cctbx_model"] = model_new
    
    return cls(rdmol,mdata=mdata)
  
  @classmethod
  def from_rdkit(cls,rdmol,mdata={}):
    return cls(rdmol,mdata=mdata)
  
  def __init__(self,rdmol,mdata={}):
    
    """
    Parameters
    ----------
    rdmol : rdkit.Chem.rdchem.Mol
            Rdkit molecule object. 
    mdata : dict
            dictionary of optional metadata for the molecule, like a filepath

    """
    self.rdmol = rdmol
    self.mdata = mdata
    
  
  @property
  def atom_graph(self):
    if not hasattr(self,"_atom_graph"):
      # build without features
      self.build_atom_graph(atom_featurizer=None,keep_xyz=True)
    return self._atom_graph
  
  @property
  def model(self):
    # try to return the cctbx model obejct
    if not hasattr(self,"_model"):
      try:
        for key,value in self.mdata.items():
          if isinstance(value,model_manager):
            self._model = value
      except:
        self._model = None
    return self._model
  
  def build_atom_graph(self,atom_featurizer=None,keep_xyz=True):
    """
    Build an all-atom graph, optionally featurized with an initial atom featurizer
    """
    rdmol = self.rdmol
    
    # build graph
    bonds = list(self.rdmol.GetBonds())
    bonds_begin_idxs = [bond.GetBeginAtomIdx() for bond in bonds]
    bonds_end_idxs = [bond.GetEndAtomIdx() for bond in bonds]
    bonds_types = [bond.GetBondType().real for bond in bonds]
    data = np.array(list(zip(bonds_begin_idxs,bonds_end_idxs)))
    data = np.vstack([data,np.flip(data,axis=1)]) # dgl requires two edges for undirected graph
    g = dgl.graph((data[:,0],data[:,1]),num_nodes=rdmol.GetNumAtoms())
    
    if atom_featurizer is not None:
      # add features
      features = np.vstack([atom_featurizer(atom) for atom in rdmol.GetAtoms()])
      g.ndata["h0"] = torch.from_numpy(features) # set initial representation
    
    # set xyz coords
    if keep_xyz:
      if len(rdmol.GetConformers())>0:
        conf = rdmol.GetConformer()
        pos = conf.GetPositions()
        g.ndata["pos"] = torch.tensor(pos,dtype=torch.float32)
      
    
    self._atom_graph = g
    
  def save(self,filepath):
    filepath = Path(filepath)
    self.mdata["binary_path"] = str(filepath)
    torch.save(self,str(filepath))
    
  def __repr__(self):
    rep = ""
    just = 20
    rep+= "\n"+object.__repr__(self)+"\n"

    # check molecule object
    rep+="\n Attributes:"
    if self.rdmol != None:
      value = object.__repr__(self.rdmol)
    else:
      value = None
    rep += "\n  rdmol"+"".join([" "]*(just+4))+": "+value

    if self.model != None:
      value = object.__repr__(self.model)
    else:
      value = None
    rep+= "\n  "+"model"+"".join([" "]*(just+4))+": "+value

    # graph
    rep+="\n  atom_graph"+"".join([" "]*(just-1))+": "+str(object.__repr__(self.atom_graph))


    # meta data
    rep+="\n"
    for key,value in self.mdata.items():
      try:
        if isinstance(value,model_manager):
          value = str(object.__repr__(value))
      except:
        value = str(value.__repr__())
      rep+="\n"
      rep+= "  mdata['"+str(key)+"']"+"".join([" "]*(just-len(key)))+": "+value
    return rep
    
class MolGraph:
    
  def __init__(self, rdmol,filepath=None,default_mol_type="mol3d_noH",levels=["n1","n2","n3","n4"],canonical_conf=False):
      self.filepath = filepath
      self.default_mol_type = default_mol_type    
      # obtain a 2d and 3d version of the molecule     
      if len(rdmol.GetConformers())>0:
        self.conformer_from_input = True
        self.mol3d = rdmol
        self.mol2d = Chem.Mol(self.mol3d)
        _ = AllChem.Compute2DCoords(self.mol2d)
      else:
        self.conformer_from_input = False
        self.mol2d = rdmol
        self.mol3d = Chem.AddHs(self.mol2d)
        _ = AllChem.EmbedMolecule(self.mol3d,randomSeed=0xf00d) 
        
      # make no-hydrogen versions of the 2d molecule
      self.mol2d_noH = Chem.RemoveHs(self.mol2d)
      _ = AllChem.Compute2DCoords(self.mol2d_noH)
      self.mol3d_noH = Chem.RemoveHs(self.mol3d)
      
      self.rdmol = getattr(self,self.default_mol_type)
      #self.offmol = Molecule.from_rdkit(self.rdmol,hydrogens_are_explicit=True)
      # make dgl graphs

      self.homograph = build_homograph(self.rdmol)
      #self.homograph = read_homogeneous_graph.from_openff_toolkit_mol(self.offmol)
      self.heterograph = build_heterograph_from_homo_mol(self.homograph,self.rdmol,levels)

      #self._rdmols = [self.mol3d,self.mol3d_noH,self.mol2d,self.mol2d_noH]
      for moltype in ["mol3d","mol3d_noH","mol2d","mol2d_noH"]:
        if moltype != default_mol_type:
          delattr(self,moltype)
      
      # set conformer properties on graph
      conf = self.rdmol.GetConformer()
      if canonical_conf:
        Chem.rdMolTransforms.CanonicalizeConformer(conf)
      if conf is not None:
        # molecule
        conf = self.rdmol.GetConformer()
        # pos = conf.GetPositions()
        # com = pos.mean(axis=0)
        # pca = PCA(n_components=3)
        # _ = pca.fit(pos)
        # rot,rmsd = Rotation.align_vectors(np.eye(3),pca.components_)
        # quat = rot.as_quat()
        # self.heterograph.nodes["g"].data["com_ref"] = torch.tensor(com[np.newaxis,:])
        # self.heterograph.nodes["g"].data["quat_ref"] = torch.tensor(quat[np.newaxis,:])

        
        #atoms
        self.heterograph.nodes["n1"].data["h0"]= self.heterograph.nodes["n1"].data["h0"].detach().clone().type(torch.float32)
        self.heterograph.nodes["n1"].data["xyz_ref"]= torch.tensor(np.array(conf.GetPositions()),dtype=torch.float32)
        
        #labelize bonds and angles
        #bonds
        if "n2" in levels:
          bond_lengths = []
          for idx1,idx2 in self.heterograph.nodes["n2"].data["idxs"]: # this will compute bond lengths with redundancy, could just do half and copy...
            bond_length = Chem.rdMolTransforms.GetBondLength(conf,int(idx1),int(idx2))
            bond_lengths.append(bond_length)
          bond_lengths = np.array(bond_lengths)[:,np.newaxis] 
          self.heterograph.nodes["n2"].data["eq_ref"] = torch.tensor(bond_lengths,dtype=torch.float32)

        # angles
        if "n3" in levels:
          angles_rad = []
          for idx1,idx2,idx3 in self.heterograph.nodes["n3"].data["idxs"]:
            angle_rad = Chem.rdMolTransforms.GetAngleRad(conf,int(idx1),int(idx2),int(idx3))
            angles_rad.append(angle_rad)
          angles_rad = np.array(angles_rad)[:,np.newaxis]
          self.heterograph.nodes["n3"].data["eq_ref"] = torch.tensor(angles_rad,dtype=torch.float32)
        
        # torsions
#         tor_degs = []
#         for idx0,idx1,idx2,idx3 in self.heterograph.nodes["n4"].data["idxs"]: 
#           tor_deg = Chem.rdMolTransforms.GetDihedralDeg(conf,int(idx0),int(idx1),int(idx2),int(idx3))
#           tor_degs.append(tor_deg)

#         tor_degs = np.array(tor_degs)
#         deg = np.full((len(tor_degs),3),0,dtype=np.float32)
#         deg[:,0] = tor_degs
#         rot = Rotation.from_euler('xyz', deg, degrees=True)
#         quat = rot.as_quat()

#         # quath is in same hemisphere
#         quath = quat.copy()
#         mask = quath[:,0]<0
#         quath[mask,0]*=-1
        
#         self.heterograph.nodes["n4"].data["deg_ref"] = torch.tensor(deg[:,0],dtype=torch.float32)
#         self.heterograph.nodes["n4"].data["q_ref"] = torch.tensor(quat[:,[0,3]],dtype=torch.float32)
#         self.heterograph.nodes["n4"].data["qh_ref"] = torch.tensor(quath[:,[0,3]],dtype=torch.float32)
#         if "n4" in levels:
#           tor_degs = []
#           for idx0,idx1,idx2,idx3 in self.heterograph.nodes["n4"].data["idxs"]: 
#             tor_deg = Chem.rdMolTransforms.GetDihedralDeg(conf,int(idx0),int(idx1),int(idx2),int(idx3))
#             tor_degs.append(tor_deg)

#           deg = np.array(tor_degs)
#           #deg = np.abs(deg)
#           angles = np.radians(deg)
#           cmplx = np.cos(angles)+1j*np.sin(angles)
#           self.heterograph.nodes["n4"].data["deg_ref"] = torch.tensor(deg,dtype=torch.float32)
#           self.heterograph.nodes["n4"].data["cmplx_ref"] = torch.tensor(cmplx,dtype=torch.cfloat)

#           deg = np.abs(deg)
#           angles = np.radians(deg)
#           cmplx = np.cos(angles)+1j*np.sin(angles)
#           self.heterograph.nodes["n4"].data["deg_ref_hem"] = torch.tensor(deg,dtype=torch.float32)
#           self.heterograph.nodes["n4"].data["cmplx_ref_hem"] = torch.tensor(cmplx,dtype=torch.cfloat)
#           #self.heterograph.nodes["n4"].data["real_ref"] = torch.tensor(cmplx.real,dtype=torch.float32)
#           #self.heterograph.nodes["n4"].data["imag_ref"] = torch.tensor(cmplx.imag,dtype=torch.float32)