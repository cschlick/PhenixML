from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import numpy as np
import dgl
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

from molgraph.build_graphs import build_heterograph_from_homo_mol, build_homograph


class MolGraphProtein:
  def __init__(self, rdmol,filepath=None,default_mol_type="mol3d_noH",input_mol_type="mol3d_noH"):
    self.rdmol = rdmol
    
    

class MolGraph:
    
  def __init__(self, rdmol,filepath=None,default_mol_type="mol3d_noH",levels=["n1","n2","n3","n4"],canonical_conf=True):
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
        pos = conf.GetPositions()
        com = pos.mean(axis=0)
        pca = PCA(n_components=3)
        _ = pca.fit(pos)
        rot,rmsd = Rotation.align_vectors(np.eye(3),pca.components_)
        quat = rot.as_quat()
        self.heterograph.nodes["g"].data["com_ref"] = torch.tensor(com[np.newaxis,:])
        self.heterograph.nodes["g"].data["quat_ref"] = torch.tensor(quat[np.newaxis,:])

        
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
        if "n4" in levels:
          tor_degs = []
          for idx0,idx1,idx2,idx3 in self.heterograph.nodes["n4"].data["idxs"]: 
            tor_deg = Chem.rdMolTransforms.GetDihedralDeg(conf,int(idx0),int(idx1),int(idx2),int(idx3))
            tor_degs.append(tor_deg)

          deg = np.array(tor_degs)
          #deg = np.abs(deg)
          angles = np.radians(deg)
          cmplx = np.cos(angles)+1j*np.sin(angles)
          self.heterograph.nodes["n4"].data["deg_ref"] = torch.tensor(deg,dtype=torch.float32)
          self.heterograph.nodes["n4"].data["cmplx_ref"] = torch.tensor(cmplx,dtype=torch.cfloat)

          deg = np.abs(deg)
          angles = np.radians(deg)
          cmplx = np.cos(angles)+1j*np.sin(angles)
          self.heterograph.nodes["n4"].data["deg_ref_hem"] = torch.tensor(deg,dtype=torch.float32)
          self.heterograph.nodes["n4"].data["cmplx_ref_hem"] = torch.tensor(cmplx,dtype=torch.cfloat)
          #self.heterograph.nodes["n4"].data["real_ref"] = torch.tensor(cmplx.real,dtype=torch.float32)
          #self.heterograph.nodes["n4"].data["imag_ref"] = torch.tensor(cmplx.imag,dtype=torch.float32)