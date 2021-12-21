from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import numpy as np
import dgl

from molgraph.build_graphs import build_heterograph_from_homo_mol, build_homograph

class MolGraph:
    
  def __init__(self, rdmol,filepath=None,default_mol_type="mol3d_noH"):
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
      self.heterograph = build_heterograph_from_homo_mol(self.homograph,self.rdmol)

      self._rdmols = [self.mol3d,self.mol3d_noH,self.mol2d,self.mol2d_noH]
      
      # set conformer properties on graph
      conf = self.rdmol.GetConformer()
      if conf is not None:
        
        #atoms
        self.heterograph.nodes["n1"].data["xyz"]= torch.tensor(np.array(conf.GetPositions()),dtype=torch.float32)
        self.heterograph.nodes["n1"].data["h0"]= self.heterograph.nodes["n1"].data["h0"].detach().clone().type(torch.float32)

        conf = self.rdmol.GetConformer()
        #labelize bonds and angles
        #bonds
        bond_lengths = []
        for idx1,idx2 in self.heterograph.nodes["n2"].data["idxs"]: # this will compute bond lengths with redundancy, could just do half and copy...
          bond_length = Chem.rdMolTransforms.GetBondLength(conf,int(idx1),int(idx2))
          bond_lengths.append(bond_length)
        bond_lengths = np.array(bond_lengths)[:,np.newaxis] 
        self.heterograph.nodes["n2"].data["eq_ref"] = torch.tensor(bond_lengths,dtype=torch.float32)

        # angles
        angles_rad = []
        for idx1,idx2,idx3 in self.heterograph.nodes["n3"].data["idxs"]: # this will compute bond lengths with redundancy, could just do half and copy...
          angle_rad = Chem.rdMolTransforms.GetAngleRad(conf,int(idx1),int(idx2),int(idx3))
          angles_rad.append(angle_rad)
        angles_rad = np.array(angles_rad)[:,np.newaxis]
        self.heterograph.nodes["n3"].data["eq_ref"] = torch.tensor(angles_rad,dtype=torch.float32)
        
        

