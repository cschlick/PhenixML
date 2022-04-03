from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS


import numpy as np
import warnings
import random
import tqdm

from rdkit import RDLogger

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from iotbx.data_manager import DataManager
    from cctbx.array_family import flex

from phenixml.utils.rdkit_utils import mol_from_smiles

class MolContainer:
    """
    The MolContainer class is meant to hold multiple molecular data structures.

    Fragmentation of a MolContainer object into fragments is the first step in a 
    regression/classification task. Because the container holds multiple data structures, 
    fragmenters/Featurizers/labelers can be written to use any datatype present (ie, cctbx, rdkit, etc)

    In addition to multiple data structures, a MolContainer is indented to manage multiple forms
    of the same molecule (ie, +/- explicit hydrogens)
    """

    suffixes_supported = [".mol",".mol2",".pdb",".mmcif",".cif"]
    supported_source_types = ["rdkit","cctbx"]
    
    @classmethod
    def from_folder(cls,
                    folder,
                    max_files=None,
                    suffix=None,
                    random_fraction=1.0,
                    removeHs=False,
                    disable_progress=False,
                    skip_failures=True,
                    **kwargs):
      
      RDLogger.DisableLog('rdApp.*') # suppress rdkit output

      folder = Path(folder)
      files = [file for file in folder.glob("**/*") if file.suffix in cls.suffixes_supported]
      if 0 <random_fraction <1:
        files = random.sample(files,k=int(len(files)*random_fraction))
      if suffix is not None:
        files = [file for file in files if file.suffix == suffix or file.suffix.strip(".")==suffix]
      
      if max_files!=None:
          files = files[:max_files]

      return cls.from_file_list(files,removeHs=removeHs,skip_failures=skip_failures,**kwargs)
    
    @classmethod
    def from_file_list(cls,
                       filenames,
                       disable_progress=False,
                       skip_failures=True,
                       **kwargs):
      # For now, always skip failures
      
      RDLogger.DisableLog('rdApp.*') # suppress rdkit output
      containers = []
      for filename in tqdm.tqdm(filenames,disable=disable_progress):
        container = cls.from_file_name(filename)
        if (container is not None and 
            container.rdkit_mol is not None and  
            container.rdkit_mol_noH is not None):
          # this should not be necessary. Eventually the class should facilitate using either +/- Hs
          if "removeHs" in kwargs and kwargs["removeHs"]==True:
             container = cls.from_rdkit(container.rdkit_mol_noH,
                                      filename=str(container.filepath))
          
          if (container is not None and 
            container.rdkit_mol is not None and
            container.rdkit_mol_noH is not None):
            containers.append(container)

      
      
      return containers
            
    
    @classmethod
    def from_smiles(cls,smiles_string,embed3d=False,addHs=False,removeHs=False):
        rdkit_mol = mol_from_smiles(smiles_string,embed3d=embed3d,addHs=addHs,removeHs=removeHs)
        return cls(rdkit_mol,source_object_type="rdkit")
    
    @classmethod
    def from_rdkit(cls,rdkit_mol,filename=None):
        return cls(rdkit_mol,source_object_type="rdkit",filepath=filename)
    
    @classmethod
    def from_cctbx_model(cls,cctbx_model,filename=None):
        return cls(cctbx_model,source_object_type="cctbx",filepath=filename)
    
    @classmethod
    def from_file_name(cls,filename,removeHs=False):
        filepath = Path(filename)
        
        # protein models use cctbx
        if len(set([".cif",".mmcif",".pdb"]).intersection(filepath.suffixes)) >0:       
            dm = DataManager()
            dm.process_model_file(str(filepath))
            cctbx_model = dm.get_model()
            return cls(cctbx_model,source_object_type="cctbx",filepath=filepath)
        
        # small molecules use rdkit by default
        elif ".mol" in filepath.suffixes:
            rdkit_mol = Chem.MolFromMolFile(str(filepath),removeHs=removeHs)
            if rdkit_mol is None:
              return None
            else:
              return cls(rdkit_mol,source_object_type="rdkit",filepath=filepath,_is_macromolecule=False)
        
        elif ".mol2" in filepath.suffixes:
            rdkit_mol = Chem.MolFromMol2File(str(filepath),removeHs=removeHs)
            return cls(rdkit_mol,source_object_type="rdkit",filepath=filepath,_is_macromolecule=False)
        else:
            raise ValueError("Unrecognized file extension(s):",filepath.suffixes)
        
    
    
    def __init__(self,source_object,
                 source_object_type="rdkit",
                 filepath=None,
                 **kwargs):
        assert source_object_type in self.supported_source_types
        self.source_object = source_object
        self.source_object_type = source_object_type
        if isinstance(filepath,str):
          filepath = Path(filepath)
        self.filepath = filepath
        if source_object_type == "rdkit":
            self._rdkit_mol = source_object
        elif source_object_type == "cctbx":
            self._cctbx_model = source_object
            self._cctbx_model.add_crystal_symmetry_if_necessary()
        for kw,arg in kwargs.items():
            setattr(self,kw,arg)
    
    def __len__(self):
        return self.rdkit_mol.GetNumAtoms()
    
    @property
    def rdkit_mol(self):
        if not hasattr(self,"_rdkit_mol"):
            self._rdkit_mol = self.to_rdkit()
        return self._rdkit_mol
    
    def to_rdkit(self):
        if self.source_object_type=="rdkit":
            return self.source_object
        else:
            rdkit_mol = Chem.MolFromPDBBlock(self.cctbx_model.model_as_pdb(),removeHs=False)
            return rdkit_mol
    
    @property
    def rdkit_mol_2d(self):
        if not hasattr(self,"_rdkit_mol_2d"):
            conformer_from_input = len(self.rdkit_mol.GetConformers())>0
            if conformer_from_input: # need to make 2d version
                mol2d = Chem.Mol(self.rdkit_mol)
                _ = AllChem.Compute2DCoords(mol2d)
                self._rdkit_mol_2d = mol2d
            else:
                self._rdkit_mol_2d = self.rdkit_mol
        
        return self._rdkit_mol_2d
    
    @property
    def rdkit_mol_3d(self):
        mol = self.rdkit_mol
        conformer_from_input = len(mol.GetConformers())>0
        if not conformer_from_input: 
            # need to make 3d version Careful! Not experimental
            mol3d = Chem.AddHs(mol)
            _ = AllChem.EmbedMolecule(mol3d,randomSeed=0xf00d) 
            return mol3d
        else:
          return mol
    
    @property
    def rdkit_mol_noH(self):
        if not hasattr(self,"_rdkit_mol_noH"):
            self._rdkit_mol_noH = Chem.RemoveHs(self.rdkit_mol)
        return self._rdkit_mol_noH
    
    @property
    def noH_match_dict(self):
        """
        Return a dict mapping atomIdx(noH) -> atomIdx(withH)
        
        Useful for converting from fragment atom indices when
        drawing without hydrogens
        """
        if not hasattr(self,"_noH_match_dict"):
            mol_list = [self.rdkit_mol_2d,self.rdkit_mol_noH]
            mol_list = [Chem.Mol(mol) for mol in mol_list]
            mcs_SMARTS = rdFMCS.FindMCS(mol_list)
            smarts_mol = Chem.MolFromSmarts(mcs_SMARTS.smartsString)
            match_list = [x.GetSubstructMatch(smarts_mol) for x in mol_list]
            match_list = list(zip(match_list[0],match_list[1]))
            self._noH_match_dict = {b:a for (a,b) in match_list}
        return self._noH_match_dict
    
    @property
    def cctbx_model(self):
        if not hasattr(self,"_cctbx_model"):
            self._cctbx_model = self.to_cctbx_model()
        return self._cctbx_model
    
    def to_cctbx_model(self):
        if self.source_object_type=="cctbx":
            return self.source_object
        else:
            dm = DataManager()
            dm.process_model_str("cctbx_model",Chem.MolToPDBBlock(self.rdkit_mol))
            cctbx_model = dm.get_model("cctbx_model")
            cctbx_model.add_crystal_symmetry_if_necessary()
            return cctbx_model
    
    # These two properties below: xyz (cartesian coordinates) and elements (array of element string) 
    # are the only two places that "data" is really stored on the model container and 
    # thus duplicated. It might be better to not duplicate it, but if you need the numpy form
    # repeatedly than retrieving it each time can be slow.
    
    @property
    def xyz(self):
      if not hasattr(self,"_xyz"):
        if hasattr(self,"_cctbx_model"):
            self._xyz = self.cctbx_model.get_sites_cart().as_numpy_array()
        elif hasattr(self,"_rdkit_mol"):
            conf = self.rdkit_mol.GetConformer()
            self._xyz = conf.GetPositions()
      return self._xyz
    
    @xyz.setter
    def xyz(self,value):
      if hasattr(self,"_xyz"):
        del self._xyz
        self._xyz = value # NEED TO UPDATE THE DATASTRUCTURES

    
    @property
    def elements(self):
      if not hasattr(self,"_elements"):
        if hasattr(self,"_cctbx_model"):
            atoms = self.cctbx_model.get_atoms()
            self._elements =  np.array([e.strip() for e in atoms.extract_element()])
        elif hasattr(self,"_rdkit_mol"):
            self._elements = np.array([atom.GetSymbol() for atom in self.rdkit_mol.GetAtoms()])
      return self._elements
        
    
    @property
    def is_macromolecule(self):
        if not hasattr(self,"_is_macromolecule"):
            if self.cctbx_model.selection("protein or nucleotide").count(True) >0:
                self._is_macromolecule = True
            else:
                self._is_macromolecule = False
        return self._is_macromolecule

    @property
    def has_small_molecules(self):
        if self.is_macromolecule:
            if not hasattr(self,"_has_small_molecules"):
                ligand_sel = container.cctbx_model.selection("not protein and not nucleotide and not water").as_numpy_array()
                ligand_sel = np.where(ligand_sel==True)[0]
                if len(ligand_sel)>0:
                    self._small_molecule_selection = ligand_sel
                    self._has_small_molecules= True
                else:
                    self._has_small_molecules = True
        else:
            self._has_small_molecules = True
        return self._has_small_molecules

    @property
    def full_fragment(self):
        if not hasattr(self,"_full_fragment"):
            self._full_fragment = Fragment(self,np.arange(len(self)))
        return self._full_fragment
                                           
    
    def show(self,**kwargs):
        from phenixml.visualization.fragment_display import FragmentDisplay
        display = FragmentDisplay()
        return display(self,**kwargs)
            
#############################################################
#### Fragment class
#############################################################

class Fragment:
    """
    A fragment is a selection on a MolContainer object, which maintains a 
    reference to the container. Fragments are usually generated from a 
    MolContainer object useing a Fragmenter object.

    A fragment is the fundamental object to pass to:
      1. Featurizers
      2. Labelers

    It is also useful for visualization purposes.
    """
    def __init__(self,mol_container,atom_selection=None,string_selection=None):
        self.mol_container = mol_container
        self.atom_selection = np.array(atom_selection)
        self.string_selection = string_selection
        if atom_selection is not None and string_selection is not None:
            raise ValueError("Provide either atom selection or string selection")
        if string_selection is not None:
          self.atom_selection = self.mol_container.cctbx_model.selection(
            string_selection).as_numpy_array().nonzero()[0]
        if atom_selection is None and string_selection is None:
            raise ValueError("fragment does not have associated selection")
        
    def __len__(self):
        return len(self.atom_selection)
    
    def extract(self):
        
        # use cctbx
        if self.atom_selection is not None:
            model = self.mol_container.cctbx_model
            sel = np.zeros(model.get_number_of_atoms(),dtype=bool)
            sel[self.atom_selection] = True
            sel = flex.bool(sel)
            new_model = self.mol_container.cctbx_model.select(sel)
            return MolContainer.from_cctbx_model(new_model)
        
        elif self.string_selection is not None:
            sel =  self.mol_container.cctbx_model.select(sel)
            new_model = self.mol_container.cctbx_model.selection(sel)
            return MolContainer.from_cctbx_model(new_model)
        
    @property
    def rdkit_mol(self):
        return self.mol_container.rdkit_mol
    @property
    def rdkit_mol_2d(self):
        return self.mol_container.rdkit_mol_2d
    @property
    def rdkit_mol_3d(self):
        return self.mol_container.rdkit_mol_3d
    @property
    def rdkit_mol_noH(self):
        return self.mol_container.rdkit_mol_noH
      
    @property
    def cctbx_model(self):
        return self.mol_container.cctbx_model
        
    @property
    def xyz(self):
        return self.mol_container.xyz[self.atom_selection]
    @property
    def elements(self):
        return self.mol_container.elements[self.atom_selection]
      
    @property # alias
    def atom_indices(self):
      return self.atom_selection
    
    # these idx properties are for rdkit, which doesn't work well with numpy
    @property 
    def atom_idxs(self):
      return [int(i) for i in self.atom_selection]
    
    @property
    def bond_idxs(self):
      return self.calc_bond_idxs(self.rdkit_mol,self.atom_selection)
    
    @staticmethod
    def calc_bond_idxs(mol,atom_idxs):
        atoms = atom_idxs
        bonds = []
        bond_dict = {bond.GetIdx():[bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()] for bond in mol.GetBonds()}
        for key,value in bond_dict.items():
            if set(value).issubset(atoms):
                bonds.append(key)
        return [int(i) for i in bonds]

    
    
    def show(self,**kwargs):
        from phenixml.visualization.fragment_display import FragmentDisplay
        display = FragmentDisplay()
        return display(self,**kwargs)
    
    def grow(self,radius=1):
        """
        Grow a fragment by N bonds in any/all directions
        Returns a new fragment
        """
        in_atoms = list(self.atom_selection)
        mol = self.mol_container.rdkit_mol
        edge_atoms = []
        for i in range(radius):
          if i>0:
            in_atoms+=edge_atoms
            edge_atoms = []
          for atomidx in in_atoms:
            atom = mol.GetAtomWithIdx(int(atomidx))
            nbrs = atom.GetNeighbors()
            for nbr in nbrs:
              nbridx = nbr.GetIdx()
              if nbridx not in in_atoms:
                if nbridx not in edge_atoms:
                  edge_atoms.append(nbridx)
        new_selection = sorted(edge_atoms+in_atoms)
        return Fragment(self.mol_container,new_selection)