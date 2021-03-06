from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem import rdFMCS
import rdkit

import io
import numpy as np
import itertools
import matplotlib.pyplot as plt
import py3Dmol

from phenixml.fragments.fragments import MolContainer, Fragment
from phenixml.fragmentation.fragmenters import SmallMoleculeFragmenter


    # Issues:
#     1. 
#     There seems to be an issue where indices passed in 3d don't match.
#     Hopefully this is an issue with py3dmol, not with cctbx/rdkit.
    
#     To reproduce: Open container with a ligand, try to display the 3d
#     model where the ligand is highlighted.
    
#     2. Issues with hydrogens. Radicals (ie MTN) have hydrogens displayed
#     implicitly, even when they were not there in the model before 
#     removing hydrogens for visualization
    


class FragmentDisplay:
    """
    A class to display fragments and containers using rdkit and py3dmol

    """

    def __call__(self,obj,**kwargs):
        if isinstance(obj,MolContainer):
            return self.show_container(obj,**kwargs)
        elif isinstance(obj,Fragment):
            return self.show_fragment(obj,**kwargs)
        elif isinstance(obj,list):
            if isinstance(obj[0],Fragment):
                return self.grid_fragments(obj,**kwargs)
        elif isinstance(obj,rdkit.Chem.rdchem.Mol):
          container = MolContainer.from_rdkit(obj)
          return self.show_container(container,**kwargs)
        else:
            raise ValueError("Call using a MolContainer or Fragment object")
    
    @staticmethod
    def show_mol_2d(mol,
                    size=(300,300),
                    highlightAtoms=[],
                    highlightBonds=[]):

        return Draw.MolToImage(mol,size=size,highlightAtoms=highlightAtoms,highlightBonds=highlightBonds)



    @staticmethod
    def show_mol_text_3d(mol_text,
                    size=(600,400),
                    style="stick",
                    surface=False,
                    opacity=0.5,
                    cartoon_color="#34eb8f",
                    cartoon_omit_atoms = [],
                    highlight_atoms=[],
                    highlight_style={'stick':{'color': "#f542f5"},'sphere':{'radius': 0.5,"color":"#f542f5"}}):
        """Draw molecule in 3D

        Args:
        ----
            mol: rdMol, molecule to show
            size: tuple(int, int), canvas size
            style: str, type of drawing molecule
                   style can be 'line', 'stick', 'sphere', 'cartoon'
            surface, bool, display SAS
            opacity, float, opacity of surface, range 0.0-1.0
        Return:
        ----
            viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
        """

        assert style in ('line', 'stick', 'sphere', 'cartoon')

        viewer = py3Dmol.view(width=size[0], height=size[1])
        viewer.addModel(mol_text)
        if style=="cartoon":
            viewer.setStyle({style:{"color":cartoon_color}})
        else:
            viewer.setStyle({style:{}})
        if len(highlight_atoms)>0: 
            viewer.setStyle({'serial':highlight_atoms},highlight_style)

        if style == "cartoon" and len(cartoon_omit_atoms)>0:

            viewer.setStyle({'serial':cartoon_omit_atoms},{"stick":{}})

        if surface:
            viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
        
        viewer.setHoverable({},True,'''function(atom,viewer,event,container) {
                   if(!atom.label) {
                    atom.label = viewer.addLabel(atom.atom+":"+atom.index,{position: atom, backgroundColor: 'mintcream', fontColor:'black'});
                   }}''',
               '''function(atom,viewer) { 
                   if(atom.label) {
                    viewer.removeLabel(atom.label);
                    delete atom.label;
                   }
                }''')
        viewer.zoomTo()
        return viewer

    @staticmethod
    def show_fragment(fragment,show3d=False,highlight=True,size=None,extract=False,hideHs=True):
        
        # the below is a hack to address a bug. It should be removed if possible    
        if fragment.mol_container.is_macromolecule and len(fragment.mol_container)>100:
            extract=True
            show3d=True
            
        
        # don't highlight if fragment is same size as container
        if len(fragment) == len(fragment.mol_container):
            highlight = False
        
        if extract:
            container = fragment.extract()
        else:
            container = fragment.mol_container
            
        mol = container.rdkit_mol_2d
        highlightAtoms = []
        highlightBonds = []
  
        
        if highlight and not extract:
            highlightAtoms = fragment.atom_selection
            if not hideHs:
              highlightBonds = Fragment.calc_bond_idxs(fragment.rdkit_mol,highlightAtoms)
            else:
              mol = container.rdkit_mol_noH
              highlightAtoms = list(FragmentDisplay.convert_idxs_HnoH(fragment))
              highlightBonds = list(Fragment.calc_bond_idxs(fragment.rdkit_mol_noH,highlightAtoms))
            
        highlightAtoms = [int(i) for i in highlightAtoms]
        highlightBonds = [int(i) for i in highlightBonds]

        if not show3d:
            if size is None:
              size = (600,600)
            mol = Chem.Mol(mol)
            _ = AllChem.Compute2DCoords(mol)
            return FragmentDisplay.show_mol_2d(mol,
                                               size=size,highlightAtoms=list(highlightAtoms),
                                               highlightBonds=highlightBonds)
        else:
            if size is None:
              size = (600,600)
            if container.is_macromolecule:
                mol_text = container.cctbx_model.model_as_pdb()
                return FragmentDisplay.show_mol_text_3d(mol_text,style="stick",
                                                        size=size,highlight_atoms=list(highlightAtoms))
            else:
                mol_text = Chem.MolToMolBlock(container.rdkit_mol_3d)
                return FragmentDisplay.show_mol_text_3d(mol_text,
                                                        size=size,
                                                        highlight_atoms=list(highlightAtoms))

    @staticmethod
    def show_container(container,show3d=False,show_3d=False,**kwargs):

        if container.is_macromolecule:
            ligand_fragments = SmallMoleculeFragmenter()(container)
            ligand_idxs = list(itertools.chain.from_iterable([frag.atom_selection for frag in ligand_fragments]))
            ligand_idxs = [int(i+1) for i in ligand_idxs]
            return FragmentDisplay.show_mol_text_3d(
              container.cctbx_model.model_as_pdb(),
              style="cartoon",cartoon_omit_atoms=ligand_idxs,*kwargs)

        elif not show3d:
            return FragmentDisplay.show_mol_2d(container.rdkit_mol_2d)
        else:
            return FragmentDisplay.show_mol_text_3d(Chem.MolToPDBBlock(container.rdkit_mol_3d),style="stick",**kwargs)


    @staticmethod
    def grid_fragments(fragments,nmax=6,subImgSize=(300,300),showH=False,**kwargs):
        fragments = fragments[:nmax]
        if showH:
          mols = [fragment.mol_container.rdkit_mol_2d for fragment in fragments]
        else:
          mols = [fragment.mol_container.rdkit_mol_noH for fragment in fragments]

        # bugs below, this doesn't work passed as a kwarg, sometimes mcs fails
        if "highlightAtomLists" not in kwargs:
          if not showH:
            highlightAtomLists=[
              list(FragmentDisplay.convert_idxs_HnoH(fragment))
              for fragment in fragments if len(fragment) < len(fragment.mol_container)]
          else:
            highlightAtomLists=[
              list(fragment.atom_idxs) for fragment in fragments if len(fragment) < len(fragment.mol_container)]

        if "highlightBondLists" not in kwargs:
          if not showH:
            highlightBondLists=[
              list(Fragment.calc_bond_idxs(fragments[i].rdkit_mol_noH,idxs))
              for i,idxs in enumerate(highlightAtomLists)]
          else:
            highlightBondLists=[
              list(Fragment.calc_bond_idxs(fragments[i].rdkit_mol_2d,idxs))
              for i,idxs in enumerate(highlightAtomLists)]
        mols = [Chem.Mol(mol) for mol in mols]
        for mol in mols:
          _ = AllChem.Compute2DCoords(mol)
          
        return MolsToGridImage(mols,
                    highlightAtomLists=highlightAtomLists,
                    highlightBondLists=highlightBondLists,subImgSize=subImgSize,**kwargs)
    
    
    @staticmethod
    def convert_idxs_HnoH(fragment):
      
      a,b = fragment.rdkit_mol, fragment.rdkit_mol_noH
      mol_list= [a,b]
      mcs_SMARTS = rdFMCS.FindMCS(mol_list,matchValences=True)
      smarts_mol = Chem.MolFromSmarts(mcs_SMARTS.smartsString)
      match_list = [x.GetSubstructMatch(smarts_mol) for x in mol_list]
      a_idxs,b_idxs = match_list
      a_to_b = {a_idx:b_idx for a_idx,b_idx in zip(a_idxs,b_idxs)}
      idxs = []
      for i in fragment.atom_selection:
        if i in a_to_b:
          idxs.append(a_to_b[i])
      if len(idxs)==len(fragment):
        return [int(i) for i in idxs]
      else:
        return []
      
      
    @staticmethod
    def compare_mcs(fragments):
        pass