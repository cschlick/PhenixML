import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter
from itertools import combinations
import copy
import io
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display

from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw


from phenixml.utils.rdkit_utils import remove_H

class Fragment:
  """
  A class to hold a molecular fragment.
  
  
  """
  def __init__(self,rdmol,atom_indices=[],parent=None):
    """
    :param rdmol: rdkit molecule
    :param atom_indices: the atom indices of the fragment
    :param parent: Optionally store a parent fragment
    """
    self.rdmol = rdmol
    self.atom_indices = atom_indices
    if len(atom_indices)==0:
      self.atom_indices = list(range(0,rdmol.GetNumAtoms()))
    
    if len(self.rdmol.GetConformers())>0:
      self.has_conformer = True
    
    self.parent = parent
    self.properties = {}
    self._figsize_default = (10,10)
    
  def __len__(self):
    return len(self.atom_indices)

# broken, displays 2 images
#   def _repr_png_(self):
#     mol_bitmap = self.show(interactive=False)

#     fig, ax = plt.subplots(1,1,figsize=self._figsize_default)
#     ax.imshow(mol_bitmap)
#     ax.axes.spines["left"].set_visible(False)
#     ax.axes.spines["top"].set_visible(False)
#     ax.axes.spines["right"].set_visible(False)
#     ax.axes.spines["bottom"].set_visible(False)
#     ax.xaxis.set_visible(False)
#     ax.yaxis.set_visible(False)
    
#     return display(fig)
  
#   @property
#   def rdmol(self):
#     #return copy.deepcopy(self._rdmol)
#     return self._rdmol
  
#   @rdmol.setter
#   def rdmol(self,value):
#     self._rdmol = value
  
  @property
  def atoms(self):
    return [self.rdmol.GetAtomWithIdx(i) for i in self.atom_indices]
  
  @property
  def atom_symbols(self):
    return [atom.GetSymbol() for atom in self.atoms]
  @property
  def atom_numbers(self):
    return [atom.GetAtomicNum() for atom in self.atoms]
  
  @property
  def xyz_mol(self):
    conf = self.rdmol.GetConformers()[0]
    coords = np.vstack([conf.GetAtomPosition(atom.GetIdx()) for atom in self.rdmol.GetAtoms()])
    return coords
  
  @property
  def xyz_fragment(self):
    conf = self.rdmol.GetConformers()[0]
    coords = np.vstack([conf.GetAtomPosition(idx) for idx in self.atom_indices])
    return coords
  
  @property
  def bonds(self):
    return [self.rdmol.GetBondWtihIdx(i) for i in self.bond_indices]
  
  
  @property
  def bond_dict(self):
    if not hasattr(self,"_bond_dict"):
      self._bond_dict = {bond.GetIdx():[bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()] for bond in self.rdmol.GetBonds()}
    return self._bond_dict
  
  @property
  def bond_indices(self):
    if not hasattr(self,"_bond_indices"):
      atom_set = set(self.atom_indices)
      self._bond_indices = [key for key,value in self.bond_dict.items() if set(value).issubset(atom_set)]
    return self._bond_indices
  
  @property
  def fragmented_bonds(self):
    if not hasattr(self,"_fragmented_bonds"):
      fragmented_bonds = []
      atom_set = set()
      for bond_idx,atom_idxs in self.bond_dict.items():
        a1,a2 = atom_idxs
        if (a1 in self.atom_indices and a2 not in self.atom_indices) or (a2 in self.atom_indices and a1 not in self.atom_indices):
          fragmented_bonds.append(bond_idx)
      self._fragmented_bonds = fragmented_bonds
    return self._fragmented_bonds

  
  def extract_fragment(self,addDummies=True,sanitizeFrags=False,radius=1):
    """
    Extract the molecular fragment from the parent rdmol

    :param addDummies: rdkit parameter, whether to add dummy atoms where bonds are broken
    """
    found_piece = False
    desired_piece = None

    in_atoms = list(self.atom_indices)
    edge_atoms = []
    for i in range(radius):
      if i>0:
        in_atoms+=edge_atoms
        edge_atoms = []
      for atomidx in in_atoms:
        atom = self.rdmol.GetAtomWithIdx(atomidx)
        nbrs = atom.GetNeighbors()
        for nbr in nbrs:
          nbridx = nbr.GetIdx()
          if nbridx not in in_atoms:
            if nbridx not in edge_atoms:
              edge_atoms.append(nbridx)

    fragmented_bonds = []
    for bond in self.rdmol.GetBonds():
      starti,endi = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
      if (starti in edge_atoms and endi in in_atoms) or (starti in in_atoms and endi in edge_atoms):
        fragmented_bonds.append(bond.GetIdx())

    if len(fragmented_bonds)>0:
      fragged = Chem.FragmentOnBonds(self.rdmol,fragmented_bonds,addDummies=addDummies)

    else:
      fragged = self.rdmol

    pieces_indices = Chem.GetMolFrags(fragged, asMols=False,sanitizeFrags=sanitizeFrags)
    pieces_mols = Chem.GetMolFrags(fragged, asMols=True,sanitizeFrags=sanitizeFrags)

    valid_pieces = []
    for inds,piece in zip(pieces_indices,pieces_mols):
      if set(self.atom_indices).issubset(set(inds)):
        valid_pieces.append(piece)
  #     target_num = len(self.atom_indices)
  #     if addDummies:
  #       target_num+=len(self.fragmented_bonds)

  #     valid_pieces = []
  #     # look for match by atom number
  #     for piece in pieces:
  #       num = piece.GetNumAtoms()
  #       if num == target_num:
  #         valid_pieces.append(piece)

  #     if len(valid_pieces)==1:
  #       found_piece = True
  #       desired_piece = valid_pieces[0]
  #     if not found_piece:
  #       # look for match by conformer
  #       if self.has_conformer:
  #         xyz_fragment = self.xyz_fragment

  #         for piece in valid_pieces:
  #           if not found_piece:
  #             conf = piece.GetConformer()
  #             piece_xyz = np.vstack([conf.GetAtomPosition(atom.GetIdx()) for atom in piece.GetAtoms()])

  #             D = cdist(xyz_fragment,piece_xyz)
  #             n_matches = np.argwhere(D==0).shape[0]
  #             if n_matches == len(self):
  #               found_piece = True
  #               desired_piece = piece

  #     if not found_piece:
  #       desired_piece = None
  #     # look for match by smarts
  #       smarts = [Chem.MolToSmarts(piece) for piece in valid_pieces]
  #       query_mols = [Chem.MolFromSmarts(smart) for smart in smarts]
  #       for i,query_mol in enumerate(query_mols):
  #         if not found_piece:
  #           for match in self.rdmol.GetSubstructMatches(query_mol):
  #             if not set(match).issubset(set(self.atom_indices)):
  #               desired_piece = valid_pieces[i]
  #               found_piece = True
    assert(len(valid_pieces)==1)
    desired_piece = valid_pieces[0]
    return desired_piece
 
  def show(self,highlight=True,hide_H=True,print_indices=False,only_connected=True,molSize=(600,600),figsize=10,interactive=True):
    """
    :param highlight: whether to highlight the frag in the mol
    :param hide_H: whether to hide hydrogens connected to carbons
    :param print_indices: whether to label the atoms using atom indices
    :param only_connected: whether to show only the connected components
                           of the parent rdmol. (omit water for example)
    :param molSize: rdkit molSize parameter, controls resolution
    :param figsize: matplotlib figsize parameter, controls image size
    :param interactive: whether this function is being called in a notebook
                        if not, it will return the image as bitmap
    """


    """
    When using only connected components,
    or removing Hs, atom indexing gets very
    confusing. Types of indices:
    1. original inds (from self.rdmol)
    2. piece inds (from a connected piece of rdmol)
    3. rmH inds (inds without Hs, possible also a piece)

    Types of index mapping dicts created below:
    piece_map:  orig_inds -> piece_inds
    piece_map_rev: orig_inds <- piece_inds

    H_map: piece_inds -> rmH_inds
    H_map_rev: piece_inds <- rmH_inds
    """
    rdmol = copy.deepcopy(self.rdmol)

    if only_connected:
      pieces_indices = Chem.GetMolFrags(self.rdmol, asMols=False)
      pieces_mols = Chem.GetMolFrags(self.rdmol, asMols=True)

      valid_pieces = []

      for inds,piece in zip(pieces_indices,pieces_mols):
        if set(self.atom_indices).issubset(set(inds)):
          valid_pieces.append((inds,piece))
      assert(len(valid_pieces)==1)
      atom_inds, rdmol_piece = valid_pieces[0]
      piece_map = {atom.GetIdx():atom_inds[i] for i,atom in enumerate(rdmol_piece.GetAtoms())}
    else:
      rdmol_piece = rdmol
      piece_map = {atom.GetIdx():atom.GetIdx() for i,atom in enumerate(rdmol_piece.GetAtoms())}



    if hide_H and "H" in self.atom_symbols:
      print("Hydrogen is present in fragment. Cannot show fragment and hide hydrogens.")
      hide_H = False

    if hide_H:
      rdmol_noH,H_map= remove_H(rdmol_piece)
    else:
      rdmol_noH = rdmol_piece
      H_map = {atom.GetIdx():atom.GetIdx() for i,atom in enumerate(rdmol_piece.GetAtoms())}


    piece_map_rev = {value:key for key,value in piece_map.items()}
    H_map_rev = {value:key for key,value in H_map.items()}


    Chem.rdDepictor.Compute2DCoords(rdmol_noH)
    drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0],molSize[1])



    if highlight:

      highlight_atoms = [H_map_rev[piece_map_rev[idx]] for idx in self.atom_indices]
    else:
      highlight_atoms = []

    bond_dict = {bond.GetIdx():[bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()] for bond in rdmol_noH.GetBonds()}
    highlight_bonds = []
    for key,value in bond_dict.items():
      if set(value).issubset(highlight_atoms):
        highlight_bonds.append(key)

    rdMolDraw2D.PrepareAndDrawMolecule(drawer,rdmol_noH,highlightAtoms=highlight_atoms,highlightBonds=highlight_bonds)

    if print_indices:
      for atom in rdmol_noH.GetAtoms():
        atom.SetProp("atomNote", str(piece_map[H_map[atom.GetIdx()]]))


    drawer.DrawMolecule(rdmol_noH)
    drawer.FinishDrawing()
    # read as bitmap
    file = io.BytesIO(drawer.GetDrawingText())
    mol_bitmap = plt.imread(file)

    if not interactive:
      return mol_bitmap
    else:
      fig, ax = plt.subplots(1,1,figsize=(figsize,figsize))
      ax.imshow(mol_bitmap)
      ax.axes.spines["left"].set_visible(False)
      ax.axes.spines["top"].set_visible(False)
      ax.axes.spines["right"].set_visible(False)
      ax.axes.spines["bottom"].set_visible(False)
      ax.xaxis.set_visible(False)
      ax.yaxis.set_visible(False)