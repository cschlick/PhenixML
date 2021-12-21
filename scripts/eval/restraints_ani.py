import sys
sys.path.append("../../")

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # suppress rdkit output

import numpy as np
from pathlib import Path
import argparse
import time
import torch

from phenixml.models.feature_regression import FeatureModel
from phenixml.evaluation.ani_evaluation import bond_length_eval, bond_angle_eval




if __name__ == '__main__':
  argparser = argparse.ArgumentParser("Evaluate ANI model for predicting restraints.")
  argparser.add_argument('--file', type=str, help="Path to a single .mol file to evaluate")
  argparser.add_argument('--smiles', type=str, help="Smiles string to evaluate")
  argparser.add_argument('--angle_pt', type=str,default="../../pretrained/ani_angles.pt", help="Path to pre-trained angle model")
  argparser.add_argument('--bond_pt', type=str,default="../../pretrained/ani_bonds.pt", help="Path to pre-trained bond model")
  args = argparser.parse_args()
  
  if args.file is None and args.smiles is None:
    argparser.print_help()
    
  # get mol  
  if args.file:
    rdmol = Chem.MolFromMolFile(args.file,removeHs=False)
  else:
    rdmol = Chem.MolFromSmiles(args.smiles)
    rdmol = Chem.AddHs(rdmol)
    
    # generate 3d coords using RDkit (needed for ani)
    from rdkit.Chem import AllChem
    _ = AllChem = AllChem.EmbedMolecule(rdmol,randomSeed=0xf00d)
  
  # check atom subset
  pt  = Chem.GetPeriodicTable()
  elements_considered = ["C","H","N","O","P","S","Cl","B","F","I","Br"]
  
  def rdmol_element_set(rdmol):
    return set([atom.GetSymbol() for atom in rdmol.GetAtoms()])
  assert rdmol_element_set(rdmol).issubset(set(elements_considered)), "Must provide a molecule containing only these elements: "+str(elements_considered)
  
  
  # get models
  angle_pt = torch.load(args.angle_pt)
  bond_pt = torch.load(args.bond_pt)
  
  # evaluate
  bond_results = bond_length_eval(rdmol,bond_pt,debug=True)
  angle_results = bond_angle_eval(rdmol,angle_pt,debug=True)
  if args.smiles:
    print("\nNote: Smiles was provided, so 'reference' refers to the values obtained by the RDkit ETKDG method.")