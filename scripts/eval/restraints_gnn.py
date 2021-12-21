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

from molgraph.MolGraph import MolGraph
from molgraph.evaluation import bond_length_eval, bond_angle_eval
from phenixml.utils.rdkit_utils import mol_from_smiles


if __name__ == '__main__':
  argparser = argparse.ArgumentParser("Evaluate ANI model for predicting restraints.")
  argparser.add_argument('--file', type=str, help="Path to a single .mol file to evaluate")
  argparser.add_argument('--smiles', type=str, help="Smiles string to evaluate")
  argparser.add_argument('--gnn_pt', type=str,default="../../pretrained/gnn_model.pt", help="Path to pre-trained GNN model")
  args = argparser.parse_args()
     
    
  # get mol  
  if args.file:
    rdmol = Chem.MolFromMolFile(args.file,removeHs=False)
  elif args.smiles:
    rdmol = mol_from_smiles(args.smiles,embed3d=True)
  else:
    argparser.print_help()
  
  # check atom subset
  mgraph = MolGraph(rdmol,default_mol_type="mol3d") # use "mol3d", which has hydrogens
  
  
  # get models
  model = torch.load(args.gnn_pt)

  
  # evaluate (this evaluates twice, which is unnecessary but clean)
  bond_results = bond_length_eval(mgraph,model,debug=True)
  angle_results = bond_angle_eval(mgraph,model,debug=True)
  if args.smiles:
    print("\nNote: Smiles was provided, so 'reference' refers to the values obtained by the RDkit ETKDG method.")