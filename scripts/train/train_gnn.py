import sys
sys.path.append("../../")
import os
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # suppress rdkit output

import tqdm
import numpy as np
 
from pathlib import Path
import argparse

import torch
from molgraph.MolGraph import MolGraph



if __name__ == '__main__':
  argparser = argparse.ArgumentParser("Train an GNN based NN model to predict restraints")
  argparser.add_argument('--mol_files', type=str, default='.',
                         help="Path to a folder with .mol files")

  argparser.add_argument('--nproc', type=int, default=16,help="Number of processes for parallel execution")

  args = argparser.parse_args()
  
  
  # get molecules
  mol_dir = Path(args.mol_files)
  mol_files = [file for file in mol_dir.glob("**/*") if file.suffix == ".mol"]
  print("Total mol files:",len(mol_files))
  
  # Read in
  mdatas = [{"mol_file":mol_file,"rdmol":Chem.MolFromMolFile(str(mol_file),removeHs=False)} for mol_file in tqdm.tqdm(mol_files)]
  
  def worker(mdata,debug=False,ret_graph=True,save=False,overwrite=True):
    try:
      rdmol,filepath = mdata["rdmol"], mdata["mol_file"]
      graph = MolGraph(rdmol,filepath=filepath,default_mol_type="mol3d")
      if save:
        graph_file = Path(filepath.parent,filepath.stem+"_mgraph.bin")
        if overwrite:
          if graph_file.exists():
            graph_file.unlink()
        torch.save(graph,str(graph_file))
      if ret_graph:
        return graph
    except:
      if debug:
        raise
  
  
  results = []
  for mdata in tqdm.tqdm(mdatas):
    mgraph = worker(mdata,save=True)
    results.append(mgraph)