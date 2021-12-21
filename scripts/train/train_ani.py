import sys
sys.path.append("../../")
import os
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # suppress rdkit output

import tqdm
import numpy as np
  import matplotlib.pyplot as plt
  
from pathlib import Path
import argparse

import torch


if __name__ == '__main__':
  argparser = argparse.ArgumentParser("Train an ANI based NN model to predict bond/angle restraints. Currently enforces elemental subset: C,H,N,O,P,S,C,L,B,F,I,Br")
  argparser.add_argument('--files_mol2', type=str, default='.',
                         help="Path to a folder with .mol files")

  argparser.add_argument('--nproc', type=int, default=16,help="Number of processes for parallel execution")

  args = argparser.parse_args()
  
  
  # get molecules
  mol_dir = Path(args.files_mol2)
  mol_files = [file for file in mol_dir.glob("**/*") if file.suffix == ".mol"]
  print("Total MOL2 files:",len(mol_files))


  # hard coded filtering of an element subset
  pt  = Chem.GetPeriodicTable()
  elements_considered = ["C","H","N","O","P","S","Cl","B","F","I","Br"]
  mdatas = [{"mol_file":mol_file,"rdmol":Chem.MolFromMolFile(str(mol_file),removeHs=False)} for mol_file in mol_files]
  mdatas = [mdata for mdata in mdatas if mdata["rdmol"] is not None]

  # filter by element
  def rdmol_element_set(rdmol):
    return set([atom.GetSymbol() for atom in rdmol.GetAtoms()])
  mdatas =[mdata for mdata in mdatas if rdmol_element_set(mdata["rdmol"]).issubset(set(elements_considered))]

  rdmols = [mdata["rdmol"] for mdata in mdatas]
  
  
  # define ANI model params
  
  from phenixml.featurizers.ani_featurizer import ANIFeaturizer
  # elements and parameters

  params = {'radial_cutoff': 4.6,
   'radial_nu': 32,
   'radial_probes': [0.7,
                    1.4,
                    1.9,
                    2.4,
                    3.2,
                    3.8,
                    4.4],
   'angular_cutoff': 3.1,
   'angular_nu': 4,
   'angular_probes': [0.0, 1.57, 3.14, 4.71],
   'angular_radial_probes': [0.7,1.4,1.9,2.4],
   'angular_zeta': 8,
   'min_probed_value': 0.0,
   'exclude_hydrogens': False,
   'elements_considered': elements_considered}
  
  #####################################
  ## Angles
  #####################################
  
  from phenixml.fragmentation.fragmenter_restraints import AngleFragmenter
  angle_fragmenter = AngleFragmenter(exclude_symbols=["H"])
  frags = []
  for rdmol in rdmols:
    fs = angle_fragmenter.fragment(rdmol)
    if fs is not None and len(fs)>0:
      frags+=fs
  print("Angle Framents:",len(frags))
  
  def worker(frag):
    #feature = ANIFeaturizer.from_bond_angle_frag_atom_centered(frag,params).featurize()
    feature = ANIFeaturizer.from_bond_angle_frags([frag],params).featurize()
    return feature

  # run worker with multiprocessing
  from contextlib import closing
  import tqdm
  from multiprocessing import Pool

  work = frags

  with closing(Pool(processes=64)) as pool:
    results = []
    for result in tqdm.tqdm(pool.map(worker, work), total=len(work)):
        results.append(result)
    pool.terminate()
    
    
  features = results
  X = np.vstack(features)
  
  
  from phenixml.labelizers.bond_angles_lengths import AngleFragLabeler

  labeler = AngleFragLabeler(degrees=True)
  labels = np.array([labeler.labelize(frag) for frag in frags])
  y = labels

  
  model = Model(X.shape[1],128,1)
  #model = model.to("cuda:0")
  X = torch.tensor(X,dtype=torch.float32)#.to("cuda:0")
  y = torch.tensor(y,dtype=torch.float32)#.to("cuda:0")
  loss_fn = torch.nn.MSELoss()
  opt = torch.optim.Adam(model.parameters())


  
  
  loss_history = []
  epochs = 3000
  for epoch in tqdm.tqdm(range(epochs)):
    opt.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y,y_pred[:,0])
    loss.backward()
    opt.step()
    loss_history.append(float(loss))
  
  
  plt.plot(loss_history[10:])
  plt.savefig("training_loss_angles.png")
  
  
  
  
  #####################################
  ## Bonds
  #####################################