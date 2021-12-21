from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # suppress rdkit output

import numpy as np
  
from pathlib import Path
import argparse

import torch

from phenixml.featurizers.ani_featurizer import ANIFeaturizer
from phenixml.fragmentation.fragmenter_restraints import BondFragmenter
from phenixml.fragmentation.fragmenter_restraints import AngleFragmenter
from phenixml.labelizers.bond_angles_lengths import AngleFragLabeler
from phenixml.labelizers.bond_angles_lengths import BondFragLabeler


default_elements_considered = ["C","H","N","O","P","S","Cl","B","F","I","Br"]
default_ani_params ={'radial_cutoff': 4.6,
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
 'elements_considered': default_elements_considered}


def bond_length_eval(rdmol,model_bond,ani_params=default_ani_params,debug=False):
  """
  A rdkit molecule, ANI params, and a pretrained pytorch model. 
  
  Returns: 
  A list of list, where each element corresponds to a bond fragment
  
  [atom_idxs,atom_symbols,predicted_angstroms]
  
  Example:
  [[(0,1),("C","C"),1.482],
  ....
  ]
  
  Notes:
  1. Debug=True prints info
  2. It is important that the ani_params be the same as was used druing training.
  """

  
  # fragment on all bonds
  bond_fragmenter = BondFragmenter(exclude_symbols=[])
  bond_frags =bond_fragmenter.fragment(rdmol)
  
  
  # check if we have xyz in the mol
  has_ref = len(rdmol.GetConformers())>0
  if has_ref: # if xyz available, calculate input labels
    labeler = BondFragLabeler()
    labels_bonds = np.array([labeler.labelize(frag) for frag in bond_frags])
  
  # get feature vector for all frags
  bond_features = []
  for frag in bond_frags:
    feature = ANIFeaturizer.from_bond_angle_frags([frag],ani_params).featurize()
    bond_features.append(feature)
  bond_features = np.vstack(bond_features).astype(np.float32)
  
  # perform inference
  X_bond = torch.from_numpy(bond_features)
  y_bond = model_bond(X_bond)[:,0].detach().numpy()
  
  # collect output
  return_value = []
  for i,frag in enumerate(bond_frags):
    return_value.append([tuple(frag.atom_indices),tuple(frag.atom_symbols),y_bond[i]])
    
  # optionally print output
  if debug:
    ljust = 20
    if has_ref:
      error = np.abs(labels_bonds-y_bond)
      print("\nBond results: (atom index, element, bond pred, bond_ref, |error| )\n")
      for i,frag in enumerate(bond_frags):
        print(str(frag.atom_indices).ljust(ljust),
              str(frag.atom_symbols).ljust(ljust),
              str(round(float(y_bond[i]),3)).ljust(ljust),
              str(round(float(labels_bonds[i]),3)).ljust(ljust),
              str(round(float(error[i]),3)).ljust(ljust))
    else:
      print("\nBond results: (atom index, element, angle pred)\n")
      for i,frag in enumerate(bond_frags):
        print(str(frag.atom_indices).ljust(ljust),str(frag.atom_symbols).ljust(ljust),str(float(y_bond[i])).ljust(ljust))
  
  return return_value



def bond_angle_eval(rdmol,model_angle,ani_params=default_ani_params,debug=False):
  """
  A rdkit molecule, ANI params, and a pretrained pytorch model. 
  
  Returns: 
  A list of list, where each element corresponds to a angle fragment
  
  [atom_idxs,atom_symbols,predicted_degrees]
  
  Example:
  [[(0,1,2),("C","C","C"),120.0],
  ....
  ]
  
  Notes:
  1. Debug=True prints info
  2. It is important that the ani_params be the same as was used druing training.
  """

  
  # fragment on all angles
  angle_fragmenter = AngleFragmenter(exclude_symbols=[])
  angle_frags =angle_fragmenter.fragment(rdmol)
  
  
  # check if we have xyz in the mol
  has_ref = len(rdmol.GetConformers())>0
  if has_ref: # if xyz available, calculate input labels
    labeler = AngleFragLabeler()
    labels_angles = np.array([labeler.labelize(frag) for frag in angle_frags])
  
  # get feature vector for all frags
  angle_features = []
  for frag in angle_frags:
    feature = ANIFeaturizer.from_bond_angle_frags([frag],ani_params).featurize()
    angle_features.append(feature)
  angle_features = np.vstack(angle_features).astype(np.float32)
  
  # perform inference
  X_angle = torch.from_numpy(angle_features)
  y_angle = model_angle(X_angle)[:,0].detach().numpy()
  
  # collect output
  return_value = []
  for i,frag in enumerate(angle_frags):
    return_value.append([tuple(frag.atom_indices),tuple(frag.atom_symbols),y_angle[i]])
    
  # optionally print output
  if debug:
    ljust = 20
    if has_ref:
      error = np.abs(labels_angles-y_angle)
      print("\nAngle results: (atom index, element, angle pred, angle_ref, |error| )\n")
      for i,frag in enumerate(angle_frags):
        print(str(frag.atom_indices).ljust(ljust),
              str(frag.atom_symbols).ljust(ljust),
              str(round(float(y_angle[i]),3)).ljust(ljust),
              str(round(float(labels_angles[i]),3)).ljust(ljust),
              str(round(float(error[i]),3)).ljust(ljust))
    else:
      print("\nAngle results: (atom index, element, angle pred)\n")
      for i,frag in enumerate(angle_frags):
        print(str(frag.atom_indices).ljust(ljust),str(frag.atom_symbols).ljust(ljust),str(float(y_angle[i])).ljust(ljust))
  
  return return_value