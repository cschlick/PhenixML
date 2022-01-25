from __future__ import absolute_import, division, print_function
import sys
import mmtbx.secondary_structure
from scitbx.array_family import flex
from libtbx.utils import null_out
from iotbx.data_manager import DataManager
import numpy as np
from pathlib import Path
import yaml
import pickle
import tqdm


def get_ss(model,
           method="ksdssp"):
  """
  Calculate the secondary structure annotations using mmtbx.
  
  Returns:
  A dict with a bool value for each atom in the model
  
  {"alpha":[True,True,False,...],
   "beta":[False,False,False,...]}
  
  """
  hierarchy = model.get_hierarchy()
  params = mmtbx.secondary_structure.manager.get_default_ss_params()
  params.secondary_structure.protein.search_method=method
  params = params.secondary_structure
  ssm = mmtbx.secondary_structure.manager(
    pdb_hierarchy         = hierarchy,
    sec_str_from_pdb_file = None,
    params                = params,
    log                   = null_out())
  alpha = ssm.helix_selection()
  beta  = ssm.beta_selection()
  # assert alpha.size() == beta.size() == hierarchy.atoms().size()
  # annotation_vector = flex.double(hierarchy.atoms().size(), 0)
  # annotation_vector.set_selected(alpha, 1)
  # annotation_vector.set_selected(beta, 2)
  # return annotation_vector
  alpha,beta = np.array(alpha),np.array(beta)
  loop = np.full(len(alpha),True)
  loop[alpha]=False
  loop[beta] = False
  if loop.sum()+alpha.sum()+beta.sum() == len(alpha):
    alpha = alpha.tolist()
    beta = beta.tolist()
    ss_atom_annotation = {"alpha":alpha,"beta":beta}

    return ss_atom_annotation