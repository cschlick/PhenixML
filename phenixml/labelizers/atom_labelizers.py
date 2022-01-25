import mmtbx.secondary_structure
from libtbx.utils import null_out
import numpy as np


class AtomLabelizer_SS:
  def __init__(self):
    pass
  
  def labelize(model,
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

    alpha,beta = np.array(alpha),np.array(beta)
    loop = np.full(len(alpha),True)
    loop[alpha]=False
    loop[beta] = False
    if loop.sum()+alpha.sum()+beta.sum() == len(alpha):
      alpha = alpha.tolist()
      beta = beta.tolist()
      ss_atom_annotation = {"alpha":alpha,"beta":beta}

      return ss_atom_annotation