from rdkit import Chem
import numpy as np




def bond_length_eval(mgraph,model,debug=False):
  """
  A MolGraph object, and a pretrained pytorch model. 
  
  Returns: 
  A list of list, where each element corresponds to a bond fragment
  
  [atom_idxs,atom_symbols,predicted_angstroms]
  
  Example:
  [[(0,1),("C","C"),1.482],
  ....
  ]
  
  Notes:
  1. Debug=True prints info
  """

  # evaluate graph
  g = model(mgraph.heterograph)
  
  # get data from graph
  idxs = g.nodes["n2"].data["idxs"].detach().cpu().numpy()  # edge atom idxs
  eq = g.nodes["n2"].data["eq"].detach().cpu().numpy()[:,0] # eq value for bond length
  eq_ref = g.nodes["n2"].data["eq_ref"].detach().cpu().numpy()[:,0] # ref bond length
  atom_type = mgraph.homograph.ndata["type"].detach().cpu().numpy()[:,0].astype(int) # atomic number
  pt  = Chem.GetPeriodicTable()
  atom_type_symbols = np.array([pt.GetElementSymbol(int(anum)) for anum in atom_type]) # symbols for debugging
  symbols = atom_type_symbols[idxs] # symbols at each idx
  
  # unpack results to return as described in docstring. Use a set so we don't assume where the reverse edges are
  unique_edges = set()
  results = []
  for i,idx in enumerate(idxs):
    s = frozenset([idx[0],idx[1]])
    if s not in unique_edges:
      unique_edges.add(s)
      results.append([tuple(idx),tuple(symbols[i]),eq[i],eq_ref[i]])

  # optionally print output
  if debug:
    ljust = 20
    print("\nBond results: (atom index, element, bond pred, bond_ref, |error| )\n")
    for result in results:
      inds,syms = str(result[0]), str(result[1])
      eq,eq_ref = result[2], result[3]
      error = abs(eq-eq_ref)
      print(inds.ljust(ljust),
            syms.ljust(ljust),
            str(round(eq,3)).ljust(ljust),
            str(round(eq_ref,3)).ljust(ljust),
            str(round(error,3)).ljust(ljust))
      
  return results


def bond_angle_eval(mgraph,model,debug=False):
  """
  A MolGraph object, and a pretrained pytorch model. 
  
  Returns: 
  A list of list, where each element corresponds to an angle fragment
  
  [atom_idxs,atom_symbols,predicted_degrees]
  
  Example:
  [[(0,1,2),("C","C","C"),120.0],
  ...
  ]
  
  Notes:
  1. Debug=True prints info
  """

  # evaluate graph
  g = model(mgraph.heterograph)
  
  # get data from graph
  idxs = g.nodes["n3"].data["idxs"].detach().cpu().numpy()  # edge atom idxs
  eq = np.degrees(g.nodes["n3"].data["eq"].detach().cpu().numpy()[:,0]) # eq value for bond length
  eq_ref = np.degrees(g.nodes["n3"].data["eq_ref"].detach().cpu().numpy()[:,0]) # ref bond length
  atom_type = mgraph.homograph.ndata["type"].detach().cpu().numpy()[:,0].astype(int) # atomic number
  pt  = Chem.GetPeriodicTable()
  atom_type_symbols = np.array([pt.GetElementSymbol(int(anum)) for anum in atom_type]) # symbols for debugging
  symbols = atom_type_symbols[idxs] # symbols at each idx
  
  # unpack results to return as described in docstring. Use a set so we don't assume where the reverse edges are
  unique_edges = set()
  results = []
  for i,idx in enumerate(idxs):
    s = frozenset([idx[0],idx[1]])
    if s not in unique_edges:
      unique_edges.add(s)
      results.append([tuple(idx),tuple(symbols[i]),eq[i],eq_ref[i]])

  # optionally print output
  if debug:
    ljust = 20
    print("\nAngle results: (atom index, element, pred, ref, |error| )\n")
    for result in results:
      inds,syms = str(result[0]), str(result[1])
      eq,eq_ref = result[2], result[3]
      error = abs(eq-eq_ref)
      print(inds.ljust(ljust),
            syms.ljust(ljust),
            str(round(eq,3)).ljust(ljust),
            str(round(eq_ref,3)).ljust(ljust),
            str(round(error,3)).ljust(ljust))
      
  return results