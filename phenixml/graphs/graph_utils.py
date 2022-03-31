import dgl
import torch
import numpy as np
from rdkit import Chem

from phenixml.utils.rdkit_utils import *

def to_np(tensor):
  return tensor.detach().cpu().numpy()

def build_atom_graph_from_rdkit(rdkit_mol,atom_featurizer=None,keep_xyz=False):
    """
    Build an all-atom graph, optionally featurized with an initial atom featurizer
    """
    rdmol = rdkit_mol
    mol = rdmol
    # build graph
    

    bonds = list(mol.GetBonds())
    bonds_types = [bond.GetBondType().real for bond in bonds]
    data = enumerate_bonds(mol)

    data = np.vstack([data,np.flip(data,axis=1)]) # add reverse direction edges
    g = dgl.graph((data[:,0],data[:,1]))
    
    if atom_featurizer is not None:
      # add features
      features = np.vstack([atom_featurizer(atom) for atom in rdmol.GetAtoms()])
      g.ndata["h0"] = torch.from_numpy(features) # set initial representation
    
    # set xyz coords
    if keep_xyz:
      if len(rdmol.GetConformers())>0:
        conf = rdmol.GetConformer()
        pos = conf.GetPositions()
        g.ndata["pos"] = torch.tensor(pos,dtype=torch.float32)
      
    
    return g



def build_fragment_heterograph(atom_graph=None,
                               atom_idxs = None,
                               bonded_idxs=None,
                               nonbonded_idxs=None,
                               frag_idxs=None,
                               frag_name = "n3",
                               node_name = "n1"):




    # initialize empty dictionary
    hg = {}
    hg[(node_name,"%s_%s_%s" % (node_name,"bonded",node_name), node_name)] = bonded_idxs
    
    if not isinstance(nonbonded_idxs,type(None)):
      hg[(node_name,"%s_%s_%s" % (node_name,"nonbonded",node_name), node_name)] = nonbonded_idxs
    
    idxs = {}
    idxs[node_name] = atom_idxs
    idxs["bonded"] = bonded_idxs
    if not isinstance(nonbonded_idxs,type(None)):
      idxs["nonbonded"] = bonded_idxs
    idxs[frag_name] = frag_idxs

    terms = [node_name,frag_name] # debug
    levels = terms
    
    
    idxs_to_ordering = {}
    for term in terms:
      idxs_to_ordering[term] = {
          tuple(subgraph_idxs): ordering
          for (ordering, subgraph_idxs) in enumerate(list(idxs[term]))
      }

    # this needs rewriting. Current is based on espaloma
    for small_idx in range(1, len(terms)+1):
      for big_idx in range(small_idx + 1, len(terms)+1):
          for pos_idx in range(frag_idxs.shape[1]):
              hg[
                  (
                      node_name,
                      "%s_as_%s_in_%s" % (node_name,pos_idx,frag_name),
                      frag_name
                  )
              ] = np.stack(
                  [
                      np.array(
                          [
                              idxs_to_ordering[node_name][tuple(x)]
                              for x in idxs[frag_name][
                                  :, pos_idx : pos_idx + small_idx
                              ]
                          ]
                      ),
                      np.arange(idxs[frag_name].shape[0]),
                  ],
                  axis=1,
              )
              # #comment out has relationships. I think this is ok since we are only using the higher level nodes for readout
              # hg[
              #     (
              #         "n%s" % big_idx,
              #         "n%s_has_%s_n%s" % (big_idx, pos_idx, small_idx),
              #         "n%s" % small_idx,
              #     )
              # ] = np.stack(
              #     [
              #         np.arange(idxs["n%s" % big_idx].shape[0]),
              #         np.array(
              #             [
              #                 idxs_to_ordering["n%s" % small_idx][tuple(x)]
              #                 for x in idxs["n%s" % big_idx][
              #                     :, pos_idx : pos_idx + small_idx
              #                 ]
              #             ]
              #         ),
              #     ],
              #     axis=1,
              # )
    import dgl
    hg = dgl.heterograph({key: list(value) for key, value in hg.items()})

    hg.nodes[node_name].data["h0"] = atom_graph.ndata["h0"].type(torch.get_default_dtype())
    # set the n1 nodes to have the features from the atoms in the homograph

    # include indices in the idxs to nodes in the homograph
    for term in levels:
        hg.nodes[term].data["idxs"] = torch.tensor(idxs[term])

    return hg




def get_indices_from_mol(mol,levels=["n1","n2","n3","n4"]):
  idxs = {}
  if "n1" in levels:
    atoms = np.arange(mol.GetNumAtoms())[:,np.newaxis]
    idxs["n1"] = atoms
  
  if "n2" in levels:
    bonds = enumerate_bonds(mol)
    if len(bonds)>0:
      bonds = np.vstack([bonds,np.flip(bonds,axis=1)])
    idxs["n2"] = bonds
  
  if "n3" in levels:
    angles = enumerate_angles(mol)
    if len(angles)>0:
      angles = np.vstack([angles,np.flip(angles,axis=1)])
    idxs["n3"] = angles
  
  if "n4" in levels:
    torsions = enumerate_torsions(mol)
    if len(torsions)>0:
      torsions = np.vstack([torsions,np.flip(torsions,axis=1)])
    idxs["n4"] = torsions
  
  return idxs

