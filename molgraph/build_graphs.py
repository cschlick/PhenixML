import numpy as np
import dgl
import torch
from rdkit import Chem

from molgraph.esp_fragments.atom_featurizer import fp_rdkit


def to_np(tensor):
  return tensor.detach().cpu().numpy()

def enumerate_bonds(mol):
  idx_set = set()
  for atom in mol.GetAtoms():
    for neigh1 in atom.GetNeighbors():
      idx1,idx2 = atom.GetIdx(), neigh1.GetIdx()
      s = frozenset([idx1,idx2])
      if len(s)==2:
        idx_set.add(s)
  # check that it matches a more direct approach
  check = {frozenset((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())) for bond in mol.GetBonds()}
  assert idx_set == check
  return np.array(sorted([list(s) for s in idx_set]))

def enumerate_angles(mol):
  idx_set = set()
  for atom in mol.GetAtoms():
    for neigh1 in atom.GetNeighbors():
      for neigh2 in neigh1.GetNeighbors():
        idx0,idx1,idx2 = atom.GetIdx(), neigh1.GetIdx(),neigh2.GetIdx()
        s = (idx0,idx1,idx2)
        if len(set(s))==3:
          if idx0>idx2:
            idx0,idx2 = idx2,idx0
          idx_set.add((idx0,idx1,idx2))
  return np.array([list(s) for s in idx_set])

def enumerate_torsions(mol):
  idx_set = set()
  for atom0 in mol.GetAtoms():
    idx0 = atom0.GetIdx()
    for atom1 in atom0.GetNeighbors():
      idx1 = atom1.GetIdx()
      for atom2 in atom1.GetNeighbors():
        idx2 = atom2.GetIdx()
        if idx2==idx0:
          continue
        for atom3 in atom2.GetNeighbors():
          idx3 = atom3.GetIdx()
          if idx3 == idx1 or idx3 == idx0:
            continue         
          s = (idx0,idx1,idx2,idx3)
          if len(set(s))==4:
            if idx0<idx3:
              idx_set.add((idx0,idx1,idx2,idx3))
            else:
              idx_set.add((idx3,idx2,idx1,idx0))
            
  return np.array([list(s) for s in idx_set])

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



def build_homograph(mol, use_fp=True,keep_xyz=False):

  bonds = list(mol.GetBonds())
  bonds_begin_idxs = [bond.GetBeginAtomIdx() for bond in bonds]
  bonds_end_idxs = [bond.GetEndAtomIdx() for bond in bonds]
  bonds_types = [bond.GetBondType().real for bond in bonds]
  data = np.array(list(zip(bonds_begin_idxs,bonds_end_idxs)))
  data = np.vstack([data,np.flip(data,axis=1)]) # add reverse direction edges
  g = dgl.graph((data[:,0],data[:,1]))

  g.ndata["type"] = torch.Tensor(
      [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
  )
  
  # one hot encode atom type
  h_v = torch.zeros(g.ndata["type"].shape[0], 100, dtype=torch.float32)

  h_v[
      torch.arange(g.ndata["type"].shape[0]),
      torch.squeeze(g.ndata["type"]).long(),
  ] = 1.0

  

  if use_fp == True:
    h_v_fp = torch.stack([fp_rdkit(atom) for atom in mol.GetAtoms()], axis=0)
    h_v = torch.cat([h_v, h_v_fp], dim=-1)  # (n_atoms, 117)

  g.ndata["h0"] = h_v
  if keep_xyz:
    if len(mol.GetConformers())>0:
      conf = mol.GetConformer()
      pos = conf.GetPositions()
      g.ndata["xyz"] = torch.tensor(pos,dtype=torch.float32)
  # g.edata["type"] = torch.Tensor(bonds_types)[:, None].repeat(2, 1)

  return g

def build_heterograph_from_homo_mol(homograph,mol,levels=["n1","n2","n3","n4"]):
  g = homograph
  # initialize empty dictionary
  hg = {}

  # get adjacency matrix
  a = g.adjacency_matrix()

  # get all the indices
  idxs = get_indices_from_mol(mol)


  # also include n1
  idxs["n1"] = np.arange(g.number_of_nodes())[:, None]

  # =========================
  # neighboring relationships
  # =========================
  # NOTE:
  # here we only define the neighboring relationship
  # on atom level
  hg[("n1", "n1_neighbors_n1", "n1")] = idxs["n2"]

  # build a mapping between homograph node indices and the ordering of nodes in the hetero subgraphs
  idxs_to_ordering = {}

  terms = levels
  for term in terms:
      idxs_to_ordering[term] = {
          tuple(subgraph_idxs): ordering
          for (ordering, subgraph_idxs) in enumerate(list(idxs[term]))
      }

  # ===============================================
  # relationships between nodes of different levels
  # ===============================================

  # Build the dictionary data for constructing a heterogeneous graph. 
  # The keys are in the form of string triplets (src_node_type, edge_type, dst_node_type)
  # "as" and "has" refer to the direction of the edges 

  # NOTE:
  # here we define all the possible
  # 'has' and 'in' relationships.
  # TODO:
  # we'll test later to see if this adds too much overhead
  #

  for small_idx in range(1, len(terms)+1):
      for big_idx in range(small_idx + 1, len(terms)+1):
          for pos_idx in range(big_idx - small_idx + 1):

              hg[
                  (
                      "n%s" % small_idx,
                      "n%s_as_%s_in_n%s" % (small_idx, pos_idx, big_idx),
                      "n%s" % big_idx,
                  )
              ] = np.stack(
                  [
                      np.array(
                          [
                              idxs_to_ordering["n%s" % small_idx][tuple(x)]
                              for x in idxs["n%s" % big_idx][
                                  :, pos_idx : pos_idx + small_idx
                              ]
                          ]
                      ),
                      np.arange(idxs["n%s" % big_idx].shape[0]),
                  ],
                  axis=1,
              )
              # comment out has relationships. I think this is ok since we are only using the higher level nodes for readout
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

  # comment out because not using
  # # ======================================
  # # relationships between nodes and graphs
  # # ======================================


  for term in levels:
      hg[(term, "%s_in_g" % term, "g",)] = np.stack(
          [np.arange(len(idxs[term])), np.zeros(len(idxs[term]))],
          axis=1,
      )

      hg[("g", "g_has_%s" % term, term)] = np.stack(
          [
              np.zeros(len(idxs[term])),
              np.arange(len(idxs[term])),
          ],
          axis=1,
      )

  import dgl
  hg = dgl.heterograph({key: list(value) for key, value in hg.items()})

  hg.nodes["n1"].data["h0"] = g.ndata["h0"] # set the n1 nodes to have the features from the atoms in the homograph

  # include indices in the indxs to nodes in the homograph
  for term in levels:
      hg.nodes[term].data["idxs"] = torch.tensor(idxs[term])

  return hg