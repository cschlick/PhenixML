import numpy as np
from phenixml.fragments.fragments import Fragment, MolContainer
from phenixml.fragmentation.fragmenters import BondFragmenter
from phenixml.featurizers.atom_featurizers import RDKIT_Fingerprint
from phenixml.graphs.graph_utils import build_fragment_heterograph, build_atom_graph_from_rdkit, get_indices_from_mol
from rdkit.Chem import rdMolTransforms

import torch
import tqdm
import random

# Bugs
# Currently this does not work well excluding elements (like hydrogen). A more robust graph
# building sequence is needed to accomodate a graph that is smaller than the "actual" atom 
# graph present in the MolContainer


class MolGraph:
  """
  A class to build and store a dgl graph, given a:
    1. MolContainer object
    2. Fragmenter
    3. Fragment Labeler
  
  
  Optionally can specify:
    1. An atom featurizer
    2. A Fragmenter to define bonded edges
    3. A Fragmenter to define nonbonded edges
    
  The resulting dgl.heterograph will have two node types: (atom,fragment)
  """
  @classmethod
  def from_containers(cls,
                     ## Required
                     containers,
                     fragmenter=None,
                     labeler=None,
                     atom_featurizer = RDKIT_Fingerprint(),


                     ## Optional
                     bonded_fragmenter = BondFragmenter(exclude_elements=[]),
                     nonbonded_fragmenter = None,
                    frag_name = "fragment",
                    node_name = "atom",
                    label_ref_name = "ref",
                      
                    # constructor specific
                     skip_failures=True,
                     disable_progress=False): # if True, failures will be skipped

      molgraphs = []
      for container in tqdm.tqdm(containers,disable=disable_progress):
        try:
          molgraph = MolGraph(mol_container=container,
                                fragmenter=fragmenter,
                                 labeler=labeler,
                                 atom_featurizer = atom_featurizer,


                                 ## Optional
                                 bonded_fragmenter =bonded_fragmenter,
                                 nonbonded_fragmenter = nonbonded_fragmenter,
                                frag_name = frag_name,
                                node_name = node_name,
                                label_ref_name = label_ref_name)
          molgraphs.append(molgraph)
        except:
          if not skip_failures:
            molgraph = MolGraph(mol_container=container,
                                fragmenter=fragmenter,
                                 labeler=labeler,
                                 atom_featurizer = atom_featurizer,


                                 ## Optional
                                 bonded_fragmenter =bonded_fragmenter,
                                 nonbonded_fragmenter = nonbonded_fragmenter,
                                frag_name = frag_name,
                                node_name = node_name,
                                label_ref_name = label_ref_name)
            assert False, "Failed building graph from container "
      return molgraphs
  
  def __init__(self,
               
               ## Required 
               mol_container=None,
               fragmenter=None,
               labeler=None,
               atom_featurizer = RDKIT_Fingerprint(),
               
               
               ## Optional
               bonded_fragmenter = BondFragmenter(exclude_elements=[]),
               nonbonded_fragmenter = None,
              frag_name = "fragment",
              node_name = "atom",
              label_ref_name = "ref"):
    
    assert None not in [mol_container,fragmenter], "Initialize with a MolContainer and a Fragmenter"
    
    self.mol_container = mol_container
    self.fragmenter = fragmenter
    self.atom_featurizer = atom_featurizer
    


    mol = mol_container.rdkit_mol


    # use rdkit
    atom_graph = build_atom_graph_from_rdkit(mol,atom_featurizer=atom_featurizer)

    idxs = get_indices_from_mol(mol)
    atom_idxs = np.arange(atom_graph.number_of_nodes())[:,np.newaxis]
    fragments = fragmenter.fragment(mol_container)
    self.fragments = fragments
    fragment_idxs = np.array([fragment.atom_selection for fragment in fragments])
    #fragment_idxs = np.vstack([fragment_idxs,np.flip(fragment_idxs,axis=1)])
    bonded_fragments = bonded_fragmenter.fragment(mol_container)
    bonded_idxs = np.array([fragment.atom_selection for fragment in bonded_fragments])
    bonded_idxs = np.vstack([bonded_idxs,np.flip(bonded_idxs,axis=1)])
    if nonbonded_fragmenter is not None:
      nonbonded_fragments = nonbonded_fragmenter.fragment(mol_container)
      nonbonded_idxs = np.array([fragment.atom_selection for fragment in nonbonded_fragments])
      nonbonded_idxs = np.vstack([nonbonded_idxs,np.flip(nonbonded_idxs,axis=1)])
    else:
      nonbonded_idxs = None

    self.heterograph = build_fragment_heterograph(atom_graph = atom_graph,
                                                atom_idxs = atom_idxs,
                                                bonded_idxs=bonded_idxs,
                                                nonbonded_idxs=nonbonded_idxs,
                                                frag_idxs=fragment_idxs,
                                                frag_name = frag_name,
                                                node_name = node_name)
    
    labels = np.array([labeler(fragment) for fragment in fragments])
    #labels = np.concatenate([labels,labels])
    labels = labels[:,np.newaxis]
    self.heterograph.nodes[frag_name].data[label_ref_name] = torch.tensor(labels,dtype=torch.get_default_dtype())
    

    
  @property
  def rdkit_mol(self):
    return self.mol_container.rdkit_mol
  
  @property
  def mol_container_noH(self):
    if not hasattr(self,"_mol_container_noH"):
      return self.mol_container