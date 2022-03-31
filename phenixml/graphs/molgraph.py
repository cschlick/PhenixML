import numpy as np
from phenixml.fragmentation.fragments import MolContainer, Fragment
from phenixml.featurizers.atom_featurizers import RDKIT_Fingerprint
from phenixml.graphs.graph_utils import build_fragment_heterograph, build_atom_graph_from_rdkit, get_indices_from_mol
from rdkit.Chem import rdMolTransforms

# debug
#from molgraph.build_graphs import build_heterograph_from_homo_mol

class MolGraph:
  def __init__(self,mol_container,
               fragmenter=None,
               bonded_fragmenter = None,
               nonbonded_fragmenter = None,
               fragment_labeler=None,
               atom_featurizer=RDKIT_Fingerprint(),
              frag_name = "fragment",
              node_name = "atom",
              label_ref_name = "ref"):
    
    
    
    self.mol_container = mol_container
    self.fragmenter = fragmenter
    self.atom_featurizer = atom_featurizer
    
    # use rdkit
    self.atom_graph = build_atom_graph_from_rdkit(mol_container.rdkit_mol,atom_featurizer=atom_featurizer)
    
#     # fragment
#     connect_fragments = edge_fragmenter.fragment(model_container)
#     fragments = fragmenter.fragment(model_container)
    
#     connect_idxs = np.array([fragment.atom_selection for fragment in connect_fragments])
#     fragment_idxs = np.array([fragment.atom_selection for fragment in fragments])
    
#     self.heterograph = build_fragment_heterograph(atom_graph = self.atom_graph,
#                                                   connect_idxs=connect_idxs,
#                                                   frag_idxs=fragment_idxs,
#                                                   frag_name = frag_name,
#                                                   node_name = node_name,
#                                                   connect_name = connect_name)
    
    
    
#     labels = [fragment_labeler(fragment) for fragment in fragments]
#     labels = np.array(labels)[:,np.newaxis]
#     self.heterograph.nodes[frag_name].data[label_ref_name] = torch.tensor(labels,dtype=torch.get_default_dtype())


    idxs = get_indices_from_mol(self.mol_container.rdkit_mol)
    self.idxs = idxs # debug
    atom_idxs = np.arange(self.atom_graph.number_of_nodes())[:,np.newaxis]
    fragments = fragmenter.fragment(self.mol_container)
    fragment_idxs = np.array([fragment.atom_selection for fragment in fragments])
    #fragment_idxs = np.vstack([fragment_idxs,np.flip(fragment_idxs,axis=1)])
    bonded_fragments = bonded_fragmenter.fragment(self.mol_container)
    bonded_idxs = np.array([fragment.atom_selection for fragment in bonded_fragments])
    bonded_idxs = np.vstack([bonded_idxs,np.flip(bonded_idxs,axis=1)])
    if nonbonded_fragmenter:
      nonbonded_fragments = nonbonded_fragmenter.fragment(self.mol_container)
      nonbonded_idxs = np.array([fragment.atom_selection for fragment in nonbonded_fragments])
      nonbonded_idxs = np.vstack([nonbonded_idxs,np.flip(nonbonded_idxs,axis=1)])
    else:
      nonbonded_idxs = None
    self.heterograph = build_fragment_heterograph(atom_graph = self.atom_graph,
                                            atom_idxs = atom_idxs,
                                            bonded_idxs=bonded_idxs,
                                            nonbonded_idxs=nonbonded_idxs,
                                            frag_idxs=fragment_idxs,
                                            frag_name = frag_name,
                                            node_name = node_name)
    
    labels = np.array([fragment_labeler(fragment) for fragment in fragments])
    #labels = np.concatenate([labels,labels])
    labels = labels[:,np.newaxis]
    self.heterograph.nodes[frag_name].data[label_ref_name] = torch.tensor(labels,dtype=torch.get_default_dtype())
    

    # conf = rdkit_mol.GetConformer()
    # angles_rad = []
    # for idx1,idx2,idx3 in idxs["n3"]:
    #   angle_rad = Chem.rdMolTransforms.GetAngleRad(conf,int(idx1),int(idx2),int(idx3))
    #   angles_rad.append(angle_rad)
    # angles_rad = np.array(angles_rad)[:,np.newaxis]
    # self.heterograph.nodes[frag_name].data["ref"] = torch.tensor(angles_rad,dtype=torch.get_default_dtype())
    
  @property
  def rdkit_mol(self):
    return self.mol_container.rdkit_mol