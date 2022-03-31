import torch
import dgl
import numpy as np

from molgraph.esp_fragments.sequential import _Sequential

class JanossyReadout(torch.nn.Module):

  def __init__(
        self,
        config,
        in_features,
        out_features={"eq":3},
        out_features_dimensions=-1,
        atom_node_name = "n1",
        fragment_name = "n3",
        fragment_size = 3,
        pool=torch.add,
    ):
    super(JanossyReadout, self).__init__()

    # setup
    out_features = {fragment_size:out_features}
    self.atom_node_name = atom_node_name
    self.fragment_name = fragment_name
    self.fragment_size = fragment_size

    # if users specify out features as lists,
    # assume dimensions to be all zero
    for level in out_features.keys():
        if isinstance(out_features[level], list):
            out_features[level] = dict(
                zip(out_features[level], [1 for _ in out_features[level]])
            )

    # bookkeeping
    self.out_features = out_features
    self.levels = [key for key in out_features.keys() if key != 1]
    self.pool = pool
    assert len(self.levels)==1 and self.levels[0]==self.fragment_size, "Malformed input. Only supports one fragment size"
    
    # get output features
    mid_features = [x for x in config if isinstance(x, int)][-1]

    # set up networks
    for level in self.levels:

        # set up individual sequential networks
        setattr(
            self,
            "sequential_%s" % level,
            _Sequential(
                in_features=in_features * level,
                config=config,
                layer=torch.nn.Linear,
            ),
        )

        for feature, dimension in self.out_features[level].items():
            setattr(
                self,
                "f_out_%s_to_%s" % (level, feature),
                torch.nn.Linear(
                    mid_features,
                    dimension,
                ),
            )

    if 1 not in self.out_features:
        return

    # atom level
    self.sequential_1 = _Sequential(
        in_features=in_features, config=config, layer=torch.nn.Linear
    )

    for feature, dimension in self.out_features[1].items():
        setattr(
            self,
            "f_out_1_to_%s" % feature,
            torch.nn.Linear(
                mid_features,
                dimension,
            ),
        )

  def forward(self, g):
      """Forward pass.
      Parameters
      ----------
      g : dgl.DGLHeteroGraph,
          input graph.
      """
      import dgl

      # copy
      # this sets the necessary fields for the higher level nodes:

      # g.nodes["n2"].data["h0"] 
      # g.nodes["n2"].data["h1"] 


      g.multi_update_all(
                  {
                      "%s_as_%s_in_%s"
                      % (self.atom_node_name,relationship_idx, self.fragment_name): (
                          dgl.function.copy_src("h", "m%s" % relationship_idx),
                          dgl.function.mean(
                              "m%s" % relationship_idx, "h%s" % relationship_idx
                          ),
                      )
                      for big_idx in self.levels
                      for relationship_idx in range(big_idx)
                  },
                  cross_reducer="sum",
              )




      # use the new (copied) hidden vectors to finish the forward pass on the higher level nodes.
      # sets g.nodes["n2"].data["log_coefficients"] 

      for big_idx in self.levels:

        # if g.number_of_nodes("n%s" % big_idx) == 0:
        #     continue

        g.apply_nodes(
            func=lambda nodes: {
                feature: getattr(
                    self, "f_out_%s_to_%s" % (big_idx, feature) # get the output layer for this level/feature
                )(                                              # call it, using as input:
                    self.pool(                                  # pooled using sum:
                        getattr(self, "sequential_%s" % big_idx)( # Cal the sequential intermediate layers, using
                            None,
                            torch.cat(                            # concatenated feature vector over node representations torch.cat([h0,h1])
                                [
                                    nodes.data["h%s" % relationship_idx]
                                    for relationship_idx in range(big_idx)
                                ],
                                dim=1,
                            ),
                        ),
                        getattr(self, "sequential_%s" % big_idx)( # Call sequential again with a "backwards" order
                            None,
                            torch.cat(                            #torch.cat([h1,h0])
                                [
                                    nodes.data["h%s" % relationship_idx]
                                    for relationship_idx in range(
                                        big_idx - 1, -1, -1
                                    )
                                ],
                                dim=1,
                            ),
                        ),
                    ),
                )
                for feature in self.out_features[big_idx].keys()
            },
            ntype=self.fragment_name,
        )
      return g