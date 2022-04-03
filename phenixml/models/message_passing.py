import torch
import dgl
import numpy as np
from phenixml.models.sequential import _Sequential

class MessagePassing(torch.nn.Module):
    """Sequential neural network with input layers.
    Parameters
    ----------
    layer : torch.nn.Module
        DGL graph convolution layers.
    config : List
        A sequence of numbers (for units) and strings (for activation functions)
        denoting the configuration of the sequential model.
    feature_units : int(default=117)
        The number of input channels.
    Methods
    -------
    forward(g, x)
        Forward pass.
    """
    

    
    def __init__(
        self,
        config,
        layer= None,
        feature_units=117,
        input_units=128,
        atom_node_name = "atom",
        fragment_name = "fragment",
        edge_types = ["bonded"],
        model_kwargs={},
    ):
        super(MessagePassing, self).__init__()
        
        
        # setup
        self.atom_node_name = atom_node_name
        self.fragment_name = fragment_name
        self.edge_types = edge_types
        if layer is None:
           layer = lambda in_feats,out_feats: dgl.nn.pytorch.conv.sageconv.SAGEConv(in_feats,out_feats,"mean",bias=True) # dgl sageconv layer
        
        # initial featurization
        self.f_in = torch.nn.Sequential(
            torch.nn.Linear(feature_units, input_units), torch.nn.Tanh()
        )

        self._sequential = _Sequential(
            layer, config, in_features=input_units, model_kwargs=model_kwargs
        )

    def _forward(self, g, x):
        for exe in self.exes:
            if exe.startswith("d"):
                x = getattr(self, exe)(g, x)
            else:
                x = getattr(self, exe)(x)

        return x

    def forward(self, g, x=None):

        import dgl
        # get homogeneous subgraph
 
        g_ = dgl.to_homogeneous(g.edge_type_subgraph(["%s_%s_%s" % (self.atom_node_name,edge_type,self.atom_node_name) for edge_type in self.edge_types]))

        if x is None:
            # get node attributes
            x = g.nodes[self.atom_node_name].data["h0"]
            x = self.f_in(x)

        # message passing on homo graph
        x = self._sequential(g_, x)

        # put attribute back in the graph
        g.nodes[self.atom_node_name].data["h"] = x

        return g