import torch

  
  
class FeatureModel(torch.nn.Module):
  """
  Simple MLP model meant for chemical feature vectors
  """
  def __init__(self,in_feats,hid_feats,out_feats,n_hid_layers=3):
    super(FeatureModel, self).__init__()
    self.layers = []

    # input
    f_in = torch.nn.Sequential(
            torch.nn.Linear(in_feats, hid_feats), torch.nn.Tanh()
        )
    self.layers.append(f_in)
    
    # hidden layers
    for i in range(n_hid_layers):
      layer = torch.nn.Sequential(torch.nn.Linear(hid_feats,hid_feats),torch.nn.ReLU())
      self.layers.append(layer)

    #output
    f_out = torch.nn.Sequential(
            torch.nn.Linear(hid_feats, out_feats))
    self.layers.append(f_out)

    setattr(self,"input_layer_"+str(i),self.layers[0])
    for i,layer in enumerate(self.layers[1:-1]):
      setattr(self,"hidden_layer_"+str(i),layer)
    setattr(self,"output_layer_"+str(i),self.layers[-1])

  def forward(self,x):
    for layer in self.layers:
      x = layer(x)
    return x