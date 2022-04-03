import torch

def to_np(tensor):
  return tensor.detach().cpu().numpy()