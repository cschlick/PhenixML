import numpy as np
import pynndescent
from pynndescent import NNDescent
import warnings


class FragmentKNN:
    def __init__(self,fragments,features,metric="cosine"):
      assert len(fragments)==len(features), "Must pass fragments,features of same length"
      with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          self.fragments = fragments
          self.features = features
          self.metric = metric
          self.index = NNDescent(features,self.metric)
          # the first query often takes a long time,
          # do it as part of building the index
          _, _ = self.index.query(features[0][np.newaxis,:],k=1)
        
    def query(self,features,k=10,return_ind=False,return_dist=False):
        assert isinstance(features,np.ndarray), "Pass a numpy feature array"
        if features.ndim <2:
            features = features[np.newaxis,:]
        
        inds,dists = self.index.query(features,k=k)
        frags = [[self.fragments[i] for i in ind] for ind in inds]
        if len(inds)==1:
            inds,dists = inds[0],dists[0]
            frags = frags[0]
        
        if return_ind and not return_dist:
            return frags, inds
        if return_dist:
            return frags, inds, dists
        else:
            return frags
        
            
