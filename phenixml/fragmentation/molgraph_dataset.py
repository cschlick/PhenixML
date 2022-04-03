import dgl
import itertools
from phenixml.fragments.fragments import MolContainer
from phenixml.graphs.molgraph import MolGraph
import random
import numpy as np

class MolGraphDataset:
    """
    A collection of MolGraph objects. Intended to be used for train/test splits,
    batching, and perhaps eventually writing to disk.
    """
    def __init__(self,molgraphs):

        self.molgraphs = molgraphs
    

    
    @property
    def _heterographs(self):
      return [molgraph.heterograph for molgraph in self.molgraphs]
       
    
    @property
    def heterograph(self):
      return dgl.batch(self._heterographs)
    
    
    @property
    def fragments(self):
      return list(itertools.chain.from_iterable([molgraph.fragments for molgraph in self.molgraphs]))
    
    def __len__(self):
      return len(self.molgraphs)

    def __getitem__(self, item):

        return self.molgraphs[item]
      
    def batches(self,batch_size=1000,n_batches=None,shuffle=True):
      # return a list of batched heterographs
      
      if n_batches is not None:
        batch_size = int(len(self)/n_batches)
        
      if batch_size > len(self):
        batch_size = len(self)
        
      mgraphs = self.molgraphs.copy()
      if shuffle:
        random.shuffle(mgraphs)
      
      def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield dgl.batch([mg.heterograph for mg in lst[i:i + n]])

      return chunks(mgraphs,batch_size)
    
    
    def train_test_split(self, test_fraction=0.2):        
      space = list(np.arange(len(self)))
      k = int(len(self)*test_fraction)
      test = random.sample(space,k=k)
      train = set(space)-set(test)
      train = self.__class__([self[i] for i in train])
      test = self.__class__([self[i] for i in test])
      return train,test