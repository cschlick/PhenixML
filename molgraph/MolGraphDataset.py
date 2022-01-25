import dgl

class AtomMolGraphDataset:
  
    def __init__(self,mgraphs):

        self.mgraphs = mgraphs
     
    def split(self, partition):
        """Split the dataset according to some partition.

        Parameters
        ----------
        partition : sequence of integers or floats

        """
        n_data = len(self)
        partition = [int(n_data * x / sum(partition)) for x in partition]
        ds = []
        idx = 0
        for p_size in partition:
            ds.append(self[idx : idx + p_size])
            idx += p_size

        return ds
    @property
    def atom_graphs(self):
      return [mgraph.atom_graph for mgraph in self.mgraphs]
    
          
    @property
    def full_atom_graph(self):
      return dgl.batch(self.atom_graphs)
    
       
    def __len__(self):
      return len(self.mgraphs)

    def __getitem__(self, item):

        return self.mgraphs[item]
      
    def batches(self,batch_size=1000,shuffle=True):
      # return a list of batched heterographs
      import random
      
      mgraphs = self.mgraphs.copy()
      if shuffle:
        random.shuffle(mgraphs)
      
      def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield dgl.batch([mg.atom_graph for mg in lst[i:i + n]])
      
      return chunks(mgraphs,batch_size)

class MolGraphDataset:
  
   
    def __init__(self,mgraphs):

        self.mgraphs = mgraphs
     
    def split(self, partition):
        """Split the dataset according to some partition.

        Parameters
        ----------
        partition : sequence of integers or floats

        """
        n_data = len(self)
        partition = [int(n_data * x / sum(partition)) for x in partition]
        ds = []
        idx = 0
        for p_size in partition:
            ds.append(self[idx : idx + p_size])
            idx += p_size

        return ds
    @property
    def homographs(self):
      return [mgraph.homograph for mgraph in self.mgraphs]
    
    @property
    def heterographs(self):
      return [mgraph.heterograph for mgraph in self.mgraphs]
       
    @property
    def full_homograph(self):
      return dgl.batch(self.homographs)
    
    @property
    def full_heterograph(self):
      return dgl.batch(self.heterographs)
    
    
    @property
    def rdmols(self):
      return (mgraph.rdmol for mgraph in self.mgraphs)
    
    @property
    def atoms(self):
      return (atom for rdmol in self.rdmols for atom in rdmol.GetAtoms())
    
    @property
    def bonds(self):
      return (bond for rdmol in self.rdmols for bond in rdmol.GetBonds())
    
    def __len__(self):
      return len(self.mgraphs)

    def __getitem__(self, item):

        return self.mgraphs[item]
      
    def batches(self,batch_size=1000,shuffle=True):
      # return a list of batched heterographs
      import random
      
      mgraphs = self.mgraphs.copy()
      if shuffle:
        random.shuffle(mgraphs)
      
      def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield dgl.batch([mg.heterograph for mg in lst[i:i + n]])
      
      return chunks(mgraphs,batch_size)