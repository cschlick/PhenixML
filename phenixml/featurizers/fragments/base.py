import numpy as np
from multiprocessing import Pool
from contextlib import closing
from tqdm.notebook import tqdm
from phenixml.fragmentation.fragments import MolContainer, Fragment

class FragmentFeaturizerBase:
    """
    Base class to generate feature vectors for fragment objects


    To make a new featurizer, subclass this object and write a custom
    featurize method which takes a fragment as an 
    argument, and returns a numpy feature vector

    def _featurize(self,fragment):

      # custom code

      return feature_vector

    """
    @staticmethod
    def _featurize_fragment_list(instance,fragments,disable_progress=False,nproc=1,**kwargs):
         
      worker = instance.featurize
      work = fragments
      results = []
      with closing(Pool(processes=nproc)) as pool:
        for result in tqdm(pool.map(worker, work), total=len(work)):
            results.append(result)
        pool.terminate()
      
      return np.array(results)
    
    def __call__(self,fragment,**kwargs):
        return self.featurize(fragment,**kwargs)
    
    def featurize(self,fragment,**kwargs):
        if isinstance(fragment,list):
            return self._featurize_fragment_list(self,fragment,**kwargs)
        if isinstance(fragment,MolContainer):
            fragment = fragment.full_fragment
        
        assert isinstance(fragment,Fragment), "Pass a Fragment object"
        
        return self._featurize(fragment,**kwargs)
    
    def _featurize(self,obj,**kwargs):
        raise NotImplementedError