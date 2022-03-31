import numpy as np

from phenixml.fragmentation.fragments import MolContainer, Fragment

class FragmentFeaturizerBase:
    """
    Base class to generate feature vectors for fragment objects


    To make a new fragmenter, subclass this object and write a custom
    featurize method which takes a fragment as an 
    argument, and returns a numpy feature vector

    def _featurize(self,fragment):

    # custom code

    return feature_vector

    """
    
    @staticmethod
    def _featurize_fragment_list(instance,fragments,progress=True,**kwargs):
        features = []
        if progress:
            import tqdm
            for fragment in tqdm.tqdm(fragments):
                feature = instance.featurize(fragment,**kwargs)
                features.append(feature)
        else:
            for fragment in fragments:
                feature = instance.featurize(fragment,**kwargs)
                features.append(feature)
        return np.array(features)
    
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