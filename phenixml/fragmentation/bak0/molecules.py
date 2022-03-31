from phenixml.fragments.fragments import Fragment

from rdkit import Chem

class MoleculeFragmenter:
    def __init__(self):
        pass

    
    def fragment(self,model_container):
        
        
        # rdkit method
        frag_indices = Chem.GetMolFrags(model_container.rdkit_mol, asMols=False,sanitizeFrags=False)
        fragments = [Fragment(model_container,atom_selection=inds) for inds in frag_indices]
        return fragments