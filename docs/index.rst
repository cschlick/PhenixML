.. PhenixML documentation master file, created by
   sphinx-quickstart on Thu Mar 31 21:50:13 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PhenixML
===========================================================================

PhenixML is a project built to support the `Phenix <https://phenix-online.org>`_ suite of tools for sturcture refinement of x-ray crystallography and cryo-EM models. Many aspects of Phenix could benefit from regression/classification tasks on molecules and molecular fragments. This framework aims to make these tasks easy.

The fundamental objects are MolContainers and Fragments. A MolContainer stores molecular data structures in a pre-existing format. A Fragment is a selection on a MolContainer. Featurization, labeling, and visualization functionality operate on Fragments. The basic workflow would be:
1. Put molecular data structures into a MolContainer
2. Write a Fragmenter function to generate fragments
3. Pass the fragments to Featurizer and Labeler functions
4. Perform regression/classification on the resulting features/labels

**Installation**
```
pip install git+https://github.com/cschlick/PhenixML.git
```
To get the cctbx functionality use conda: 
```
conda install -c conda-forge cctbx-base
```

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples/index
