from distutils.core import setup

setup(
    name='PhenixML',
    version='0.4dev',
    packages=['phenixml','phenixml.fragments',"phenixml.featurizers","phenixml.fragmentation","phenixml.visualization","phenixml.graphs","phenixml.models",
             "phenixml.labelers","phenixml.utils"],
    install_requires=["rdkit-pypi","torch","scikit-learn","numpy","scipy","matplotlib"],
    license=open('LICENSE.txt').read(),
    long_description=open('README.md').read(),
)
