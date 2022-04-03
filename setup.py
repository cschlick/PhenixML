from distutils.core import setup

setup(
    name='PhenixML',
    version='0.4dev',
    packages=['phenixml','phenixml.fragments',"phenixml.featurizers","phenixml.fragmentation","phenixml.visualization","phenixml.graphs","phenixml.models",
             "phenixml.labelers","phenixml.utils"],
    license=open('LICENSE.txt').read(),
    long_description=open('README.md').read(),
)
