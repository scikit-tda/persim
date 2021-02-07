#!/usr/bin/env python

from setuptools import setup


import re
VERSIONFILE="persim/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open('README.md') as f:
    long_description = f.read()

setup(name='persim',
      version=verstr,
      description='Distances and representations of persistence diagrams',
      long_description=long_description,
      long_description_content_type="text/markdown",	
      author='Nathaniel Saul, Chris Tralie, Francis Motta, Michael Catanzaro, Gabrielle Angeloro, Calder Sheagren',
      author_email='nat@riverasaul.com, chris.tralie@gmail.com, francis.c.motta@gmail.com, catanzaromj@gmail.com, gabrielleangeloro@gmail.com, caldersheagren@gmail.com',
      url='https://persim.scikit-tda.org',
      license='MIT',
      packages=['persim'],
      include_package_data=True,
      install_requires=[
        'scikit-learn',
        'numpy',
        'matplotlib',
        'scipy',
        'hopcroftkarp',
        'deprecated',
        'joblib'
      ],
      extras_require={ # use `pip install -e ".[testing]"``
        'testing': [
          'pytest',
          'pytest-cov' 
        ],
        'docs': [ # `pip install -e ".[docs]"``
          'sktda_docs_config'
        ]
      },
      python_requires='>=3.6',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
      ],
      keywords='persistent homology, persistence images, persistence diagrams, topological data analysis, algebraic topology, unsupervised learning, supervised learning, machine learning, sliced wasserstein distance, bottleneck distance'
     )
