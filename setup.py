#!/usr/bin/env python

from setuptools import setup

setup(name='persimmon',
      version='0.0.1',
      description='Python implementation persistent images representation of persistence diagrams.',
      long_description=""" :D """,
      author='Nathaniel Saul',
      author_email='nat@saulgill.com',
      url='https://github.com/sauln/persistence-images',
      license='MIT',
      packages=['persimmon'],
      include_package_data=True,
      install_requires=[
        'numpy'
      ],
      python_requires='>=2.7,!=3.1,!=3.2,!=3.3',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
      ],
      keywords='persistent homology, persistence images, persistence diagrams, topology data analysis, algebraic topology, unsupervised learning'
     )
