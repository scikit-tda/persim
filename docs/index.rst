.. Persim documentation master file, created by
   sphinx-quickstart on Tue Jul 24 23:08:15 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|PyPI version| |Build Status| |Codecov| |License: MIT|

Persim
==================================


Persim is a Python implementation of Persistence Images, as first introduced in `this paper <https://arxiv.org/abs/1507.06217>`_.

It is designed to interface with `Ripser <https://github.com/sauln/ripser>`_, though works fine for any persistence diagrams.


Setup
--------

Currently, the only option is to install the library from source:

.. code:: Bash

    pip install persim


Usage
-------

First, construct a diagram. In this example, we will use `Ripser <https://github.com/sauln/ripser>`_.

.. code:: Python

    import numpy as np
    from ripser import Rips

    data = np.random.random((100,2))
    rips = Rips()
    dgm = rips.fit_transform(data)
    diagram = dgm[1] # Just diagram for H1

.. image:: images/data-and-pd.png

Then from this diagram, we construct the persistence image

.. code:: Python

    from persim import PersImage

    pim = PersImage(diagram)
    img = pim.transform()
    pim.show(img)


.. image:: images/pers-im-h1.png



.. toctree::
    :maxdepth: 2
    :caption: Background

    about
    Basic Usage

.. toctree::
    :maxdepth: 2
    :caption: Tutorials

    Classification with persistence images


.. toctree::
    :maxdepth: 2
    :caption: API Reference
    
    reference






Reference
===========

.. automodule:: persim
    :members:

.. autoclass:: PersImage
    :members:
    :undoc-members:

.. toctree::
    :maxdepth: 2


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



.. |PyPI version| image:: https://badge.fury.io/py/persim.svg
   :target: https://badge.fury.io/py/persim
.. |Build Status| image:: https://travis-ci.org/scikit-tda/persim.svg?branch=master
   :target: https://travis-ci.org/scikit-tda/persim
.. |Codecov| image:: https://codecov.io/gh/scikit-tda/persim/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/scikit-tda/persim
.. |License: MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT)



