[![Build Status](https://travis-ci.org/sauln/persim.svg?branch=master)](https://travis-ci.org/sauln/persim)
[![codecov](https://codecov.io/gh/sauln/persim/branch/master/graph/badge.svg)](https://codecov.io/gh/sauln/persim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Persim <img align="right" width="40" height="40" src="https://imgur.com/8p6VwFm.jpg">

Persim is a Python implementation of Persistence Images as first introduced in [https://arxiv.org/abs/1507.06217](https://arxiv.org/abs/1507.06217).

It is designed to interface with [Ripser](https://github.com/sauln/ripser), though any persistence diagram should work fine.

# Setup

Currently, the only option is to install the library from source:

```
pip install persim
```


# Usage

First, construct a diagram. In this example, we will use [Ripser](https://github.com/sauln/ripser).

``` Python
import numpy as np
from ripser import Rips

data = np.random.random((100,2))
rips = Rips()
dgm = rips.fit_transform(data)
diagram = dgm[1] # Just diagram for H1
```

Then from this diagram, we construct the persistence image

``` Python
from persim import PersImage

pim = PersImage(diagram)
img = pim.transform()
pim.show(img)
```


# TODO

- The API needs a little work, not quite sklearn compliant. Please do offer any suggestions.
- Implement more varieties of weighting and kernel functions.
- Build tests.

