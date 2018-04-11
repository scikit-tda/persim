# persimmon
Python implementation of Persistence Images as first introduced in [https://arxiv.org/abs/1507.06217](https://arxiv.org/abs/1507.06217).

It is designed to interface with [Ripser](https://github.com/sauln/ripser), though any persistence diagram should work fine.

# Setup

Currently, the only option is to install the library from source:

```
git clone https://github.com/sauln/persimmon
cd persimmon
pip install .
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
from persimmon import PersImage

pim = PersImage(diagram)
img = pim.transform()
pim.show(img)
```


# TODO

- The API needs a little work, not quite sklearn compliant. Please do offer any suggestions.
- Implement more varieties of weighting and kernel functions.
- Build tests.

