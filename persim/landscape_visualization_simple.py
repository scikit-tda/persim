"""
Visualization methods for plotting persistence landscapes.

"""

import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from .PersLandscape import PersLandscape
from .PersLandscapeExact import PersLandscapeExact
from .PersLandscapeGrid import PersLandscapeGrid
from operator import itemgetter
from matplotlib import cm

# TODO: Use styles instead of colormaps directly? Or both?

# mpl.rcParams['text.usetex'] = True

def plot_landscape_simple(landscape: PersLandscape,
                   padding = 0.1, 
                   num_steps = 1000,
                   title = None,
                   ):
    """
    plot landscape functions. 
    """
    if isinstance(landscape, PersLandscapeExact):
        return plot_landscape_exact_simple(landscape=landscape, title=title)

    if isinstance(landscape, PersLandscapeGrid):
        return plot_landscape_grid_simple(landscape=landscape, padding=padding, num_steps=num_steps, title=title)

def plot_landscape_exact_simple(landscape: PersLandscapeExact,
                   alpha = 1,
                   title = None):
    """
    A simple plot of the persistence landscape. This is a faster plotting utility than the standard plotting, but is recommended for smaller landscapes for ease of visualization.
    
    
    Parameters
    ----------
    alpha, default 1
        transparency of shading
        
    """
    fig = plt.figure()

    landscape.compute_landscape()

    # itemgetter index selects which entry to take max/min wrt.
    # the hanging [0] or [1] takes that entry.
    crit_pairs = list(itertools.chain.from_iterable(landscape.critical_pairs))

    min_crit_pt = min(crit_pairs, key=itemgetter(0))[0] # smallest birth time

    max_crit_pt = max(crit_pairs, key=itemgetter(0))[0] # largest death time

    max_crit_val = max(crit_pairs,key=itemgetter(1))[1] # largest peak of landscape

    min_crit_val = min(crit_pairs, key=itemgetter(1))[1] # smallest peak of landscape

    # for each landscape function
    for depth, l in enumerate(landscape):
        ls = np.array(l)

        plt.plot(ls[:,0], ls[:,1], label=f'$\lambda_{{{depth}}}$', alpha=alpha)

    plt.legend()
    if title: plt.title(title)
    plt.show()

def plot_landscape_grid_simple(landscape: PersLandscapeGrid,
                   alpha = 1,
                   padding = 0.1,
                   num_steps = 1000,
                   smooth = True,
                   title = None):
    """
    A simple plot of the persistence landscape. This is a faster plotting utility than the standard plotting, but is recommended for smaller landscapes for ease of visualization.
    
    Parameters
    ----------
    alpha, default 1
        transparency of shading
        
    """
    fig = plt.figure()

    landscape.compute_landscape()

    # TODO: RE the following line: is this better than np.concatenate?
    #       There is probably an even better way without creating an intermediary.
    _vals = list(itertools.chain.from_iterable(landscape.values)) 
    min_val = min(_vals) 
    max_val = max(_vals) 

    # for each landscape function
    for depth, l in enumerate(landscape):

        # instantiate depth-specific domain
        domain = np.linspace(landscape.start-padding*0.1, landscape.stop+padding*0.1, num=len(l))

        plt.plot(domain, l, label=f"$\lambda_{{{depth}}}$", alpha=alpha)

    
    plt.legend()
    if title: plt.title(title)
    plt.show()

