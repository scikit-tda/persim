# -*- coding: utf-8 -*-
"""
    Implementation of the modified Gromov–Hausdorff (mGH) distance
    between compact metric spaces induced by unweighted graphs. This
    code complements the results from "Efficient estimation of a
    Gromov–Hausdorff distance between unweighted graphs" by V. Oles et
    al. (https://arxiv.org/pdf/1909.09772). The mGH distance was first
    defined in "Some properties of Gromov–Hausdorff distances" by F.
    Mémoli (Discrete & Computational Geometry, 2012).

    Author: Vladyslav Oles

    ===================================================================

    Usage examples:

    1) Estimating the mGH distance between 4-clique and single-vertex
    graph from their adjacency matrices. Note that it suffices to fill
    only the upper triangle of an adjacency matrix.

    >>> AG = [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
    >>> AH = [[0]]
    >>> lb, ub = gromov_hausdorff(AG, AH)
    >>> lb, ub
    (0.5, 0.5)

    2) Estimating the mGH distance between cycle graphs of length 2 and
    5 from their adjacency matrices. Note that the adjacency matrices
    can be given in both dense and sparse SciPy formats.

    >>> AI = np.array([[0, 1], [0, 0]])
    >>> AJ = sps.csr_matrix(([1] * 5, ([0, 0, 1, 2, 3], [1, 4, 2, 3, 4])), shape=(5, 5))
    >>> lb, ub = gromov_hausdorff(AI, AJ)
    >>> lb, ub
    (0.5, 1.0)

    3) Estimating all pairwise mGH distances between multiple graphs
    from their adjacency matrices as an iterable.

    >>> As = [AG, AH, AI, AJ]
    >>> lbs, ubs = gromov_hausdorff(As)
    >>> lbs
    array([[0. , 0.5, 0.5, 0.5],
           [0.5, 0. , 0.5, 1. ],
           [0.5, 0.5, 0. , 0.5],
           [0.5, 1. , 0.5, 0. ]])
    >>> ubs
    array([[0. , 0.5, 0.5, 0.5],
           [0.5, 0. , 0.5, 1. ],
           [0.5, 0.5, 0. , 1. ],
           [0.5, 1. , 1. , 0. ]])

    ===================================================================

    Notations:

    |X| denotes the number of elements in set X.

    X → Y denotes the set of all mappings of set X into set Y.

    V(G) denotes vertex set of graph G.

    mGH(X, Y) denotes the modified Gromov–Hausdorff distance between
    compact metric spaces X and Y.

    row_i(A) denotes the i-th row of matrix A.

    PSPS^n(A) denotes the set of all permutation similarities of
    all n×n principal submatrices of square matrix A.

    PSPS^n_{i←j}(A) denotes the set of all permutation similarities of
    all n×n principal submatrices of square matrix A whose i-th row is
    comprised of the entries in row_j(A).

    ===================================================================

    Glossary:

    Distance matrix of metric space X is a |X|×|X| matrix whose
    (i, j)-th entry holds the distance between i-th and j-th points of
    X. By the properties of a metric, distance matrices are symmetric
    and non-negative, their diagonal entries are 0 and off-diagonal
    entries are positive.

    Curvature is a generalization of distance matrix that allows
    repetitions in the underlying points of a metric space. Curvature
    of an n-tuple of points from metric space X is an n×n matrix whose
    (i, j)-th entry holds the distance between the points from i-th and
    j-th positions of the tuple. Since these points need not be
    distinct, the off-diagonal entries of a curvature can equal 0.

    n-th curvature set of metric space X is the set of all curvatures
    of X that are of size n×n.

    d-bounded curvature for some d > 0 is a curvature whose
    off-diagonal entries are all ≥ d.

    Positive-bounded curvature is a curvature whose off-diagonal
    entries are all positive, i.e. the points in the underlying tuple
    are distinct. Equivalently, positive-bounded curvatures are
    distance matrices on the subsets of a metric space.
"""
import numpy as np
import warnings
import scipy.sparse as sps
from scipy.sparse.csgraph import shortest_path, connected_components


__all__ = ["gromov_hausdorff"]


# To sample √|X| * log (|X| + 1) mappings from X → Y by default.
DEFAULT_MAPPING_SAMPLE_SIZE_ORDER = np.array([.5, 1])


def gromov_hausdorff(
        AG, AH=None, mapping_sample_size_order=DEFAULT_MAPPING_SAMPLE_SIZE_ORDER):
    """
    Estimate the mGH distance between simple unweighted graphs,
    represented as compact metric spaces based on their shortest
    path lengths.

    Parameters
    -----------
    AG: (N,N) np.array
        (Sparse) adjacency matrix of graph G with N vertices, or an iterable of
        adjacency matrices if AH=None.
    AH: (M,M) np.array
        (Sparse) adjacency matrix of graph H with M vertices, or None.
    mapping_sample_size_order: (2,) np.array 
        Parameter that regulates the number of mappings to sample when
        tightening upper bound of the mGH distance.

    Returns
    --------
    lb: float
        Lower bound of the mGH distance, or a square matrix holding
        lower bounds of pairwise mGH distances if AH=None.
    ub: float
        Upper bound of the mGH distance, or a square matrix holding
        upper bounds of pairwise mGH distances if AH=None.
    """
    # Form iterable with adjacency matrices.
    if AH is None:
        if len(AG) < 2:
            raise ValueError("'estimate_between_unweighted_graphs' needs at least"
                             "2 graphs to discriminate")
        As = AG
    else:
        As = (AG, AH)

    N = len(As)
    # Find lower and upper bounds of each pairwise mGH distance between
    # the graphs.
    lbs = np.zeros((N, N))
    ubs = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            # Transform adjacency matrices of a pair of graphs to
            # distance matrices.
            DX = make_distance_matrix_from_adjacency_matrix(As[i])
            DY = make_distance_matrix_from_adjacency_matrix(As[j])
            # Find lower and upper bounds of the mGH distance between
            # the pair of graphs.
            lbs[i, j], ubs[i, j] = estimate(
                DX, DY, mapping_sample_size_order=mapping_sample_size_order)

    if AH is None:
        # Symmetrize matrices with lower and upper bounds of pairwise
        # mGH distances between the graphs.
        lower_triangle_indices = np.tril_indices(N, -1)
        lbs[lower_triangle_indices] = lbs.T[lower_triangle_indices]
        ubs[lower_triangle_indices] = ubs.T[lower_triangle_indices]

        return lbs, ubs
    else:
        return lbs[0, 1], ubs[0, 1]


def make_distance_matrix_from_adjacency_matrix(AG):
    """
    Represent simple unweighted graph as compact metric space (with
    integer distances) based on its shortest path lengths.

    Parameters
    -----------
    AG: (N,N) np.array
        (Sparse) adjacency matrix of simple unweighted graph G with N vertices.

    Returns
    --------
    DG: (N,N)  np.array
        (Dense) distance matrix of the compact metric space
        representation of G based on its shortest path lengths.
    """
    # Convert adjacency matrix to SciPy format if needed.
    if not sps.issparse(AG) and not isinstance(AG, np.ndarray):
        AG = np.asarray(AG)

    # Compile distance matrix of the graph based on its shortest path
    # lengths.
    DG = shortest_path(AG, directed=False, unweighted=True)
    # Ensure compactness of metric space, represented by distance
    # matrix.
    if np.any(np.isinf(DG)):
        warnings.warn("disconnected graph is approximated by its largest connected component")
        # Extract largest connected component of the graph.
        _, components_by_vertex = connected_components(AG, directed=False)
        components, component_sizes = np.unique(components_by_vertex, return_counts=True)
        largest_component = components[np.argmax(component_sizes)]
        DG = DG[components_by_vertex == largest_component]

    # Cast distance matrix to optimal integer type.
    DG = cast_distance_matrix_to_optimal_int_type(DG)

    return DG


def cast_distance_matrix_to_optimal_int_type(DX):
    """
    Given a metric space X induced by simple unweighted graph,
    cast its distance matrix to the smallest signed integer type,
    sufficient to hold all its entries.

    Parameters
    -----------
    DX: np.array (|X|×|X|)
        Distance matrix of X.

    Returns
    --------
    DX: np.array (|X|×|X|)
        Distance matrix of X, cast to optimal type.
    """
    max_distance = np.max(DX)
    # Type is signed integer to allow subtractions.
    optimal_int_type = determine_optimal_int_type(max_distance)
    DX = DX.astype(optimal_int_type)

    return DX


def determine_optimal_int_type(value):
    """
    Determine smallest signed integer type sufficient to hold a value.

    Parameters
    -----------
    value: non-negative integer

    Returns
    --------
    optimal_int_type: np.dtype
        Optimal signed integer type to hold the value.
    """
    feasible_int_types = (int_type for int_type in [np.int8, np.int16, np.int32, np.int64]
                          if value <= np.iinfo(int_type).max)
    try:
        optimal_int_type = next(feasible_int_types)
    except StopIteration:
        raise ValueError("value {} too large to be stored as unsigned integer")

    return optimal_int_type


def estimate(DX, DY, mapping_sample_size_order=DEFAULT_MAPPING_SAMPLE_SIZE_ORDER):
    """
    For X, Y metric spaces induced by simple unweighted graphs, find
    lower and upper bounds of mGH(X, Y).

    Parameters
    ----------
    DX: np.array (|X|×|X|)
        (Integer) distance matrix of X.
    DY: np.array (|Y|×|Y|)
        (Integer) distance matrix of Y.
    mapping_sample_size_order: np.array (2)
        Parameter that regulates the number of mappings to sample when
        tightening upper bound of mGH(X, Y).

    Returns
    --------
    lb: float
        Lower bound of mGH(X, Y).
    ub: float
        Upper bound of mGH(X, Y).
    """
    # Ensure distance matrices are of integer type.
    if not np.issubdtype(DX.dtype, np.integer) or not np.issubdtype(DY.dtype, np.integer):
        raise ValueError("non-integer metrics are not yet supported")
    # Cast distance matrices to signed integer type to allow
    # subtractions.
    if np.issubdtype(DX.dtype, np.uint):
        DX = cast_distance_matrix_to_optimal_int_type(DX)
    if np.issubdtype(DY.dtype, np.uint):
        DY = cast_distance_matrix_to_optimal_int_type(DY)
    # Estimate mGH(X, Y).
    double_lb = find_lb(DX, DY)
    double_ub = find_ub(
        DX, DY, mapping_sample_size_order=mapping_sample_size_order, double_lb=double_lb)

    return 0.5 * double_lb, 0.5 * double_ub


def find_lb(DX, DY):
    """
    For X, Y metric spaces induced by simple unweighted graphs, find
    lower bound of mGH(X, Y).

    Parameters
    ----------
    DX: np.array (|X|×|X|)
        (Integer) distance matrix of X.
    DY: np.array (|Y|×|Y|)
        (Integer) distance matrix of Y.

    Returns
    --------
    double_lb: float
        Lower bound of 2*mGH(X, Y).
    """
    diam_X = np.max(DX)
    diam_Y = np.max(DY)
    max_diam = max(diam_X, diam_Y)
    # Obtain trivial lower bound of 2*mGH(X, Y) from
    # 1) mGH(X, Y) ≥ 0.5*|diam X - diam Y|;
    # 2) if X and Y are not isometric, mGH(X, Y) ≥ 0.5.
    trivial_double_lb = max(abs(diam_X - diam_Y), int(len(DX) != len(DY)))
    # Initialize lower bound of 2*mGH(X, Y).
    double_lb = trivial_double_lb
    # Try tightening the lower bound.
    d = max_diam
    while d > double_lb:
        # Try proving 2*mGH(X, Y) ≥ d using d-bounded curvatures of
        # X and Y of size 3×3 or larger. 2×2 curvatures are already
        # accounted for in trivial lower bound.

        if d <= diam_X:
            K = find_largest_size_bounded_curvature(DX, diam_X, d)
            if len(K) > 2 and confirm_lb_using_bounded_curvature(d, K, DY, max_diam):
                double_lb = d
        if d > double_lb and d <= diam_Y:
            L = find_largest_size_bounded_curvature(DY, diam_Y, d)
            if len(L) > 2 and confirm_lb_using_bounded_curvature(d, L, DX, max_diam):
                double_lb = d

        d -= 1

    return double_lb


def find_largest_size_bounded_curvature(DX, diam_X, d):
    """
    Find a largest-size d-bounded curvature of metric space X induced
    by simple unweighted graph.

    Parameters
    ----------
    DX: np.array (|X|×|X|)
        (Integer) distance matrix of X.
    diam_X: int
        Largest distance in X.

    Returns
    --------
    K: np.array (n×n)
        d-bounded curvature of X of largest size; n ≤ |X|.
    """
    # Initialize curvature K with the distance matrix of X.
    K = DX
    while np.any(K[np.triu_indices_from(K, 1)] < d):
        # Pick a row (and column) with highest number of off-diagonal
        # distances < d, then with smallest sum of off-diagonal
        # distances ≥ d.
        K_rows_sortkeys = -np.sum(K < d, axis=0) * (len(K) * diam_X) + \
                      np.sum(np.ma.masked_less(K, d), axis=0).data
        row_to_remove = np.argmin(K_rows_sortkeys)
        # Remove the row and column from K.
        K = np.delete(K, row_to_remove, axis=0)
        K = np.delete(K, row_to_remove, axis=1)

    return K


def confirm_lb_using_bounded_curvature(d, K, DY, max_diam):
    """
    For X, Y metric spaces induced by simple unweighted graph, try to
    confirm 2*mGH(X, Y) ≥ d using K, a d-bounded curvature of X.

    Parameters
    ----------
    d: int
        Lower bound candidate for 2*mGH(X, Y).
    K: np.array (n×n)
        d-bounded curvature of X; n ≥ 3.
    DY: np.array (|Y|×|Y|)
        Integer distance matrix of Y.
    max_diam: int
        Largest distance in X and Y.

    Returns
    --------
    lb_is_confirmed: bool
        Whether confirmed that 2*mGH(X, Y) ≥ d.
    """
    # If K exceeds DY in size, the Hausdorff distance between the n-th
    # curvature sets of X and Y is ≥ d, entailing 2*mGH(X, Y) ≥ d (from
    # Theorem A).
    lb_is_confirmed = len(K) > len(DY) or \
                      confirm_lb_using_bounded_curvature_row(d, K, DY, max_diam)

    return lb_is_confirmed


def confirm_lb_using_bounded_curvature_row(d, K, DY, max_diam):
    """
    For X, Y metric spaces induced by simple unweighted graph, and K,
    a d-bounded curvature of X, try to confirm 2*mGH(X, Y) ≥ d using
    some row of K.

    Parameters
    ----------
    d: int
        Lower bound candidate for 2*mGH(X, Y).
    K: np.array (n×n)
        d-bounded curvature of X; n ≥ 3.
    DY: np.array (|Y|×|Y|)
        Integer distance matrix of Y; n ≤ |Y|.
    max_diam: int
        Largest distance in X and Y.

    Returns
    --------
    lb_is_confirmed: bool
        Whether confirmed that 2*mGH(X, Y) ≥ d.
    """
    lb_is_confirmed = False
    # Represent row of K as distance distributions, and retain those
    # that are maximal by the entry-wise partial order.
    K_max_rows_distance_distributions = find_unique_max_distributions(
        represent_distance_matrix_rows_as_distributions(K, max_diam))
    # Represent rows of DY as distance distributions.
    DY_rows_distance_distributions = represent_distance_matrix_rows_as_distributions(DY, max_diam)
    # For each i ∈ 1,...,n, check if ||row_i(K) - row_i(L)||_∞ ≥ d
    # ∀L ∈ PSPS^n(DY), which entails that the Hausdorff distance
    # between the n-th curvature sets of X and Y is ≥ d, and therefore
    # 2*mGH(X, Y) ≥ d (from Theorem B).
    i = 0
    while not lb_is_confirmed and i < len(K_max_rows_distance_distributions):
        lb_is_confirmed = True
        # For fixed i, check if ||row_i(K) - row_i(L)||_∞ ≥ d
        # ∀L ∈ PSPS^n_{i←j}(DY)  ∀j ∈ 1,...,|Y|, which is equivalent to
        # ||row_i(K) - row_i(L)||_∞ ≥ d  ∀L ∈ PSPS^n(DY).
        j = 0
        while lb_is_confirmed and j < len(DY_rows_distance_distributions):
            # For fixed i and j, checking ||row_i(K) - row_i(L)||_∞ ≥ d
            # ∀L ∈ PSPS^n_{i←j}(DY) is equivalent to solving a linear
            # (bottleneck) assignment feasibility problem between the
            # entries of row_i(K) and row_j(DY).
            lb_is_confirmed = not check_assignment_feasibility(
                K_max_rows_distance_distributions[i], DY_rows_distance_distributions[j], d)
            j += 1

        i += 1

    return lb_is_confirmed

def represent_distance_matrix_rows_as_distributions(DX, max_d):
    """
    Given a metric space X induced by simple unweighted graph,
    represent each row of its distance matrix as the frequency
    distribution of its entries. Entry 0 in each row is omitted.

    Parameters
    ----------
    DX: np.array (n×n)
        (Integer) distance matrix of X.
    max_d: int
        Upper bound of the entries in DX.

    Returns
    --------
    DX_rows_distributons: np.array (n×max_d)
        Each row holds frequencies of each distance from 1 to
        max_d in the corresponding row of DX. Namely, the (i, j)-th
        entry holds the frequency of distance (max_d - j) in row_i(DX).
    """
    # Add imaginary part to distinguish identical distances from
    # different rows of D.
    unique_distances, distance_frequencies = np.unique(
        DX + 1j * np.arange(len(DX))[:, None], return_counts=True)
    # Type is signed integer to allow subtractions.
    optimal_int_type = determine_optimal_int_type(len(DX))
    DX_rows_distributons = np.zeros((len(DX), max_d + 1), dtype=optimal_int_type)
    # Construct index pairs for distance frequencies, so that the
    # frequencies of larger distances appear on the left.
    distance_frequencies_index_pairs = \
        (np.imag(unique_distances).astype(optimal_int_type),
         max_d - np.real(unique_distances).astype(max_d.dtype))
    # Fill frequency distributions of the rows of DX.
    DX_rows_distributons[distance_frequencies_index_pairs] = distance_frequencies
    # Remove (unit) frequency of distance 0 from each row.
    DX_rows_distributons = DX_rows_distributons[:, :-1]

    return DX_rows_distributons


def find_unique_max_distributions(distributions):
    """
    Given frequency distributions of entries in M positive integer
    vectors of size p, find unique maximal vectors under the following
    (entry- wise) partial order: for v, u vectors, v < u if and only if
    there exists a bijection f: {1,...,p} → {1,...,p} such that
    v_k < u_{f(k)}  ∀k ∈ {1,...,p}.

    Parameters
    ----------
    distributions: np.array (M×max_d)
        Frequency distributions of entries in the m vectors; the
        entries are bounded from above by max_d.

    Returns
    --------
    unique_max_distributions: np.array (m×max_d)
        Unique frequency distributions of the maximal vectors; m ≤ M.
    """
    pairwise_distribution_differences = \
        np.cumsum(distributions - distributions[:, None, :], axis=2)
    pairwise_distribution_less_thans = np.logical_and(
        np.all(pairwise_distribution_differences >= 0, axis=2),
        np.any(pairwise_distribution_differences > 0, axis=2))
    distributions_are_max = ~np.any(pairwise_distribution_less_thans, axis=1)
    try:
        unique_max_distributions = np.unique(distributions[distributions_are_max], axis=0)
    except AttributeError:
        # `np.unique` is not implemented in NumPy 1.12 (Python 3.4).
        unique_max_distributions = np.vstack(
            {tuple(distribution) for distribution in distributions[distributions_are_max]})

    return unique_max_distributions


def check_assignment_feasibility(v_distribution, u_distribution, d):
    """
    For positie integer vectors v of size p and u of size q ≥ p, check
    if there exists injective f: {1,...,p} → {1,...,q}, such that
    |v_k - u_{f(k)}| < d  ∀k ∈ {1,...,p}

    Parameters
    ----------
    v_distribution: np.array (max_d)
        Frequency distribution of entries in v; the entries are bounded
        from above by max_d.
    u_distribution: np.array (max_d)
        Frequency distribution of entries in u; the entries are bounded
        from above by max_d.
    d: int
        d > 0.

    Returns
    --------
    is_assignment_feasible: bool
        Whether such injective f: {1,...,p} → {1,...,q} exists.
    """
    def next_i_and_j(min_i, min_j):
        # Find reversed v distribution index of smallest v entries yet
        # to be assigned. Then find index in reversed u distribution of
        # smallest u entries to which the b entries can be assigned to.
        try:
            i = next(i for i in range(min_i, len(reversed_v_distribution))
                     if reversed_v_distribution[i] > 0)
        except StopIteration:
            # All v entries are assigned.
            i = None
            j = min_j
        else:
            j = next_j(i, max(i - (d - 1), min_j))

        return i, j

    def next_j(i, min_j):
        # Find reversed u distribution index of smallest u entries to
        # which v entries, corresponding to a given reversed v
        # distribution index, can be assigned to.
        try:
            j = next(j for j in range(min_j, min(i + (d - 1),
                                                 len(reversed_u_distribution) - 1) + 1)
                     if reversed_u_distribution[j] > 0)
        except StopIteration:
            # No u entries left to assign the particular v entries to.
            j = None

        return j

    # Copy to allow modifications and stay pure; reverse to be
    # compatible with distributions of different size.
    reversed_v_distribution = list(v_distribution[::-1])
    reversed_u_distribution = list(u_distribution[::-1])
    # Injectively assign v entries to u entries if their difference
    # is < d, going from smallest to largest entries in both v and u,
    # until all v entries are assigned or such assignment proves
    # infeasible.
    i, j = next_i_and_j(0, 0)
    while i is not None and j is not None:
        if reversed_v_distribution[i] <= reversed_u_distribution[j]:
            reversed_u_distribution[j] -= reversed_v_distribution[i]
            reversed_v_distribution[i] = 0
            i, j = next_i_and_j(i, j)
        else:
            reversed_v_distribution[i] -= reversed_u_distribution[j]
            reversed_u_distribution[j] = 0
            j = next_j(i, j)

    # The assignment is feasible if and only if for some injective f,
    # |v_k - u_{f(k)}| < d  ∀k ∈ {1,...,p}.
    is_assignment_feasible = j is not None

    return is_assignment_feasible

def find_ub(DX, DY, mapping_sample_size_order=DEFAULT_MAPPING_SAMPLE_SIZE_ORDER, double_lb=0):
    """
    For X, Y metric spaces, find an upper bound of mGH(X, Y).

    Parameters
    ----------
    DX: np.array (|X|×|X|)
        Distance matrix of X.
    DY: np.array (|Y|×|Y|)
        Distance matrix of Y.
    mapping_sample_size_order: np.array (2)
        Parameter that regulates the number of mappings to sample when
        tightening the upper bound.
    double_lb: float
        Lower bound of 2*mGH(X, Y).

    Returns
    --------
    double_ub: float
        Upper bound of 2*mGH(X, Y).
    """
    # Find upper bound of smallest distortion of a mapping in X → Y.
    ub_of_X_to_Y_min_distortion = find_ub_of_min_distortion(
        DX, DY, mapping_sample_size_order=mapping_sample_size_order, goal_distortion=double_lb)
    # Find upper bound of smallest distortion of a mapping in Y → X.
    ub_of_Y_to_X_min_distortion = find_ub_of_min_distortion(
        DY, DX, mapping_sample_size_order=mapping_sample_size_order,
        goal_distortion=ub_of_X_to_Y_min_distortion)

    return max(ub_of_X_to_Y_min_distortion, ub_of_Y_to_X_min_distortion)


def find_ub_of_min_distortion(DX, DY,
                              mapping_sample_size_order=DEFAULT_MAPPING_SAMPLE_SIZE_ORDER,
                              goal_distortion=0):
    """
    For X, Y metric spaces, find an upper bound of smallest distortion
    of a mapping in X → Y by heuristically constructing some mappings
    and choosing the smallest distortion in the sample.

    Parameters
    ----------
    DX: np.array (|X|×|X|)
        Distance matrix of X.
    DY: np.array (|Y|×|Y|)
        Distance matrix of Y.
    mapping_sample_size_order: np.array (2)
        Exponents of |X| and log (|X|+1) in their product that defines
        how many mappings from X → Y to sample.
    goal_distortion: float
        No need to look for distortion smaller than this.

    Returns
    --------
    ub_of_min_distortion: float
        Upper bound of smallest distortion of a mapping in X → Y.
    """
    # Compute the numper of mappings to sample.
    n_mappings_to_sample = int(np.ceil(np.prod(
        np.array([len(DX), np.log(len(DX) + 1)]) ** mapping_sample_size_order)))
    # Construct each mapping in X → Y in |X| steps by choosing the image
    # of π(i)-th point in X at i-th step, where π is randomly sampled
    # |X|-permutation. Image of each point is chosen to minimize the
    # intermediate distortion at each step.
    permutations_generator = (np.random.permutation(len(DX)) for _ in range(n_mappings_to_sample))
    ub_of_min_distortion = np.inf
    goal_distortion_is_matched = False
    all_sampled_permutations_are_tried = False
    pi = next(permutations_generator)
    while not goal_distortion_is_matched and not all_sampled_permutations_are_tried:
        mapped_xs_images, distortion = construct_mapping(DX, DY, pi)
        ub_of_min_distortion = min(distortion, ub_of_min_distortion)
        if ub_of_min_distortion <= goal_distortion:
            goal_distortion_is_matched = True

        try:
            pi = next(permutations_generator)
        except StopIteration:
            all_sampled_permutations_are_tried = True

    return ub_of_min_distortion


def construct_mapping(DX, DY, pi):
    """
    # For X, Y metric spaces and |X|-permutation π, construct a mapping
    from X → Y in |X| steps by choosing the image of π(i)-th point in X
    at i-th step. The image of each point is chosen to minimize the
    intermediate distortion at the corresponding step.

    DX: np.array (|X|×|X|)
        Distance matrix of X.
    DY: np.array (|Y|×|Y|)
        Distance matrix of Y.
    pi: np.array (|X|)
        |X|-permutation specifying the order in which the points in X
        are mapped.
    Returns
    --------
    mapped_xs_images: list
        image of the constructed mapping.
    distortion: float
        distortion of the constructed mapping.
    """
    # Map π(1)-th point in X to a random point in Y, due to the
    # lack of better criterion.
    mapped_xs = [pi[0]]
    mapped_xs_images = [np.random.choice(len(DY))]
    distortion = 0
    # Map π(i)-th point in X for each i = 2,...,|X|.
    for x in pi[1:]:
        # Choose point in Y that minimizes the distortion after
        # mapping π(i)-th point in X to it.
        bottlenecks_from_mapping_x = np.max(
            np.abs(DX[x, mapped_xs] - DY[:, mapped_xs_images]), axis=1)
        y = np.argmin(bottlenecks_from_mapping_x)
        # Map π(i)-th point in X to the chosen point in Y.
        mapped_xs.append(x)
        mapped_xs_images.append(y)
        distortion = max(bottlenecks_from_mapping_x[y], distortion)
        
    return mapped_xs_images, distortion
