"""
    Implementation of the modified Gromov-Hausdorff distance as defined
    in:

    Mémoli, F. (2012). Some properties of Gromov–Hausdorff distances.
    Discrete & Computational Geometry, 48(2), 416-440.

    Author: Vladyslav Oles

    ===================================================================

    Notations:

    V(G) denotes vertex set of graph G.

    |X| denotes the number of elements in set X.

    mGH(X, Y) denotes the modified Gromov-Hausdorff distance between
    compact metric spaces X and Y.

    ===================================================================

    Terms:

    Distance matrix of metric space X is a |X|×|X| matrix whose
    (i, j)-th entry holds the distance between i-th and j-th points of
    X. By the properties of a metric, distance matrices are symmetric
    and non-negative, their diagonal entries are 0 and off-diagonal
    entries are positive.

    Curvature of metric space X generalizes a distance matrix, induced
    by an n-tuple of points from X (i.e. by an element of X^n). The
    (i, j)-th entry of this n×n matrix holds the distance between the
    points from X contained at i-th and j-th positions of the tuple.
    However, the points contained in an element of X^n need not be
    distinct, and so the off-diagonal entries of a curvature can be 0.

    n-th curvature set of metric space X is the set of all curvatures
    of X that are of size n×n.

    d-bounded curvature for some d > 0 is a curvature whose
    off-diagonal entries are all ≥ d.

    Positive-bounded curvature is a curvature whose off-diagonal
    entries are all positive, i.e. the points in the underlying tuple
    are distinct.
"""
import numpy as np
import warnings
import scipy.sparse as sps
from scipy.sparse.csgraph import shortest_path, connected_components
from pyscipopt import Model


__all__ = ["gromov_hausdorff_between_graphs", "gromov_hausdorff"]


# To sample O(√|X| * log |X|) mappings from X → Y by default.
DEFAULT_MAPPING_SAMPLE_SIZE_ORDER = (.5, 1)


def gromov_hausdorff_between_graphs(
        A_G, A_H=None, mapping_sample_size_order=DEFAULT_MAPPING_SAMPLE_SIZE_ORDER):
    """
    Estimate the modified Gromov-Hausdorff distance between simple
    unweighted graphs, represented as compact metric spaces based on
    their shortest path lengths.

    Parameters
    -----------
    A_G: np.array (|V(G)|×|V(G)|)
        (Sparse) adjacency matrix of graph G, or an iterable of
        adjacency matrices if A_H=None.
    A_H: np.array (|V(H)|×|V(H)|)
        (Sparse) adjacency matrix of graph H, or None.
    mapping_sample_size_order: 2-tuple of floats
        Parameter that regulates the number of mappings to sample when
        tightening upper bound of the modified Gromov-Hausdorff
        distance.

    Returns
    --------
    lb: float
        Lower bound of the modified Gromov-Hausdorff distance, or a
        square matrix of lower bounds of pairwise modified
        Gromov-Hausdorff distances if A_H=None.
    ub: float
        Upper bound of the modified Gromov-Hausdorff distance, or a
        square matrix of upper bounds of pairwise modified
        Gromov-Hausdorff distances if A_H=None.
    """
    # Form iterable with adjacency matrices.
    if A_H is None:
        if len(A_G) < 2:
            raise ValueError("`estimate_between_unweighted_graphs` needs at least"
                             "2 graphs to discriminate")
        As = A_G
    else:
        As = (A_G, A_H)

    n = len(As)
    # Find lower and upper bounds of each pairwise modified
    # Gromov-Hausdorff distance between the graphs.
    lbs = np.zeros((n, n))
    ubs = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Transform adjacency matrices of a pair of graphs to
            # distance matrices.
            D_X = make_distance_matrix_from_adjacency_matrix(As[i])
            D_Y = make_distance_matrix_from_adjacency_matrix(As[j])
            # Find lower and upper bounds of the modified
            # Gromov-Hausdorff distance between the pair of graphs.
            lbs[i, j], ubs[i, j] = gromov_hausdorff(
                D_X, D_Y, mapping_sample_size_order=mapping_sample_size_order)

    if A_H is None:
        # Symmetrize matrices with lower and upper bounds of pairwise
        # modified Gromov-Hausdorff distances between the graphs.
        lower_triangle_indices = np.tril_indices(n, -1)
        lbs[lower_triangle_indices] = lbs.T[lower_triangle_indices]
        ubs[lower_triangle_indices] = ubs.T[lower_triangle_indices]

        return lbs, ubs
    else:
        return lbs[0, 1], ubs[0, 1]


def make_distance_matrix_from_adjacency_matrix(A_G):
    """
    Represent simple unweighted graph as compact metric space (with
    integer distances) based on its shortest path lengths.

    Parameters
    -----------
    A_G: np.array (|V(G)|×|V(G)|)
        (Sparse) adjacency matrix of simple unweighted graph G.

    Returns
    --------
    D_G: np.array (|V(G)|×|V(G)|)
        (Dense) distance matrix of G, represented as compact
        metric space based on its shortest path lengths.
    """
    # Convert adjacency matrix to SciPy format if needed.
    if not sps.issparse(A_G) and not isinstance(A_G, np.ndarray):
        A_G = np.asarray(A_G)

    # Compile distance matrix of the graph based on its shortest path
    # lengths.
    D_G = shortest_path(A_G, directed=False, unweighted=True)
    # Ensure compactness of metric space, represented by distance
    # matrix.
    if np.any(np.isinf(D_G)):
        warnings.warn("disconnected graph is approximated by its largest connected component")
        # Extract largest connected component of the graph.
        _, components_by_vertex = connected_components(A_G, directed=False)
        components, component_sizes = np.unique(components_by_vertex, return_counts=True)
        largest_component = components[np.argmax(component_sizes)]
        D_G = D_G[components_by_vertex == largest_component]

    # Cast distance matrix to optimal integer type.
    D_G = cast_distance_matrix_to_optimal_integer_type(D_G)

    return D_G


def cast_distance_matrix_to_optimal_integer_type(D_X):
    """
    Cast distance matrix to smallest signed integer type, sufficient
    to hold all its distances.

    Parameters
    -----------
    D_X: np.array (|X|×|X|)
        Distance matrix of a compact metric space X with integer
        distances.

    Returns
    --------
    D: np.array (|X|×|X|)
        Distance matrix of the metric space, cast to optimal type.
    """
    max_distance = np.max(D_X)
    # Type is signed integer to allow subtractions.
    optimal_int_type = determine_optimal_int_type(max_distance)
    D_X = D_X.astype(optimal_int_type)

    return D_X


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


def gromov_hausdorff(D_X, D_Y, mapping_sample_size_order=DEFAULT_MAPPING_SAMPLE_SIZE_ORDER):
    """
    For X, Y metric spaces, find lower and upper bounds of mGH(X, Y).

    Parameters
    ----------
    D_X: np.array (|X|×|X|)
        Integer distance matrix of X.
    D_Y: np.array (|Y|×|Y|)
        Integer distance matrix of Y.
    mapping_sample_size_order: 2-tuple of floats
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
    if not np.issubdtype(D_X.dtype, np.integer) or not np.issubdtype(D_Y.dtype, np.integer):
        raise ValueError("non-integer metrics are not yet supported")
    # Cast distance matrices to signed integer type to allow
    # subtractions.
    if np.issubdtype(D_X.dtype, np.uint):
        D_X = cast_distance_matrix_to_optimal_integer_type(D_X)
    if np.issubdtype(D_Y.dtype, np.uint):
        D_Y = cast_distance_matrix_to_optimal_integer_type(D_Y)
    # Estimate mGH(X, Y).
    double_lb = find_lb(D_X, D_Y)
    double_ub = find_ub(
        D_X, D_Y, mapping_sample_size_order=mapping_sample_size_order, double_lb=double_lb)

    return 0.5 * double_lb, 0.5 * double_ub


def find_lb(D_X, D_Y):
    """
    For X, Y metric spaces, find lower bound of mGH(X, Y).

    Parameters
    ----------
    D_X: np.array (|X|×|X|)
        Integer distance matrix of X.
    D_Y: np.array (|Y|×|Y|)
        Integer distance matrix of Y.

    Returns
    --------
    double_lb: float
        Lower bound of 2*mGH(X, Y).
    """
    diam_X = np.max(D_X)
    diam_Y = np.max(D_Y)
    max_distance = max(diam_X, diam_Y)
    # Obtain trivial lower bound of 2*mGH(X, Y) from
    # 1) mGH(X, Y) ≥ 0.5*|diam X - diam Y|;
    # 2) mGH(X, Y) = 0 if and only if X and Y are isometric.
    trivial_double_lb = max(abs(diam_X - diam_Y), int(len(D_X) != len(D_Y)))
    # Initialize lower bound of 2*mGH(X, Y).
    double_lb = trivial_double_lb
    # Try tightening the lower bound.
    d = max_distance
    while d > double_lb:
        # Try proving 2*mGH(X, Y) ≥ d using d-bounded curvatures of
        # X and Y of size 3×3 or larger. 2×2 curvatures are already
        # accounted for in trivial lower bound.

        if d <= diam_X:
            K = find_largest_size_bounded_curvature(D_X, diam_X, d)
            if len(K) > 2 and confirm_lb_using_bounded_curvature(d, K, D_Y, max_distance):
                double_lb = d
        if d > double_lb and d <= diam_Y:
            L = find_largest_size_bounded_curvature(D_Y, diam_Y, d)
            if len(L) > 2 and confirm_lb_using_bounded_curvature(d, L, D_X, max_distance):
                double_lb = d

        d -= 1

    return double_lb


def find_largest_size_bounded_curvature(D_X, diam_X, d):
    """
    Find a largest-size d-bounded curvature of metric space X.

    Parameters
    ----------
    D_X: np.array (|X|, |X|)
        Integer distance matrix of X.
    diam_X: int
        Largest distance in X.

    Returns
    --------
    K: np.array (N, N)
        d-bounded curvature of X of largest size; N ≤ |K|.
    """
    # Initialize curvature K with entire distance matrix.
    K = D_X
    while np.any(K[np.triu_indices_from(K, 1)] < d):
        # Pick a row (and column) with highest number of off-diagonal
        # distances < d, then with smallest sum of off-diagonal
        # distances ≥ d.
        K_rows_sortkeys = -np.sum(K < d, axis=0) * (len(K) * diam_X) + \
                      np.sum(np.ma.masked_less(K, d), axis=0).data
        row_to_remove = np.argmin(K_rows_sortkeys)
        # Remove the point from K.
        K = np.delete(K, row_to_remove, axis=0)
        K = np.delete(K, row_to_remove, axis=1)

    return K


def confirm_lb_using_bounded_curvature(d, K, D_Y, max_distance):
    """
    For X, Y metric spaces, try to confirm 2*mGH(X, Y) ≥ d
    using K, a d-bounded curvature of X.

    Parameters
    ----------
    d: int
        Lower bound candidate for 2*mGH(X, Y).
    K: np.array (N, N)
        d-bounded curvature of X; N ≥ 3.
    D_Y: np.array (|Y|, |Y|)
        Integer distance matrix of Y.
    max_distance: int
        Largest distance in X and Y.

    Returns
    --------
    lb_is_confirmed: bool
        Whether confirmed that 2*mGH(X, Y) ≥ d.
    """
    # If K exceeds D_Y in size, the Hausdorff distance between the n-th
    # curvature sets of X and Y is ≥ d, entailing 2*mGH(X, Y) ≥ d.
    lb_is_confirmed = len(D_Y) < len(K) or confirm_lb_using_bounded_curvature_principal_subrows(
        d, K, D_Y, max_distance)

    return lb_is_confirmed


def confirm_lb_using_bounded_curvature_principal_subrows(d, K, D_Y, max_distance):
    """
    For X, Y metric spaces, and K, a d-bounded curvature of X, try to
    confirm 2*mGH(X, Y) ≥ d using selected rows of selected principal
    submatrices of K.

    Parameters
    ----------
    d: int
        Lower bound candidate for 2*mGH(X, Y).
    K: np.array (N, N)
        d-bounded curvature of X; N ≥ 3.
    D_Y: np.array (|Y|, |Y|)
        Integer distance matrix of Y; |Y| ≥ N.
    max_distance: int
        Largest distance in X and Y.

    Returns
    --------
    lb_is_confirmed: bool
        Whether confirmed that 2*mGH(X, Y) ≥ d.
    """
    # Represent rows of D_Y as distance distributions.
    D_Y_rows_distance_distributions = \
        represents_distance_matrix_rows_as_distributions(D_Y, max_distance)

    # For each m ≥ 3, select those rows of m×m principal
    # submatrices of K whose entries are the largest, represent them as
    # distance distributions, and try using them to confirm that the
    # Hausdorff distance between m-th curvature sets of X and Y is ≥ d,
    # entailing 2*mGH(X, Y) ≥ d. The case of m = 2 is disregarded as
    # 2×2 curvatures are already accounted for in trivial lower bound.
    K_principal_subrows_distance_distributions = find_unique_maximal_distributions(
        represents_distance_matrix_rows_as_distributions(K, max_distance))
    lb_is_confirmed = confirm_distance_between_curvature_sets_lb(
        K_principal_subrows_distance_distributions, D_Y_rows_distance_distributions, d)
    m = len(K) - 1
    while not lb_is_confirmed and m > 2:
        K_principal_subrows_distance_distributions = find_unique_maximal_distributions(
            remove_smallest_entry_from_vectors(
                K_principal_subrows_distance_distributions))
        lb_is_confirmed = confirm_distance_between_curvature_sets_lb(
            K_principal_subrows_distance_distributions, D_Y_rows_distance_distributions, d)

        m -= 1

    return lb_is_confirmed


def represents_distance_matrix_rows_as_distributions(D, max_distance):
    """
    Represent each row of a distance matrix by frequency distribution
    of distances in it. The only distance 0 in each row is omitted.

    Parameters
    ----------
    D: np.array (N, N)
        Integer distance matrix.
    max_distance: int
        Max distance of the distribution, upper bound of entries in D.

    Returns
    --------
    D_rows_distributons: np.array (N, max_distance)
        Each row holds frequencies of each distance from 1 to
        max_distance in the corresponding row of D:
        (i, j)-th entry holds frequency of distance (max_distance - j)
        in the i-th row of D.
    """
    unique_distances, distance_frequencies = np.unique(
        D + 1j * np.arange(len(D))[:, None], return_counts=True)
    optimal_int_type = determine_optimal_int_type(len(D))
    # Type is signed integer to allow subtractions.
    D_rows_distributons = np.zeros((len(D), max_distance + 1), dtype=optimal_int_type)
    # Make larger distances appear on the left in the distributions.
    D_rows_distributons[
        (np.imag(unique_distances).astype(optimal_int_type),
         max_distance - np.real(unique_distances).astype(optimal_int_type))] = distance_frequencies
    # Remove frequency of distance 0 from each row.
    D_rows_distributons = D_rows_distributons[:, :-1]

    return D_rows_distributons


def find_unique_maximal_distributions(distributions):
    """
    Given some vectors represented as frequency distributions of their
    entries, find unique maximal vectors under the following partial
    order: for r, s vectors, r < s if and only if there exists
    bijection between the individual entries of r and s, under which
    each entry in r is smaller than the corresponding entry in s.

    Parameters
    ----------
    distributions: np.array (N, max_distance)
        Frequency distributions of N positive vectors of the same size,
        i.e. the row sums are all equal.

    Returns
    --------
    unique_maximal_distributions: np.array (M, max_distance)
        Unique frequency distributions of the maximal vectors; M ≤ N.
    """
    pairwise_distribution_differences = \
        np.cumsum(distributions - distributions[:, None, :], axis=2)
    pairwise_distribution_less_thans = np.logical_and(
        np.all(pairwise_distribution_differences >= 0, axis=2),
        np.any(pairwise_distribution_differences > 0, axis=2))
    distributions_are_maximal = ~np.any(pairwise_distribution_less_thans, axis=1)
    unique_maximal_distributions = np.unique(distributions[distributions_are_maximal], axis=0)

    return unique_maximal_distributions


def confirm_distance_between_curvature_sets_lb(rs_distributions, D_Y_rows_distributions, d):
    """
    For X, Y metric spaces, try to confirm that the Hausdorff distance
    between the m-th curvature sets of X and Y is ≥ d, using a row of
    some m×m d-bounded curvature of X.

    Parameters
    ----------
    rs_distributions: np.array (m, max_distance)
        Frequency distributions of the rows of some m×m d-bounded
        curvature of X; m ≥ 3.
    D_Y_rows_distributions: np.array (|Y|, max_distance)
        Frequency distributions of the rows of distance matrix of Y;
        |Y| ≥ m.
    d: int
        Lower bound candidate for the Hausdorff distance between the
        m-th curvature sets of X and Y.

    Returns
    --------
    distance_between_curvature_sets_lb_is_confirmed: bool
        Whether confirmed that the Hausdorff distance between the
        m-th curvature sets of X and Y is ≥ d.
    """
    # For each r, k-th row of an m×m d-bounded curvature of X for some
    # k ∈ 1,...,m, check if the l∞-distance from r to the set of k-th
    # rows of positive-bounded m×m curvatures of Y is ≥ d, then the
    # Hausdorff distance between m-th curvature sets of X and Y is ≥ d.
    distance_between_curvature_sets_lb_is_confirmed = False
    i = 0
    while not distance_between_curvature_sets_lb_is_confirmed and i < len(rs_distributions):
        distance_between_curvature_sets_lb_is_confirmed = True
        j = 0
        while distance_between_curvature_sets_lb_is_confirmed and j < len(D_Y_rows_distributions):
            distance_between_curvature_sets_lb_is_confirmed = \
                confirm_distance_to_principal_subrows_lb(
                    rs_distributions[i], D_Y_rows_distributions[j], d)
            j += 1

        i += 1

    return distance_between_curvature_sets_lb_is_confirmed


def remove_smallest_entry_from_vectors(distributions):
    """
    Remove smallest entry from each of the given vectors, represented
    as frequency distributions of their entries.

    Parameters
    ----------
    distributions: np.array (N, max_distance)
        Frequency distributions of N positive vectors.

    Returns
    --------
    updated_distributions: np.array (N, max_distance)
        Frequency distributions of N vectors, obtained by removing
        smallest entry from each of the input vectors.
    """
    updated_distributions = distributions.copy()
    # Find smallest entry in each distribution.
    max_distance = distributions.shape[1]
    smallest_entry_indices = max_distance - 1 - np.argmin(np.fliplr(distributions) == 0, axis=1)
    # Decrease frequency of smallest entry by 1 in each distribution.
    updated_distributions[np.arange(len(distributions)), smallest_entry_indices] -= 1

    return updated_distributions


def confirm_distance_to_principal_subrows_lb(r_distribution, s_distribution, d):
    """
    For vectors r of size m and s, a row of some n×n distance matrix D,
    try to confirm that l∞-distance from r to the set of those rows of
    m×m principal submatrices of D that are subvectors of s is ≥ d.

    Parameters
    ----------
    r_distribution: np.array (max_distance)
        Frequency distribution of r, positive integer vector of size m.
    s_distribution: np.array (max_distance)
        Frequency distribution of s, positive integer vector of size n;
        n ≥ m.
    d: int
        Lower bound candidate for l∞-distance from r to the set of
        those rows of m×m principal submatrices of D that are
        subvectors of s; d > 0.

    Returns
    --------
    distance_to_principal_subrows_lb_is_confirmed: bool
        Whether confirmed that l∞-distance from r to the set of those
        rows of m×m principal submatrices of D that are subvectors of
        s is ≥ d.
    """
    # For each distance, find distances that are closer than d to it.
    allowed_assignments_by_entry_distribution_index = \
        {entry_distribution_index: range(
            max(entry_distribution_index - (d - 1), 0),
            min(entry_distribution_index + (d - 1), len(r_distribution) - 1) + 1)
         for entry_distribution_index in range(len(r_distribution))}
    model = Model()
    # Set linear bottleneck assignment problem that is feasible if and
    # only if l∞-distance from r to the set of those rows of m×m
    # principal submatrices of D that are subvectors of s, is < d.
    n_assignments_of_r_entries_to_s_entries = dict()
    for r_entry_distribution_index, r_entry_frequency in enumerate(r_distribution):
        # Only allow to assign a distance from r to a distance from s
        # if they are closer than d to each other.
        for s_entry_distribution_index in \
                allowed_assignments_by_entry_distribution_index[r_entry_distribution_index]:
            s_entry_frequency = s_distribution[s_entry_distribution_index]
            n_assignments_of_r_distance_to_s_distance = model.addVar(
                "n_assignments_of_r{}_to_s{}".format(r_entry_distribution_index,
                                                     s_entry_distribution_index),
                vtype="INTEGER", ub=min(r_entry_frequency, s_entry_frequency))
            try:
                n_assignments_of_r_entries_to_s_entries[r_entry_distribution_index][
                    s_entry_distribution_index] = n_assignments_of_r_distance_to_s_distance
            except KeyError:
                n_assignments_of_r_entries_to_s_entries[r_entry_distribution_index] = \
                    {s_entry_distribution_index: n_assignments_of_r_distance_to_s_distance}
        # Ensure assigning every distance from r.
        model.addCons(r_entry_frequency == sum(
            n_assignments_of_r_entries_to_s_entries[r_entry_distribution_index].values()))

    # Solve linear bottleneck assignment problem.
    model.hideOutput()
    model.optimize()
    status = model.getStatus()
    distance_to_principal_subrows_lb_is_confirmed = (status == "infeasible")
    model.freeProb()

    return distance_to_principal_subrows_lb_is_confirmed


def find_ub(D_X, D_Y, mapping_sample_size_order=DEFAULT_MAPPING_SAMPLE_SIZE_ORDER, double_lb=0):
    """
    For X, Y metric spaces, find upper bound of mGH(X, Y).

    Parameters
    ----------
    D_X: np.array (|X|, |X|)
        Integer distance matrix of X.
    D_Y: np.array (|Y|, |Y|)
        Integer distance matrix of Y.
    mapping_sample_size_order: 2-tuple of floats
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
        D_X, D_Y, mapping_sample_size_order=mapping_sample_size_order, goal_distortion=double_lb)
    # Find upper bound of smallest distortion of a mapping in Y → X.
    ub_of_Y_to_X_min_distortion = find_ub_of_min_distortion(
        D_Y, D_X, mapping_sample_size_order=mapping_sample_size_order,
        goal_distortion=ub_of_X_to_Y_min_distortion)

    return max(ub_of_X_to_Y_min_distortion, ub_of_Y_to_X_min_distortion)


def find_ub_of_min_distortion(D_X, D_Y,
                              mapping_sample_size_order=DEFAULT_MAPPING_SAMPLE_SIZE_ORDER,
                              goal_distortion=0):
    """
    For X, Y metric spaces, finds upper bound of smallest distortion of
    a mapping in X → Y by heuristically constructing some mappings and
    choosing the smallest distortion in the sample.

    Parameters
    ----------
    D_X: np.array (|X|, |X|)
        Integer distance matrix of X.
    D_Y: np.array (|Y|, |Y|)
        Integer distance matrix of Y.
    mapping_sample_size_order: 2-tuple of floats
        Exponents a, b that define the sample size
        O(|X|^a * (log |X|)^b) of the mappings in X → Y.
    goal_distortion: float
        No need to look for distortion smaller than this.

    Returns
    --------
    ub_of_smallest_distortion: float
        Upper bound of smallest distortion of a mapping in X → Y.
    """
    ub_of_smallest_distortion = np.inf
    goal_distortion_is_matched = False
    all_sampled_permutations_are_tried = False
    n_mappings_to_sample = compute_sample_size(len(D_X), mapping_sample_size_order)
    permutations_generator = (np.random.permutation(len(D_X)) for _ in range(n_mappings_to_sample))
    # Construct each mapping in X → Y in |X| steps by choosing the image
    # of π(i)-th point in X at i-th step, where π is randomly sampled
    # |X|-permutation. Image of each point is chosen to minimize the
    # intermediate distortion at each step.
    pi = next(permutations_generator)
    while not goal_distortion_is_matched and not all_sampled_permutations_are_tried:
        # Map π(1)-th point in X to a random point in Y, due to the
        # lack of better criterion.
        mapped_xs = [pi[0]]
        mapped_xs_images = [np.random.choice(len(D_Y))]
        distortion = 0
        # Map π(i)-th point in X for each i = 2,...,|X|.
        for x in pi[1:]:
            # Choose point in Y that minimizes the distortion after
            # mapping π(i)-th point in X to it.
            bottlenecks_from_mapping_x = np.max(
                np.abs(D_X[x, mapped_xs] - D_Y[:, mapped_xs_images]), axis=1)
            y = np.argmin(bottlenecks_from_mapping_x)
            # Map π(i)-th point in X to the chosen point in Y.
            mapped_xs.append(x)
            mapped_xs_images.append(y)
            distortion = max(bottlenecks_from_mapping_x[y], distortion)

        ub_of_smallest_distortion = min(distortion, ub_of_smallest_distortion)
        if ub_of_smallest_distortion <= goal_distortion:
            goal_distortion_is_matched = True

        try:
            pi = next(permutations_generator)
        except StopIteration:
            all_sampled_permutations_are_tried = True

    return ub_of_smallest_distortion


def compute_sample_size(n, sample_size_order):
    """
    Compute sample size from its order of magnitude and the population
    size.

    Parameters
    ----------
    n: int
        Population size; n > 0.
    sample_size_order: 2-tuple of floats
        Exponents a, b that define the sample size O(n^a * (log n)^b).

    Returns
    --------
    sample_size: int
        Computed sample size.
    """
    a, b = sample_size_order
    sample_size = int(np.ceil(n**a * np.log(n + 1)**b))

    return sample_size
