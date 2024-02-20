"""
    Persistence Landscape Exact class
"""

import itertools
from operator import itemgetter

import numpy as np

from .approximate import PersLandscapeApprox
from .auxiliary import _p_norm, union_crit_pairs
from .base import PersLandscape

__all__ = ["PersLandscapeExact"]


class PersLandscapeExact(PersLandscape):
    """Persistence Landscape Exact class.

    This class implements an exact version of Persistence Landscapes. The landscape
    functions are stored as a list of critical pairs, and the actual function is the
    linear interpolation of these critical pairs. All
    computations done with these classes are exact. For much faster but
    approximate methods that should suffice for most applications, consider
    `PersLandscapeApprox`.

    Parameters
    ----------
    dgms : list of (-,2) numpy.ndarrays, optional
        A nested list of numpy arrays, e.g., [array( array([:]), array([:]) ),..., array([:])]
        Each entry in the list corresponds to a single homological degree.
        Each array represents the birth-death pairs in that homological degree. This is
        the output format from ripser.py: ripser(data_user)['dgms']. Only
        one of diagrams or critical pairs should be specified.

    hom_deg : int
        Represents the homology degree of the persistence diagram.

    critical_pairs : list, optional
        List of lists of critical pairs (points, values) for specifying a persistence landscape.
        These do not necessarily have to arise from a persistence
        diagram. Only one of diagrams or critical pairs should be specified.

    compute : bool, optional
        Flag determining whether landscape functions are computed upon instantiation.


    Examples
    --------
    Define a persistence diagram and instantiate the landscape::

        >>> from persim import PersLandscapeExact
        >>> import numpy as np
        >>> pd = [ np.array([[0,3],[1,4]]), np.array([[1,4]]) ]
        >>> ple = PersLandscapeExact(dgms=pd, hom_deg=0)
        >>> ple

    `PersLandscapeExact` instances store the critical pairs of the landscape as a list of lists in the `critical_pairs` attribute. The `i`-th entry corresponds to the critical values of the depth `i` landscape::

        >>> ple.critical_pairs

        [[[0, 0], [1.5, 1.5], [2.0, 1.0], [2.5, 1.5], [4, 0]],
        [[1, 0], [2.0, 1.0], [3, 0]]]

    Addition, subtraction, and scalar multiplication between landscapes is implemented::

        >>> pd2 = [ np.array([[0.5,7],[3,5],[4.1,6.5]]), np.array([[1,4]])]
        >>> pl2 = PersLandscapeExact(dgms=pd2,hom_deg=0)
        >>> pl_sum = ple + pl2
        >>> pl_sum.critical_pairs

        [[[0, 0], [0.5, 0.5], [1.5, 2.5], [2.0, 2.5],
        [2.5, 3.5], [3.75, 3.5],[4, 3.0], [7.0, 0.0]],
        [[1, 0], [2.0, 1.0], [3, 0.0], [4.0, 1.0],
        [4.55, 0.45], [5.3, 1.2], [6.5, 0.0]],
        [[4.1, 0], [4.55, 0.45], [5.0, 0]]]

        >>> diff_pl = ple - pl2
        >>> diff_pl.critical_pairs

        [[[0, 0], [0.5, 0.5], [1.5, 0.5], [2.0, -0.5],
        [2.5, -0.5], [3.75, -3.0], [4, -3.0], [7.0, 0.0]],
        [[1, 0], [2.0, 1.0], [3, 0.0], [4.0, -1.0],
        [4.55, -0.45], [5.3, -1.2], [6.5, 0.0]],
        [[4.1, 0], [4.55, -0.45], [5.0, 0]]]

        >>> (5*ple).critical_pairs

        [[(0, 0), (1.5, 7.5), (2.0, 5.0), (2.5, 7.5), (4, 0)],
        [(1, 0), (2.0, 5.0), (3, 0)]]

    Landscapes are sliced by depth and slicing returns the critical pairs in the range specified::

        >>> ple[0]

        [[0, 0], [1.5, 1.5], [2.0, 1.0], [2.5, 1.5], [4, 0]]

        >>> pl2[1:]

        [[[3.0, 0], [4.0, 1.0], [4.55, 0.4500000000000002],
        [5.3, 1.2000000000000002], [6.5, 0]],
        [[4.1, 0], [4.55, 0.4500000000000002], [5.0, 0]]]

    `p` norms are implemented for all `p` as well as the supremum norm::

        >>> ple.p_norm(p=3)

        1.7170713638299977

        >>> pl2.sup_norm()

        3.25
    """

    def __init__(
        self,
        dgms: list = [],
        hom_deg: int = 0,
        critical_pairs: list = [],
        compute: bool = True,
    ) -> None:
        super().__init__(dgms=dgms, hom_deg=hom_deg)
        self.critical_pairs = critical_pairs
        if dgms:
            self.dgms = dgms[self.hom_deg]
        else:  # critical pairs are passed. Is this the best check for this?
            self.dgms = dgms
        if not dgms and not critical_pairs:
            raise ValueError("dgms and critical_pairs cannot both be empty")
        self.max_depth = len(self.critical_pairs)
        if compute:
            self.compute_landscape()

    def __repr__(self):
        return f"Exact persistence landscape in homological degree {self.hom_deg}"

    def __neg__(self):
        """
        Computes the negation of a persistence landscape object

        """
        self.compute_landscape()
        return PersLandscapeExact(
            hom_deg=self.hom_deg,
            critical_pairs=[
                [[a, -b] for a, b in depth_list] for depth_list in self.critical_pairs
            ],
        )

    def __add__(self, other):
        """
        Computes the sum of two persistence landscape objects

        """

        if self.hom_deg != other.hom_deg:
            raise ValueError("homological degrees must match")
        return PersLandscapeExact(
            critical_pairs=union_crit_pairs(self, other), hom_deg=self.hom_deg
        )

    def __sub__(self, other):
        """
        Computes the difference of two persistence landscape objects

        """
        return self + -other

    def __mul__(self, other: float):
        """
        Computes the product of a persistence landscape object and a float

        Parameters
        -------
        other: float
            the real scalar the persistence landscape will be multiplied by

        """
        self.compute_landscape()
        return PersLandscapeExact(
            hom_deg=self.hom_deg,
            critical_pairs=[
                [(a, other * b) for a, b in depth_list]
                for depth_list in self.critical_pairs
            ],
        )

    def __rmul__(self, other: float):
        """
        Computes the product of a persistence landscape object and a float

        Parameters
        -------
        other: float
            the real scalar the persistence landscape will be multiplied by

        """
        return self.__mul__(other)

    def __truediv__(self, other: float):
        """
        Computes the quotient of a persistence landscape object and a float

        Parameters
        -------
        other: float
            the real divisor of the persistence landscape object

        """

        if other == 0.0:
            raise ValueError("Cannot divide by zero")
        return self * (1.0 / other)

    def __getitem__(self, key: slice) -> list:
        """
        Returns a list of critical pairs corresponding in range specified by
        depth

        Parameters
        ----------
        key : slice object

        Returns
        -------
        list
            The critical pairs of the landscape function corresponding
        to depths given by key

        Note
        ----
        If the slice is beyond `self.max_depth` an IndexError is raised.
        """
        self.compute_landscape()
        return self.critical_pairs[key]

    def compute_landscape(self, verbose: bool = False) -> list:
        """
        Stores the persistence landscape in `self.critical_pairs` as a list

        Parameters
        ----------
        verbose: bool, optional
            If true, progress messages are printed during computation

        """

        verboseprint = print if verbose else lambda *a, **k: None

        # check if landscapes were already computed
        if self.critical_pairs:
            verboseprint(
                "self.critical_pairs was not empty and stored value was returned"
            )
            return self.critical_pairs

        A = self.dgms
        # change A into a list
        A = list(A)
        # change inner nparrays into lists
        for i in range(len(A)):
            A[i] = list(A[i])
        if A[-1][1] == np.inf:
            A.pop(-1)

        landscape_idx = 0
        L = []

        # Sort A: read from right to left inside ()
        A = sorted(A, key=lambda x: [x[0], -x[1]])

        while A:
            verboseprint(f"computing landscape index {landscape_idx+1}...")

            # add a 0 element to begin count of lamda_k
            # size_landscapes = np.append(size_landscapes, [0])

            # pop first term
            b, d = A.pop(0)
            verboseprint(f"(b,d) is ({b},{d})")

            # outer brackets for start of L_k
            L.append([[-np.inf, 0], [b, 0], [(b + d) / 2, (d - b) / 2]])

            # check for duplicates of (b,d)
            duplicate = 0

            for j, itemj in enumerate(A):
                if itemj == [b, d]:
                    duplicate += 1
                    A.pop(j)
                else:
                    break

            while L[landscape_idx][-1] != [np.inf, 0]:
                # if d is > = all remaining pairs, then end lambda
                # includes edge case with (b,d) pairs with the same death time
                if all(d >= _[1] for _ in A):
                    # add to end of L_k
                    L[landscape_idx].extend([[d, 0], [np.inf, 0]])
                    # for duplicates, add another copy of the last computed lambda
                    for _ in range(duplicate):
                        L.append(L[-1])
                        landscape_idx += 1

                else:
                    # set (b', d')  to be the first term so that d' > d
                    for i, item in enumerate(A):
                        if item[1] > d:
                            b_prime, d_prime = A.pop(i)
                            verboseprint(f"(bp,dp) is ({b_prime},{d_prime})")
                            break

                    # Case I
                    if b_prime > d:
                        L[landscape_idx].extend([[d, 0]])

                    # Case II
                    if b_prime >= d:
                        L[landscape_idx].extend([[b_prime, 0]])

                    # Case III
                    else:
                        L[landscape_idx].extend(
                            [[(b_prime + d) / 2, (d - b_prime) / 2]]
                        )
                        # push (b', d) into A in order
                        # find the first b_i in A so that b'<= b_i

                        # push (b', d) to end of list if b' not <= any bi
                        ind = len(A)
                        for i in range(len(A)):
                            if b_prime <= A[i][0]:
                                ind = i  # index to push (b', d) if b' != b_i
                                break
                        # if b' not < = any bi, put at the end of list
                        if ind == len(A):
                            pass
                        # if b' = b_i
                        elif b_prime == A[ind][0]:
                            # pick out (bk,dk) such that b' = bk
                            A_i = [item for item in A if item[0] == b_prime]

                            # move index to the right one for every d_i such that d < d_i
                            for j in range(len(A_i)):
                                if d < A_i[j][1]:
                                    ind += 1

                                # d > dk for all k

                        A.insert(ind, [b_prime, d])

                    L[landscape_idx].extend(
                        [[(b_prime + d_prime) / 2, (d_prime - b_prime) / 2]]
                    )
                    # size_landscapes[landscape_idx] += 1

                    b, d = b_prime, d_prime  # Set (b',d')= (b, d)

            landscape_idx += 1

        verboseprint("self.critical_pairs was empty and algorthim was executed")
        self.max_depth = len(L)
        self.critical_pairs = [item[1:-1] for item in L]

    def compute_landscape_by_depth(self, depth: int) -> list:
        """
        Returns the function of depth from `self.critical_pairs` as a list

        Parameters
        ----------
        depth: int
            the depth of the desired landscape function
        """

        if self.critical_pairs:
            return self.critical_pairs[depth]
        else:
            return self.compute_landscape()[depth]

    def p_norm(self, p: int = 2) -> float:
        """
        Returns the L_{`p`} norm of an exact persistence landscape

        Parameters
        ----------
        p: float, default 2
            value p of the L_{`p`} norm
        """
        super().p_norm(p=p)
        return _p_norm(p=p, critical_pairs=self.critical_pairs)

    def sup_norm(self) -> float:
        """
        Returns the supremum norm of an exact persistence landscape
        """

        self.compute_landscape()
        cvals = list(itertools.chain.from_iterable(self.critical_pairs))
        return max(np.abs(cvals), key=itemgetter(1))[1]
