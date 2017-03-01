import itertools
import numpy
from scipy.misc import comb
from tqdm import tqdm


def check(ei_dot_ej, idx, zeta):
    # get the dot product <e_i, e_j>
    a = ei_dot_ej[idx[..., 0], idx[..., 1]].T
    # multiply the dot products
    # <e_i0, e_j0> * <e_i1, e_j1> * <e_i2, e_j2>
    A = numpy.prod(a, axis=1)

    x, residuals, rank, s = numpy.linalg.lstsq(A, zeta)

    # A is of shape (n+1, n), where each row corresponds to a random
    # tetrahedron, each column to a coefficient.
    # If this equation system has a solution, it's probably valid for many more
    # tetrahedra.
    res = numpy.dot(A, x) - zeta
    if numpy.all(abs(res) < 1.0e-10):
        return x

    return None


def create_combinations(num_edges, num_summands):
    # create the combinations
    # two edges per dot-product
    i0 = itertools.combinations_with_replacement(range(num_edges), 2)
    # three dot products per summand
    j0 = itertools.combinations_with_replacement(i0, 3)
    # Add up a bunch of summands. Make sure they are differnet (no
    # `_with_replacement`); the rest is handled by the coefficients later.
    idx_it = itertools.combinations(j0, num_summands)

    # Number of elements from combinations_with_replacement(a, r) is
    #   (n-1+r)! / r! / (n-1)! = (n-1+r (over) r)
    # if len(a) == n.
    len_i0 = comb(num_edges+1, 2, exact=True)
    len_j0 = comb(len_i0+2, 3, exact=True)
    # n! / r! / (n-r)! = (n (over) r)
    # len_idx = comb(len_j0, num_summands, exact=True)
    len_idx = comb(len_j0, num_summands, exact=True)
    return idx_it, len_idx


def triple_tet_find(compute_targets, edges, num_summands=5, verbose=True):
    # Create num_summands+1 many random tetrahedra.
    x = numpy.random.rand(4, num_summands+1, 3)

    targets = compute_targets(x)

    i = numpy.array(edges)
    e = x[i[:, 0]] - x[i[:, 1]]
    ei_dot_ej = numpy.einsum('ilj, klj->ikl', e, e)

    idx_it, len_idx = create_combinations(len(e), num_summands)

    solutions = []
    if verbose:
        it = tqdm(idx_it, total=len_idx)
    else:
        it = idx_it
    for idx in it:
        idx_array = numpy.array(idx)
        cc = check(ei_dot_ej, idx_array, targets)

        if cc is not None:
            if verbose:
                print(cc, idx)
            solutions.append((cc, idx))

    return solutions
