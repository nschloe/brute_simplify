import itertools
import numpy
from scipy.misc import comb
import sys
from tqdm import tqdm


def least_squares(A, b):
    # http://stackoverflow.com/a/42537466/353337
    # https://github.com/numpy/numpy/issues/8720
    u, s, v = numpy.linalg.svd(A, full_matrices=False)
    uTb = numpy.einsum('ijk,ij->ik', u, b)
    xx = numpy.einsum('ijk, ij->ik', v, uTb / s)
    return xx


def check(ei_dot_ej, idx, zeta):
    # get the dot product <e_i, e_j>
    out = []
    # get the dot product <e_i, e_j>
    a = ei_dot_ej[idx[..., 0], idx[..., 1]].T
    # multiply the dot products
    # <e_i0, e_j0> * <e_i1, e_j1> * <e_i2, e_j2>
    A = numpy.prod(a, axis=1)
    # roll the last axis to the front; numpy's vectorization needs that. :(
    # A is now of shape (777, n+1, n), where each row corresponds to a random
    # tetrahedron, each column to a coefficient.
    # If this equation system has a solution, it's probably valid for many more
    # tetrahedra.
    A = numpy.rollaxis(A, 2)

    # adapt the rhs for size
    zeta = numpy.outer(numpy.ones(A.shape[0]), zeta)
    x = least_squares(A, zeta)

    A_dot_x = numpy.einsum('ijk, ik->ij', A, x)
    res = A_dot_x - zeta
    # Check if for any item, all of the residuals are 0.
    res2 = numpy.all(abs(res) < 1.0e-10, axis=1)

    out = []
    if numpy.any(res2):
        K = numpy.where(res2)[0]
        out += [(x[k], idx[k]) for k in K]

    return out


def create_combinations(num_edges, num_summands):
    # create the combinations
    # two edges per dot-product
    i0 = itertools.combinations_with_replacement(range(num_edges), 2)
    # three dot products per summand
    j0 = itertools.combinations_with_replacement(i0, 3)
    # Add up a bunch of summands. Make sure they are different (no
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


# http://stackoverflow.com/a/8290490/353337
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    if sys.version_info[0] == 3:
        zl = itertools.zip_longest
    else:
        zl = itertools.izip_longest
    return zl(*args, fillvalue=fillvalue)


def triple_tet_find(
        compute_targets,
        edges,
        num_summands=5,
        batch_size=100,
        verbose=True
        ):
    # Create num_summands+1 many random tetrahedra.
    x = numpy.random.rand(4, num_summands+1, 3)

    targets = compute_targets(x)

    edges_array = numpy.array(edges)
    e = x[edges_array[:, 0]] - x[edges_array[:, 1]]

    ei_dot_ej = numpy.einsum('ilj, klj->ikl', e, e)

    idx_it, len_idx = create_combinations(len(e), num_summands)

    idx_it = grouper(idx_it, batch_size)
    len_idx = int(len_idx / batch_size) + 1

    solutions = []
    it = tqdm(idx_it, total=len_idx) if verbose else idx_it
    for idx in it:
        # The last batch may be smaller and contains Nones as fillers. Remove
        # them.
        idx = list(idx)
        while idx and idx[-1] is None:
            idx.pop()
        idx_array = numpy.array(idx)

        out = check(ei_dot_ej, idx_array, targets)
        if out:
            print(out)
            solutions += out

    return solutions
