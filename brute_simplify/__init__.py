import itertools
import numpy
from scipy.misc import comb
from tqdm import tqdm


def get_ce_ratio(ei_dot_ej):
    zeta = (
        + ei_dot_ej[0, 0] * ei_dot_ej[1, 1] * ei_dot_ej[2, 2]
        - 4 * ei_dot_ej[0, 1] * ei_dot_ej[1, 2] * ei_dot_ej[2, 0]
        - (
            + ei_dot_ej[0, 0] * ei_dot_ej[1, 2]
            + ei_dot_ej[1, 1] * ei_dot_ej[2, 0]
            + ei_dot_ej[2, 2] * ei_dot_ej[0, 1]
            )
        * (
            + ei_dot_ej[0, 0] + ei_dot_ej[1, 1] + ei_dot_ej[2, 2]
            - ei_dot_ej[0, 1] - ei_dot_ej[1, 2] - ei_dot_ej[2, 0]
        )
        + ei_dot_ej[0, 0]**2 * ei_dot_ej[1, 2]
        + ei_dot_ej[1, 1]**2 * ei_dot_ej[2, 0]
        + ei_dot_ej[2, 2]**2 * ei_dot_ej[0, 1]
        )
    return zeta


def check(ei_dot_ej, idx, zeta):
    # get the dot product <e_i, e_j>
    a = ei_dot_ej[idx[..., 0], idx[..., 1]].T
    # multiply the dot products
    # <e_i0, e_j0> * <e_i1, e_j1> * <e_i2, e_j2>
    A = numpy.prod(a, axis=1)

    x, residuals, rank, s = numpy.linalg.lstsq(A, zeta)

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


def triple_tet_find(compute_target, num_summands=5):

    # Create num_summands many random tetrahedra. Those are used to determine
    # the coefficients for the summands later. Take one more for validation.
    x = numpy.random.rand(4, num_summands+1, 3)
    e = numpy.array([
        x[1] - x[0],
        x[2] - x[0],
        x[3] - x[0],
        # #
        # x_full[3] - x_full[2],
        # x_full[2] - x_full[1],
        # x_full[1] - x_full[3],
        ])

    target = compute_target(e)
    # different targets
    # target_full = get_ce_ratio(ei_dot_ej_full)
    # target_full = get_scalar_triple_product_squared(e_full)

    # ei_dot_ej = numpy.einsum('ij, kj->ik', e, e)
    ei_dot_ej = numpy.einsum('ilj, klj->ikl', e, e)

    idx_it, len_idx = create_combinations(len(e), num_summands)

    solutions = []
    for idx in tqdm(idx_it, total=len_idx):
        idx_array = numpy.array(idx)
        cc = check(ei_dot_ej, idx_array, target)

        if cc is not None:
            print(cc, idx)
            solutions.append((cc, idx))

    return solutions
