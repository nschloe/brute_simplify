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


def get_scalar_triple_product(ei_dot_ej):
    return (
        + ei_dot_ej[0, 0] * ei_dot_ej[1, 1] * ei_dot_ej[2, 2]
        + 2 * ei_dot_ej[0, 1] * ei_dot_ej[1, 2] * ei_dot_ej[2, 0]
        - ei_dot_ej[0, 0] * ei_dot_ej[1, 2]**2
        - ei_dot_ej[1, 1] * ei_dot_ej[2, 0]**2
        - ei_dot_ej[2, 2] * ei_dot_ej[0, 1]**2
        )


# def evaluate(idx, coeff_combos, ei_dot_ej):
#     # get the dot product <e_i, e_j>
#     a = ei_dot_ej[idx[..., 0], idx[..., 1]]
#     # multiply the dot products
#     # <e_i0, e_j0> * <e_i1, e_j1> * <e_i2, e_j2>
#     prd = numpy.prod(a, axis=-1)
#     # compute the sums with the coefficients
#     # + alpha_0 * <e_i0, e_j0> * <e_i1, e_j1> * <e_i2, e_j2>
#     # + [...]
#     alpha = numpy.dot(coeff_combos, prd)
#     return alpha


def create_coeffs(ei_dot_ej, idx, zeta):
    # get the dot product <e_i, e_j>
    a = ei_dot_ej[idx[..., 0], idx[..., 1]].T
    # multiply the dot products
    # <e_i0, e_j0> * <e_i1, e_j1> * <e_i2, e_j2>
    A = numpy.prod(a, axis=1)
    vals = numpy.linalg.eigvals(A)
    if min(abs(vals)) < 1.0e-10:
        # It happens that A is singular, e.g., for
        #   <e0, e0> <e0, e0> <e0, e1>,
        #   <e0, e0> <e0, e0> <e0, e2>,
        #   <e0, e0> <e0, e0> <e0, e3>
        # if e1, e2, e3 are linearly dependent. In this case, there is an
        # equivalent nonsingular matrix that is smaller. In any case, skip.
        return None
    x = numpy.linalg.solve(A, zeta)
    if False:
        # Should all pass, but it double-checked later anyways.
        for k in range(len(zeta)):
            assert validate_coeffs(ei_dot_ej[..., k], idx, zeta[k], x)
    return x


def validate_coeffs(ei_dot_ej, idx, zeta, x):
    a = ei_dot_ej[idx[..., 0], idx[..., 1]]
    A = numpy.prod(a, axis=1)
    return abs(numpy.dot(A, x) - zeta) < 1.0e-10


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


def _main():
    num_summands = 3

    # Create num_summands many random tetrahedra. Those are used to determine
    # the coefficients for the summands later. Take one more for validation.
    x_full = numpy.random.rand(4, num_summands+1, 3)
    e_full = numpy.array([
        x_full[1] - x_full[0],
        x_full[2] - x_full[0],
        x_full[3] - x_full[0],
        #
        x_full[3] - x_full[2],
        x_full[2] - x_full[1],
        x_full[1] - x_full[3],
        ])
    # ei_dot_ej = numpy.einsum('ij, kj->ik', e, e)
    ei_dot_ej_full = numpy.einsum('ilj, klj->ikl', e_full, e_full)

    # different targets
    # target_full = get_ce_ratio(ei_dot_ej_full)
    target_full = get_scalar_triple_product(ei_dot_ej_full)

    ei_dot_ej = ei_dot_ej_full[..., :-1]
    target = target_full[:-1]
    #
    ei_dot_ej_valid = ei_dot_ej_full[..., -1]
    target_valid = target_full[-1]

    idx_it, len_idx = create_combinations(len(e_full), num_summands)

    for idx in tqdm(idx_it, total=len_idx):
        idx_array = numpy.array(idx)
        cc = create_coeffs(ei_dot_ej, idx_array, target)

        if cc is not None and \
                validate_coeffs(ei_dot_ej_valid, idx_array, target_valid, cc):
            print(cc, idx)

    return


if __name__ == '__main__':
    _main()
