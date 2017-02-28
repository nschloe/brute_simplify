from __future__ import print_function
import itertools
import numpy
from scipy.misc import comb
from tqdm import tqdm


def get_value(ei_dot_ej):
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


def get_value2(idx, coeff_combos):
    # get the dot product
    a = ei_dot_ej[idx[..., 0], idx[..., 1]]
    # multiply the dot products
    prd = numpy.prod(a, axis=1)
    # compute the sums with the coefficients
    alpha = numpy.dot(coeff_combos, prd)
    return alpha

def is_valid(cc, idx):
    d = numpy.array(idx)
    num_samples = 10
    for k in range(num_samples):
        x = numpy.random.rand(4, 3)
        e = numpy.array([
            x[1] - x[0],
            x[2] - x[0],
            x[3] - x[0],
            x[3] - x[2],
            x[2] - x[1],
            x[1] - x[3],
            ])
        ei_dot_ej = numpy.einsum('ij, kj-> ik', e, e)
        zeta = get_value(ei_dot_ej)
        alpha = get_value2(d, cc)
        # check for zeta equality
        if abs(alpha - zeta) > 1.0e-10:
            return False
    return True

x = numpy.random.rand(4, 3)
# x = numpy.array([
#     [0.0, 0.0, 0.0],
#     [1.3, 1.1, 0.7],
#     [0.3, 2.3, 1.0],
#     [0.2, 0.1, 2.0],
#     ])
e = numpy.array([
    x[1] - x[0],
    x[2] - x[0],
    x[3] - x[0],
    #
    x[3] - x[2],
    x[2] - x[1],
    x[1] - x[3],
    ])

ei_dot_ej = numpy.einsum('ij, kj-> ik', e, e)
zeta = get_value(ei_dot_ej)
print(zeta)

# two edges per dot-product
num_edges = len(e)
i0 = itertools.combinations_with_replacement(range(num_edges), 2)
# three dot products per summand
j0 = itertools.combinations_with_replacement(i0, 3)
# four summands in total
num_summands = 3
idx_it = itertools.combinations_with_replacement(j0, num_summands)

# Number of elements from combinations_with_replacement(a, r) is
#   (n-1+r)! / r! / (n-1)! = (n-1+r (over) r)
# if len(a) == n.
len_i0 = int(comb(num_edges+1, 2))
len_j0 = int(comb(len_i0+2, 3))
len_idx = int(comb(len_j0+2, 3))

# add a coefficient to each summand
coeffs = [
        0.0,
        +1.0, -1.0,
        +2.0, -2.0,
        +3.0, -3.0,
        # +4.0, -4.0,
        # +5.0, -5.0,
        # +6.0, -6.0,
        # +7.0, -7.0,
        # +8.0, -8.0,
        # +9.0, -9.0,
        # +10.0, -10.0,
        # +11.0, -11.0,
        # +12.0, -12.0,
        # +24.0, -24.0,
        ]
coeff_combos = numpy.array(
        list(itertools.product(coeffs, repeat=num_summands))
        )

for idx in tqdm(idx_it, total=len_idx):
    alpha = get_value2(numpy.array(idx), coeff_combos)
    # check for zeta equality
    eql = abs(alpha - zeta) < 1.0e-10
    if numpy.any(eql):
        cc = coeff_combos[numpy.where(eql)[0]]
        print('Candidate found.')
        if is_valid(cc, idx):
            print(cc, idx)
            print('Success!')
        else:
            print('Failure.')
