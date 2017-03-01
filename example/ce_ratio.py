import brute_simplify
import numpy


def get_ce_ratio(x):
    e = numpy.array([
        x[1] - x[0],
        x[2] - x[0],
        x[3] - x[0],
        x[2] - x[1],
        x[3] - x[2],
        x[1] - x[3],
        ])
    ei_dot_ej = numpy.einsum('ilj, klj->ikl', e, e)
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


out = brute_simplify.triple_tet_find(
    get_ce_ratio,
    [[1, 0], [2, 0], [3, 0], [2, 1], [3, 2], [1, 3]],
    num_summands=5
    )
