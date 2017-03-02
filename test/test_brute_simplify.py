import brute_simplify
import numpy


def test_coefficient():
    def get_simple(x):
        e = numpy.array([x[1] - x[0], x[2] - x[0], x[3] - x[0]])
        ei_dot_ej = numpy.einsum('ilj, klj->ikl', e, e)
        return 0.225 * ei_dot_ej[0, 0] * ei_dot_ej[1, 1] * ei_dot_ej[2, 2]

    out = brute_simplify.triple_tet_find(
        get_simple,
        [[1, 0], [2, 0], [3, 0]],
        num_summands=1,
        verbose=False
        )

    assert len(out) == 1
    assert (abs(out[0][0] - numpy.array([0.225])) < 1.0e-12).all()
    assert (out[0][1] == (
        ((0, 0), (1, 1), (2, 2)),
        )).all()

    return


def test_simplification():
    def get_simple(x):
        e = numpy.array([x[1] - x[0], x[2] - x[0], x[3] - x[0]])
        ei_dot_ej = numpy.einsum('ilj, klj->ikl', e, e)
        return (
            + ei_dot_ej[0, 0] * ei_dot_ej[1, 1] * ei_dot_ej[2, 2]
            - ei_dot_ej[0, 1] * ei_dot_ej[1, 1] * ei_dot_ej[2, 2]
            )

    out = brute_simplify.triple_tet_find(
        get_simple,
        [[1, 0], [2, 0], [3, 0], [2, 1], [3, 2], [1, 3]],
        num_summands=1,
        verbose=False
        )

    assert len(out) == 1
    assert (abs(out[0][0] - numpy.array([-1.0])) < 1.0e-12).all()
    assert (out[0][1] == (
        ((0, 3), (1, 1), (2, 2)),
        )).all()

    return
