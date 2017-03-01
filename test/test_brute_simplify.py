import brute_simplify
import numpy


def test_simple():
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
    assert out[0][1] == (
        ((0, 0), (1, 1), (2, 2)),
        )

    return


# def test_tsps():
#     # triple scalar product squared
#     def get_tsps(x):
#         e = numpy.array([x[1] - x[0], x[2] - x[0], x[3] - x[0]])
#         return numpy.einsum('ij, ij->i', e[0], numpy.cross(e[1], e[2]))
#
#     out = brute_simplify.triple_tet_find(
#         get_tsps,
#         [[1, 0], [2, 0], [3, 0]],
#         num_summands=5
#         )
#
#     assert len(out) == 1
#     assert (abs(out[0][0] - numpy.array([1., -1., -1.,  2., -1.]))).all()
#     assert out[0][1] == (
#         ((0, 0), (1, 1), (2, 2)),
#         ((0, 0), (1, 2), (1, 2)),
#         ((0, 1), (0, 1), (2, 2)),
#         ((0, 1), (0, 2), (1, 2)),
#         ((0, 2), (0, 2), (1, 1)),
#         )
#
#     return


if __name__ == '__main__':
    test_simple()
