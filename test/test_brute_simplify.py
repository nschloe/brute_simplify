import brute_simplify
import numpy


def test_tsps():
    # triple scalar product squared
    def get_tsps(e):
        return numpy.einsum('ij, ij->i', e[0], numpy.cross(e[1], e[2]))**2

    out = brute_simplify.triple_tet_find(
        get_tsps,
        num_summands=5
        )

    assert len(out) == 1

    return


if __name__ == '__main__':
    test_tsps()
