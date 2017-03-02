import brute_simplify
import numpy


# triple scalar product squared
def get_tsps(x):
    e = numpy.array([x[1] - x[0], x[2] - x[0], x[3] - x[0]])
    return numpy.einsum('ij, ij->i', e[0], numpy.cross(e[1], e[2]))**2


out = brute_simplify.triple_tet_find(
    get_tsps,
    [[1, 0], [2, 0], [3, 0]],
    num_summands=5
    )
# out = brute_simplify.triple_tet_find(
#     get_tsps,
#     [[1, 0], [2, 0], [3, 0], [2, 1], [3, 2], [1, 3]],
#     num_summands=3
#     )
print(out)
