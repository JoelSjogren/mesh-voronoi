dist2(s::PointSimplex, coords, x) = sum(abs2, x - coords[s.v])

function dist2(s::SegmentSimplex, coords, x)
    a, b = coords[s.a], coords[s.b]
    ab = b - a
    t = clamp(dot(x - a, ab) / dot(ab, ab), 0.0, 1.0)
    return sum(abs2, x - (a + t * ab))
end

dist2(complex::InputComplex, i::Int, x) = dist2(complex.simplices[i], complex.coords, x)

"""
Ground-truth oracle: the set of input-simplex indices attaining the minimum
squared distance to `x`, within a tolerance scaled to the magnitude of the
minimum distance. This is the reference every later (DCEL-based) computation
is cross-checked against.
"""
function brute_force_label(complex::InputComplex, x; atol=1e-9)
    ds = [dist2(s, complex.coords, x) for s in complex.simplices]
    m = minimum(ds)
    return Set(i for (i, d) in enumerate(ds) if d <= m + atol * max(1.0, m))
end
