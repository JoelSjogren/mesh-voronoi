abstract type Quadratic end

struct PointQuadratic <: Quadratic
    p::SVector{2,Float64}
end

"""
Squared distance to an infinite line `{x : n·x = d}` (n unit) -- the formula
that applies within a segment's *interior* feature region.
"""
struct LineQuadratic <: Quadratic
    n::SVector{2,Float64}
    d::Float64
end

sqdist(q::PointQuadratic, x) = sum(abs2, x - q.p)
sqdist(q::LineQuadratic, x) = (dot(q.n, x) - q.d)^2

"""
`bisector(q1, q2)`: the curve(s) where `sqdist(q1, x) == sqdist(q2, x)`.
Always 1 line (point-vs-point), 1 parabola (point-vs-line), or 2 lines
(line-vs-line, the classical angle bisectors) -- never a harder conic, for
this restricted family of quadratics.
"""
function bisector(q1::PointQuadratic, q2::PointQuadratic)
    w = q2.p - q1.p
    # Identical points -- the two quadratics are the same function, tied
    # everywhere, not just along some curve. This is routine (not a rare
    # coincidence): it's exactly what happens comparing a shared vertex's
    # point-feature against itself from two different simplices. No curve
    # needed; callers should really avoid this comparison in the first
    # place (see `incorporate_simplex!`'s `face`-equality check), but guard
    # here too rather than divide by a zero norm.
    norm(w) < GEOM_ATOL && return Curve[]
    n = w / norm(w)
    m = (q1.p + q2.p) / 2
    return Curve[Line(n, dot(n, m))]
end

function bisector(q1::PointQuadratic, q2::LineQuadratic)
    n, d = q2.n, q2.d
    h = dot(n, q1.p) - d
    if abs(h) < GEOM_ATOL
        # Degenerate: the point lies on the directrix itself (routine, e.g.
        # any segment endpoint lies on its own supporting line, and that
        # endpoint's point-feature gets compared against a neighboring
        # segment's line-feature all the time). The "parabola" collapses to
        # the line through the point, perpendicular to the directrix.
        t̂ = SVector(-n[2], n[1])
        return Curve[Line(t̂, dot(t̂, q1.p))]
    end
    if h < 0
        n, d = -n, -d
    end
    return Curve[Parabola(q1.p, n, d)]
end
bisector(q1::LineQuadratic, q2::PointQuadratic) = bisector(q2, q1)

function bisector(q1::LineQuadratic, q2::LineQuadratic)
    out = Curve[]
    for (w, rhs) in ((q1.n - q2.n, q1.d - q2.d), (q1.n + q2.n, q1.d + q2.d))
        nrm = norm(w)
        nrm < 1e-9 && continue   # parallel segments -- degenerate, out of generic scope
        push!(out, Line(w / nrm, rhs / nrm))
    end
    return out
end
