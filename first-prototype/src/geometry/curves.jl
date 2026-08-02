"""
Shared numerical tolerance for the geometry kernel (vertex snapping,
crossing deduplication, edge-range checks, degenerate-curve detection).
Coordinates are O(1-10) in our examples; this is loose enough to absorb the
root-finding error that accumulates over many chained curve insertions,
tight enough not to merge genuinely distinct nearby features in the generic
case this project targets.
"""
const GEOM_ATOL = 1e-6

"""
A line, stored in unit-normal form: the line is `{x : n·x = d}` with `n` unit.
"""
struct Line
    n::SVector{2,Float64}
    d::Float64
end

"""
A parabola: the locus of points equidistant from a focus `p` and a directrix
line `{x : n·x = d}` (`n` unit, and `p` must lie strictly on the `n·x > d`
side -- the parabola opens away from the directrix, toward the focus).
"""
struct Parabola
    p::SVector{2,Float64}
    n::SVector{2,Float64}
    d::Float64
end

const Curve = Union{Line,Parabola}

# --- implicit form, zero on the curve. Written generically in (x,y) so it
# also accepts Polynomials.jl `Polynomial` arguments for curve intersection. ---
implicit_xy(c::Line, x, y) = c.n[1] * x + c.n[2] * y - c.d
function implicit_xy(c::Parabola, x, y)
    dx, dy = x - c.p[1], y - c.p[2]
    dirval = c.n[1] * x + c.n[2] * y - c.d
    return (dx^2 + dy^2) - dirval^2
end

evaluate(c::Curve, x::SVector{2,Float64}) = implicit_xy(c, x[1], x[2])

# --- orthonormal local frame: `tangent` and `origin` (a fixed reference point
# on the curve), used to build a single real-parameter t -> point map. ---
tangent(c::Line) = SVector(-c.n[2], c.n[1])
tangent(c::Parabola) = SVector(-c.n[2], c.n[1])

origin(c::Line) = c.d * c.n
focal_distance(c::Parabola) = dot(c.n, c.p) - c.d
origin(c::Parabola) = c.p - focal_distance(c) * c.n

function point_at(c::Line, t::Real)
    return origin(c) + t * tangent(c)
end

function point_at(c::Parabola, u::Real)
    h = focal_distance(c)
    v = (u^2 + h^2) / (2h)
    return origin(c) + u * tangent(c) + v * c.n
end

"""
Inverse of `point_at`: the tangential coordinate of a point assumed to lie on
`c`. Just a projection onto the curve's own local frame -- exact for `Line`,
and exact for `Parabola` too since the tangential coordinate doesn't depend
on the (quadratic) normal-direction offset.
"""
param_of(c::Curve, x) = dot(x - origin(c), tangent(c))

function param_poly(c::Line)
    o, t = origin(c), tangent(c)
    return Polynomial([o[1], t[1]]), Polynomial([o[2], t[2]])
end

function param_poly(c::Parabola)
    o, t = origin(c), tangent(c)
    h = focal_distance(c)
    vp = Polynomial([h / 2, 0.0, 1 / (2h)])   # v(u) = (u^2+h^2)/(2h)
    return Polynomial([o[1], t[1]]) + vp * c.n[1], Polynomial([o[2], t[2]]) + vp * c.n[2]
end

"""
All points where curve `c1` crosses curve `c2`: substitute `c1`'s
parametrization into `c2`'s implicit equation and solve the resulting
polynomial (degree 1: line-line: degree <=2: line-parabola; degree <=4:
parabola-parabola) for real roots.
"""
function intersect_points(c1::Curve, c2::Curve)
    xp, yp = param_poly(c1)
    poly = implicit_xy(c2, xp, yp)
    pts = SVector{2,Float64}[]
    degree(poly) < 1 && return pts   # parallel / coincident / no solution: out of generic scope
    for r in roots(poly)
        abs(imag(r)) > 1e-7 && continue
        t = real(r)
        # A near-zero (but not exactly zero) leading coefficient means the
        # polynomial only *nominally* has its full degree: one root
        # legitimately flies off toward +-infinity as that coefficient's
        # contribution vanishes. For line-vs-line that's the *only* root, so
        # there's nothing left to keep -- but for a parabola involved on
        # either side (degree 2 or 4), a line running parallel to the
        # parabola's axis, or a parabola whose axis runs parallel to the
        # other curve's own axis/tangent, still legitimately crosses it at a
        # *finite* point -- this shows up routinely here, since a bisector
        # parabola's axis (perpendicular to its directrix, an existing
        # segment's own line) can easily end up parallel to some other
        # segment's perpendicular cutline. Filtering per-root by magnitude
        # (rather than bailing out on the whole polynomial once, based on
        # its leading coefficient) discards only the genuinely-flung-out
        # root and keeps any other, real, finite one.
        abs(t) > 1e8 && continue
        push!(pts, SVector(xp(t), yp(t)))
    end
    return pts
end
