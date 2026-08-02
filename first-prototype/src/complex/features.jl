abstract type ValidityRegion end

struct WholePlane <: ValidityRegion end

"""
Valid where `n·x >= d` (n need not be unit; only the sign matters)."""
struct HalfPlane <: ValidityRegion
    n::SVector{2,Float64}
    d::Float64
end

struct Strip <: ValidityRegion
    lo::HalfPlane
    hi::HalfPlane
end

is_valid(r::WholePlane, x) = true
is_valid(r::HalfPlane, x) = dot(r.n, x) - r.d >= -1e-9
is_valid(r::Strip, x) = is_valid(r.lo, x) && is_valid(r.hi, x)

"""
One feature (sub-face) of an input simplex: the quadratic distance formula
that applies within `validity`, and `face` -- the actual sub-simplex (an
element of `P(VertexIdx)`) this feature represents: `{v}` for a point
simplex or a segment endpoint feature, `{a,b}` for a segment's interior
feature. Two different simplices whose currently-active feature is
*literally the same function* (e.g. two segments sharing an endpoint) are
detected via `face` equality -- exact `VertexIdx`-set equality, no
floating-point tolerance -- which doubles as the classification atom used
for labeling (see `src/algorithm/pipeline.jl`).
"""
struct Feature
    simplex::Int
    quad::Quadratic
    validity::ValidityRegion
    face::Set{VertexIdx}
end

function features(complex::InputComplex, idx::Int)::Vector{Feature}
    s = complex.simplices[idx]
    if s isa PointSimplex
        p = complex.coords[s.v]
        return [Feature(idx, PointQuadratic(p), WholePlane(), Set([s.v]))]
    else
        s::SegmentSimplex
        a, b = complex.coords[s.a], complex.coords[s.b]
        ab = b - a
        t̂ = ab / norm(ab)
        n̂ = SVector(-t̂[2], t̂[1])
        d = dot(n̂, a)
        return [
            Feature(idx, PointQuadratic(a), HalfPlane(-t̂, -dot(t̂, a)), Set([s.a])),        # beyond a
            Feature(idx, PointQuadratic(b), HalfPlane(t̂, dot(t̂, b)), Set([s.b])),          # beyond b
            Feature(idx, LineQuadratic(n̂, d), Strip(HalfPlane(t̂, dot(t̂, a)), HalfPlane(-t̂, -dot(t̂, b))), Set([s.a, s.b])),
        ]
    end
end

all_features(complex::InputComplex) = [f for i in eachindex(complex.simplices) for f in features(complex, i)]
