"""
A segment feature's quadratic is either a point (`K=0`, one of the two
endpoints) or the segment's own supporting line (`K=1` -- a 1-dimensional
affine subspace, which is the segment's *interior* feature regardless of
ambient `N`; only at `N=2` does `K=1` happen to also be the hyperplane
case `K=N-1`).
"""
const SegmentQuadratic{N} = Union{AffineQuadratic{N,0,Float64},AffineQuadratic{N,1,Float64}}

"""
One feature of an input sub-simplex: the quadratic distance formula that
applies within `validity`, and `face` -- the actual sub-simplex (an
element of `P(VertexIdx)`) this feature represents. A point feature has
`validity = HalfSpace{N,Float64}[]` (valid everywhere).
"""
struct GFeature{N}
    quad::SegmentQuadratic{N}
    validity::Vector{HalfSpace{N,Float64}}
    face::Set{VertexIdx}
end

"""
The three features of a segment from `pa` (vertex `idxa`) to `pb` (vertex
`idxb`): "beyond a" and "beyond b" (point features, each valid only past
its own endpoint), and the interior (a line feature, valid only in the
strip between the two endpoints).
"""
function segment_features(pa::Pt{N,Float64}, pb::Pt{N,Float64}, idxa::VertexIdx, idxb::VertexIdx) where {N}
    d = pb - pa
    len = norm(d)
    t̂ = d / len
    # Computed once each and reused (rather than calling `dot` again at
    # each use site) so that `beyond_a`/`beyond_b`'s own halfspace and
    # `interior`'s matching one are *exactly* the same plane, bit for bit --
    # otherwise the compiler is free to contract the repeated `dot(t̂,pa)`/
    # `dot(t̂,pb)` calls differently at each call site (e.g. via FMA fusion),
    # landing a ULP apart despite being the mathematically identical
    # expression. That's enough for `insert_own_lines!`'s own plane dedup
    # (which canonicalizes and compares these exact values) to treat what
    # is geometrically one plane as two, clip it twice, and produce
    # near-duplicate vertices a ULP apart -- a real, common failure mode in
    # practice (a lone random segment's own two endpoint planes hit this
    # roughly 1 time in 3), not just a rare exact-coincidence edge case.
    da = dot(t̂, pa)
    db = dot(t̂, pb)
    beyond_a = GFeature{N}(AffineQuadratic(pa), [HalfSpace(-t̂, -da)], Set([idxa]))
    beyond_b = GFeature{N}(AffineQuadratic(pb), [HalfSpace(t̂, db)], Set([idxb]))
    interior = GFeature{N}(AffineQuadratic{N,1,Float64}(pa, SMatrix{N,1,Float64}(t̂)),
        [HalfSpace(t̂, da), HalfSpace(-t̂, -db)], Set([idxa, idxb]))
    return [beyond_a, beyond_b, interior]
end

"""
The single feature of an isolated point (vertex `idx`): valid everywhere.
Lets point and segment sub-simplices be handled through the same
`GFeature` machinery.
"""
point_feature(p::Pt{N,Float64}, idx::VertexIdx) where {N} = GFeature{N}(AffineQuadratic(p), HalfSpace{N,Float64}[], Set([idx]))
