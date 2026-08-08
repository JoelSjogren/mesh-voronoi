function padded_bbox(points::Vector{Pt{N,Float64}}; pad=0.3) where {N}
    lo = SVector{N,Float64}(minimum(p[i] for p in points) for i in 1:N)
    hi = SVector{N,Float64}(maximum(p[i] for p in points) for i in 1:N)
    scale = max(maximum(hi - lo), 1.0)
    return lo .- pad * scale, hi .+ pad * scale
end

"""
The full points-only (G1) construction for exactly two input points: build
a padded bounding box and clip it by their bisector. Returns
`(cx, a_id, b_id, cut_id)`.
"""
function two_points_complex(pa::Pt{N,Float64}, pb::Pt{N,Float64}) where {N}
    lo, hi = padded_bbox([pa, pb])
    cx, top = init_bbox_complex(Val(N), lo, hi)
    A, B = AffineQuadratic(pa), AffineQuadratic(pb)
    a_id, b_id, cut_id = clip_by_hyperplane!(cx, top, A, B, Set([1]), Set([2]))
    return cx, a_id, b_id, cut_id
end

"""
Brute-force oracle: the set of input point indices (`1` or `2`) achieving
the minimum squared distance to `x`, within floating-point tie tolerance.
"""
function brute_force_label_two_points(pa::Pt{N,Float64}, pb::Pt{N,Float64}, x::Pt{N,Float64}; atol=1e-9) where {N}
    da, db = sum(abs2, x - pa), sum(abs2, x - pb)
    m = min(da, db)
    out = Set{Int}()
    da <= m + atol * max(1.0, m) && push!(out, 1)
    db <= m + atol * max(1.0, m) && push!(out, 2)
    return out
end
