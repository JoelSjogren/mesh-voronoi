"""
The half-space `{x : n·x >= d}` (`n` need not be unit; only its sign
matters). Dimension-generic replacement for the 2D prototype's `HalfPlane`
-- a validity region restricting where one feature of a sub-simplex (e.g.
"beyond endpoint A" of a segment) is the locally active one.
"""
struct HalfSpace{N,T}
    n::Pt{N,T}
    d::T
end

"""
Whether `x` satisfies every half-space in `regions` -- an empty list means
"valid everywhere" (a point feature's validity region, matching G1 where
this concept wasn't needed at all). Generalizes the 2D prototype's
`WholePlane`/`HalfPlane`/`Strip` into one representation: a `Strip` is
exactly two half-spaces, and a general `k`-simplex's relative interior
(one half-space per facet, per the project plan) is the same idea with
more of them.
"""
is_valid(regions::Vector{HalfSpace{N,Float64}}, x::Pt{N,Float64}) where {N} = all(r -> dot(r.n, x) - r.d >= -1e-9, regions)
