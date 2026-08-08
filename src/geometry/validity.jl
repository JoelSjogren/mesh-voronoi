"""
The half-space `{x : n·x >= d}` (`n` need not be unit; only its sign
matters) -- restricts where one feature of a sub-simplex (e.g. "beyond
endpoint A" of a segment) is the locally active one.
"""
struct HalfSpace{N,T}
    n::Pt{N,T}
    d::T
end

"""
Whether `x` satisfies every half-space in `regions` -- an empty list means
"valid everywhere" (a point feature's validity region).
"""
is_valid(regions::Vector{HalfSpace{N,Float64}}, x::Pt{N,Float64}) where {N} = all(r -> dot(r.n, x) - r.d >= -1e-9, regions)
