"""
A point in `N`-dimensional ambient space, over scalar type `T`. `T` is
`Float64` for ordinary use, left generic so `predicates.jl` can build the
*same* `AffineQuadratic`/`Quadric` via the *same* formulas over `Interval`
or `Rational{BigInt}` -- error propagates correctly through bisector
construction itself, not just a final evaluation.
"""
const Pt{N,T} = SVector{N,T}

"""
`AffineQuadratic{N,K,T}`: the squared-distance-to-a-`K`-dimensional-
affine-subspace function, `Q(x) = dist(x, A)²`, for `A = {p + basis*t : t
∈ R^K}` (`basis` orthonormal, `K` columns). One family covers point (`K=0`)
and hyperplane (`K=N-1`) alike, and every intermediate `K` too (e.g. a 3D
line has `K=1`) -- no separate types needed.
"""
struct AffineQuadratic{N,K,T}
    p::Pt{N,T}
    basis::SMatrix{N,K,T}
end

"""Point case (`K=0`) -- the single most common feature."""
AffineQuadratic(p::Pt{N,T}) where {N,T} = AffineQuadratic{N,0,T}(p, SMatrix{N,0,T}())

"""Hyperplane case (`K=N-1`): distance to `{x : n·x = d}` (`n` unit)."""
function hyperplane_quadratic(n::Pt{N,T}, d::T) where {N,T}
    p = d * n
    basis = _orthogonal_complement_basis(n)
    return AffineQuadratic{N,N - 1,T}(p, basis)
end

"""
Squared distance from `x` to the affine subspace `q` represents:
`|x-p|² - |basisᵀ(x-p)|²` (norm of the orthogonal/normal component alone).
`K=0` (empty `basis`) reduces to plain `|x-p|²`, no special-casing needed.
"""
function sqdist(q::AffineQuadratic{N,K,T}, x::Pt{N,T}) where {N,K,T}
    d = x - q.p
    return sum(abs2, d) - sum(abs2, q.basis' * d)
end

"""
`Quadric{N,T}`: a general (possibly degenerate) implicit quadric
hypersurface, `{x : xᵀMx + 2bᵀx + c = 0}` (`M` symmetric) -- a hyperplane
is `M=0`, a paraboloid a specific rank of `M`, and an `N=3`
segment-vs-segment bisector falls out of the same formula too.
"""
struct Quadric{N,T}
    M::SMatrix{N,N,T}
    b::Pt{N,T}
    c::T
end

"""`xᵀMx + 2bᵀx + c`, zero exactly on `q`'s surface."""
evaluate(q::Quadric{N,T}, x::Pt{N,T}) where {N,T} = dot(x, q.M * x) + 2 * dot(q.b, x) + q.c

"""
`AffineQuadratic`'s squared-distance function expanded into `Quadric`
form: `M = I - basis·basisᵀ` (projector onto the normal complement, rank
`N-K`), `b = -M·p`, `c = pᵀMp`.
"""
function to_quadric(q::AffineQuadratic{N,K,T}) where {N,K,T}
    # StaticArrays' matmul doesn't handle K=0 (0-column) directly; basis*basisᵀ is the zero matrix there anyway.
    BBt = K == 0 ? zeros(SMatrix{N,N,T}) : q.basis * q.basis'
    P = SMatrix{N,N,T}(I) - BBt
    b = -P * q.p
    c = dot(q.p, P * q.p)
    return Quadric{N,T}(P, b, c)
end

"""
The bisector of two `AffineQuadratic`s: `{x : sqdist(A,x) = sqdist(B,x)}`,
as `to_quadric(A) - to_quadric(B)` coefficient-wise.
"""
function bisector(A::AffineQuadratic{N,KA,T}, B::AffineQuadratic{N,KB,T}) where {N,KA,KB,T}
    qa, qb = to_quadric(A), to_quadric(B)
    return Quadric{N,T}(qa.M - qb.M, qa.b - qb.b, qa.c - qb.c)
end

"""
Orthonormal basis for the orthogonal complement of unit vector `n`, via
`nullspace` of `nᵀ`. `Float64`-only -- the exact/interval paths build
hyperplane quadratics directly from a known basis instead.
"""
function _orthogonal_complement_basis(n::Pt{N,Float64}) where {N}
    ns = nullspace(reshape(Vector(n), 1, N))
    return SMatrix{N,N - 1,Float64}(ns)
end
