"""
A point in `N`-dimensional ambient space, over scalar type `T`. `T` is
`Float64` for ordinary use, but is left generic so the exact-predicate
layer (see `predicates.jl`) can build the *same* `AffineQuadratic`/`Quadric`
via the *same* formulas over `Interval` (for a rigorous error bound) or
`Rational{BigInt}` (for an exact fallback) -- error propagates correctly
through bisector construction itself this way, not just through the final
evaluation of an already-rounded result.
"""
const Pt{N,T} = SVector{N,T}

"""
`AffineQuadratic{N,K,T}`: the squared-distance-to-a-`K`-dimensional-affine-subspace
function in `N`-dimensional ambient space, `Q(x) = dist(x, A)²`, for the affine
subspace `A = {p + basis*t : t ∈ R^K}` (`basis` orthonormal, `K` columns).

This single family replaces the 2D prototype's separate `PointQuadratic`
(`K=0`) and `LineQuadratic` (`K=N-1`) types -- both are special cases here,
recovered automatically by the formula below rather than by separate code
paths, and every intermediate `K` (needed once `N ≥ 3`, e.g. a 3D line has
`K=1`) is representable the same way.
"""
struct AffineQuadratic{N,K,T}
    p::Pt{N,T}
    basis::SMatrix{N,K,T}
end

"""
The point case (`K=0`): distance to a single point. Kept as a convenience
constructor -- `AffineQuadratic(p)` -- since it's the single most common
feature (every vertex of the input complex has one).
"""
AffineQuadratic(p::Pt{N,T}) where {N,T} = AffineQuadratic{N,0,T}(p, SMatrix{N,0,T}())

"""
The hyperplane case (`K=N-1`, codimension 1): distance to the hyperplane
`{x : n·x = d}` (`n` unit). Kept as a convenience constructor since this is
the other extremely common case (every top-dimensional input simplex's
interior feature is one of these).
"""
function hyperplane_quadratic(n::Pt{N,T}, d::T) where {N,T}
    # any point on the hyperplane as anchor, and an orthonormal basis for
    # its direction (the orthogonal complement of n)
    p = d * n
    basis = _orthogonal_complement_basis(n)
    return AffineQuadratic{N,N - 1,T}(p, basis)
end

"""
Squared distance from `x` to the affine subspace `q` represents:
`|x-p|² - |basisᵀ(x-p)|²` -- the norm of `x-p` minus the norm of its
projection onto the subspace's own direction, i.e. the norm of the
orthogonal (normal) component alone. For `K=0` (empty `basis`) this
reduces to plain `|x-p|²`; no special-casing needed.
"""
function sqdist(q::AffineQuadratic{N,K,T}, x::Pt{N,T}) where {N,K,T}
    d = x - q.p
    return sum(abs2, d) - sum(abs2, q.basis' * d)
end

"""
`Quadric{N,T}`: a general (possibly degenerate) implicit quadric hypersurface
in `N`-dimensional ambient space, `{x : xᵀMx + 2bᵀx + c = 0}` (`M`
symmetric). This is the dimension-generic replacement for the 2D
prototype's closed `Curve = Union{Line,Parabola}` enum -- a hyperplane is
the `M=0` special case, a paraboloid a specific rank structure of `M`, and
`N=3` segment-vs-segment bisectors (not representable by either 2D case)
fall out of the same general formula rather than needing new curve types.
"""
struct Quadric{N,T}
    M::SMatrix{N,N,T}
    b::Pt{N,T}
    c::T
end

"""
`xᵀMx + 2bᵀx + c`, zero exactly on `q`'s surface.
"""
evaluate(q::Quadric{N,T}, x::Pt{N,T}) where {N,T} = dot(x, q.M * x) + 2 * dot(q.b, x) + q.c

"""
`AffineQuadratic{N,K,T}`'s squared-distance function, `|x-p|² - |basisᵀ(x-p)|²`,
expanded into `Quadric{N,T}` (`M,b,c`) form: `M = I - basis·basisᵀ` (the
orthogonal projector onto the subspace's *normal* complement, rank `N-K`),
`b = -M·p`, `c = pᵀMp`.
"""
function to_quadric(q::AffineQuadratic{N,K,T}) where {N,K,T}
    # StaticArrays' matrix multiply doesn't handle the K=0 (0-column) case,
    # so special-case it directly -- mathematically basis*basisᵀ is just the
    # zero matrix there anyway (an empty basis projects nothing away).
    BBt = K == 0 ? zeros(SMatrix{N,N,T}) : q.basis * q.basis'
    P = SMatrix{N,N,T}(I) - BBt
    b = -P * q.p
    c = dot(q.p, P * q.p)
    return Quadric{N,T}(P, b, c)
end

"""
The bisector of two `AffineQuadratic`s: `{x : sqdist(A,x) = sqdist(B,x)}`,
computed as the `Quadric` whose zero set is exactly that -- `to_quadric(A)`
minus `to_quadric(B)`, coefficient-wise. Always well-defined (unlike the 2D
`bisector` functions, this doesn't need per-pair-of-types dispatch).
"""
function bisector(A::AffineQuadratic{N,KA,T}, B::AffineQuadratic{N,KB,T}) where {N,KA,KB,T}
    qa, qb = to_quadric(A), to_quadric(B)
    return Quadric{N,T}(qa.M - qb.M, qa.b - qb.b, qa.c - qb.c)
end

"""
An orthonormal basis (as columns of an `N×(N-1)` matrix) for the orthogonal
complement of unit vector `n` -- used to build a hyperplane's
`AffineQuadratic` representation. `nullspace` of the 1×N matrix `nᵀ` is
exactly `{x : n·x = 0}`, given already orthonormalized. Only meaningful
over `Float64` (or another type `nullspace` supports) -- the exact/interval
paths build hyperplane quadratics directly from a known basis instead of
going through this.
"""
function _orthogonal_complement_basis(n::Pt{N,Float64}) where {N}
    ns = nullspace(reshape(Vector(n), 1, N))
    return SMatrix{N,N - 1,Float64}(ns)
end
