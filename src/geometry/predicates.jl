"""
Exact rational type used for the guaranteed-correct fallback. `Float64`
converts to this losslessly (every `Float64` is exactly representable as a
rational with a power-of-two denominator).
"""
const Exact = Rational{BigInt}

# Generic conversions: map every Float64 field of a geometry object to a
# target scalar type T (Interval or Exact). sqdist/to_quadric/bisector/
# evaluate (types.jl) are written generically over scalar type, so once an
# object is converted, error propagates correctly through bisector
# construction itself, not just a final evaluation.
_convert_scalar(::Type{T}, x::Float64) where {T} = T(x)
_convert_scalar(::Type{T}, v::SVector{N,Float64}) where {T,N} = SVector{N,T}(_convert_scalar.(T, v))
_convert_scalar(::Type{T}, m::SMatrix{N,K,Float64}) where {T,N,K} = SMatrix{N,K,T}(_convert_scalar.(T, m))

function _convert_scalar(::Type{T}, q::AffineQuadratic{N,K,Float64}) where {T,N,K}
    AffineQuadratic{N,K,T}(_convert_scalar(T, q.p), _convert_scalar(T, q.basis))
end

function _convert_scalar(::Type{T}, q::Quadric{N,Float64}) where {T,N}
    Quadric{N,T}(_convert_scalar(T, q.M), _convert_scalar(T, q.b), _convert_scalar(T, q.c))
end

to_interval(x) = _convert_scalar(Interval, x)
to_exact(x) = _convert_scalar(Exact, x)

"""
`exact_sign(f, args...)`: the sign of `f(args...)` (`-1`, `0`, or `1`),
guaranteed correct -- `args` are `Float64`-valued geometry objects, `f` is
generic over scalar type. Tries `Interval` first (fast path); only when
that straddles zero does it fall back to `Exact` rational arithmetic. A
`0` result means genuine geometric degeneracy, not float noise -- resolve
it via `symbolic_tiebreak`, not by treating it as "equal by luck".
"""
function exact_sign(f, args...)
    iv_result = f(to_interval.(args)...)
    if !straddles_zero(iv_result)
        return iv_result.lo > 0.0 ? 1 : -1
    end
    exact_result = f(to_exact.(args)...)
    return exact_result > 0 ? 1 : (exact_result < 0 ? -1 : 0)
end

"""
Deterministic tiebreak for a genuine `exact_sign` zero: compares two
canonical, orderable tags (e.g. `sort(collect(face))`) so the *same*
degenerate configuration always resolves the same way, everywhere it's
checked -- this project's stand-in for symbolic perturbation.
"""
function symbolic_tiebreak(tag_a, tag_b)
    tag_a == tag_b && return 0
    return tag_a < tag_b ? -1 : 1
end
