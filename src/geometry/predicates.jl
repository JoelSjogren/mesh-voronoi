"""
Exact rational type used for the guaranteed-correct fallback. `Float64`
coordinates convert to this losslessly (every `Float64` is exactly
representable as a rational with a power-of-two denominator), so no
precision is lost going from the fast path's inputs to the exact path's.
"""
const Exact = Rational{BigInt}

# --- generic conversions: map every Float64 field of a geometry object to
# a target scalar type T (Interval for the bounded fast path, Exact for the
# guaranteed-correct fallback). Because `sqdist`/`to_quadric`/`bisector`/
# `evaluate` in types.jl are written generically over the scalar type, once
# an object is converted, every one of those functions works on it
# unchanged -- error propagates correctly through bisector construction
# itself, not just through a final evaluation of already-rounded numbers.

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
guaranteed correct -- `args` are ordinary `Float64`-valued geometry objects
(points, `AffineQuadratic`s, ...) and `f` is a function generic over scalar
type (e.g. `(A, B, x) -> evaluate(bisector(A, B), x)`).

Evaluates `f` first over `Interval`-converted arguments; if the resulting
interval doesn't straddle zero, that settles it (the common, fast case).
Only when the interval is inconclusive does it fall back to `f` over
`Exact`-converted (`Rational{BigInt}`) arguments, whose sign is correct by
construction -- exactly zero here means the configuration is a genuine
geometric degeneracy, not a floating-point artifact, and the caller should
resolve it via a fixed rule (see `symbolic_tiebreak`) rather than treating
it as "equal by luck".
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
A fixed, deterministic rule for resolving a genuine tie (an `exact_sign`
call that returned `0`, i.e. the configuration is exactly degenerate, not
just numerically close): compares two canonical, orderable tags -- e.g.
`sort(collect(face))` for a sub-simplex's vertex-index set -- so that
whenever the *same* degenerate configuration is tested again (anywhere,
from any code path), it resolves to the *same* answer, rather than
however incidental floating-point noise or evaluation order happens to
break the tie. This is what "symbolic perturbation" comes down to in
practice for this project: not a full formal infinitesimal scheme, but a
consistent convention applied uniformly at every genuine tie.
"""
function symbolic_tiebreak(tag_a, tag_b)
    tag_a == tag_b && return 0
    return tag_a < tag_b ? -1 : 1
end
