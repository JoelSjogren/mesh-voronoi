"""
A conservative floating-point interval `[lo,hi]`, guaranteed to contain
the true real-number result of whatever chain of `+`,`-`,`*` produced it.
Fast path of `exact_sign`: if the interval doesn't straddle zero, the sign
is settled without exact arithmetic. Correctness comes from *epsilon
inflation* (widening by a safety margin after every op) rather than
hardware directed rounding, which is unreliable under SIMD/vectorization.
"""
struct Interval <: Real
    lo::Float64
    hi::Float64
end

Interval(x::Real) = Interval(Float64(x), Float64(x))
Interval(x::Interval) = x   # more specific than the Real method above (avoids Interval->Float64)

const _IV_SLOP = 8 * eps(Float64)   # per-operation safety margin

function _inflate(lo::Float64, hi::Float64)
    mag = max(abs(lo), abs(hi), 1.0)
    pad = mag * _IV_SLOP
    return Interval(lo - pad, hi + pad)
end

Base.:+(a::Interval, b::Interval) = _inflate(a.lo + b.lo, a.hi + b.hi)
Base.:-(a::Interval, b::Interval) = _inflate(a.lo - b.hi, a.hi - b.lo)
Base.:-(a::Interval) = Interval(-a.hi, -a.lo)
function Base.:*(a::Interval, b::Interval)
    p1, p2, p3, p4 = a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi
    return _inflate(min(p1, p2, p3, p4), max(p1, p2, p3, p4))
end

Base.zero(::Type{Interval}) = Interval(0.0, 0.0)
Base.one(::Type{Interval}) = Interval(1.0, 1.0)
Base.convert(::Type{Interval}, x::Real) = Interval(x)
Base.promote_rule(::Type{Interval}, ::Type{<:Real}) = Interval

"""Whether `iv` could be zero -- the exact fallback triggers when true."""
straddles_zero(iv::Interval) = iv.lo <= 0.0 <= iv.hi

"""Does `iv` contain `x`? Named to avoid colliding with `Base.contains`."""
interval_contains(iv::Interval, x::Real) = iv.lo <= x <= iv.hi
