"""
A conservative floating-point interval `[lo,hi]`, guaranteed (not just
typically) to contain the true real-number result of whatever chain of
`+`,`-`,`*` produced it, given exact inputs. Used as the fast path of the
exact-sign predicate (see `exact_sign` below): evaluate a formula with
`Interval` in place of `Float64`, and if the resulting interval doesn't
straddle zero, the sign is settled without needing exact arithmetic at all.

Correctness comes from *epsilon inflation* rather than hardware directed
rounding modes (`setrounding`): after every operation, the naively-computed
interval is widened by a safety margin covering that operation's worst-case
IEEE 754 rounding error (`eps()/2` relative, per elementary operation).
This is simpler and more portable than directed rounding -- which performs
poorly or is unsupported for SIMD/vectorized code on modern hardware, and
is exactly the kind of subtlety this project already got burned by once
this session (ad hoc tolerances silently disagreeing with each other) --
at the cost of slightly looser bounds, which only means the exact fallback
triggers a bit more often than the theoretical minimum, never that the
fast path is wrong.
"""
struct Interval <: Real
    lo::Float64
    hi::Float64
end

Interval(x::Real) = Interval(Float64(x), Float64(x))
Interval(x::Interval) = x   # more specific than the Real method above (Interval <: Real):
                            # avoid trying (and failing) to convert an Interval to a single Float64

"""
Safety factor per operation: a generous multiple of `eps()`, wide enough to
cover accumulated rounding across the handful of elementary operations any
one predicate chains together, without being so wide it routinely triggers
the exact fallback for genuinely well-separated cases.
"""
const _IV_SLOP = 8 * eps(Float64)

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

"""
Whether `iv` could be zero -- i.e. whether the fast interval path is
inconclusive about the true value's sign and the exact fallback is needed.
"""
straddles_zero(iv::Interval) = iv.lo <= 0.0 <= iv.hi

"""
Does `iv` actually contain `x` -- used only in tests, to empirically check
the containment property `Interval` arithmetic is supposed to guarantee
(computing the same formula in exact `Rational{BigInt}` should always land
inside the `Interval`-computed bound). Named to avoid colliding with
`Base.contains` (for strings), which callers would otherwise silently
resolve to instead of this method.
"""
interval_contains(iv::Interval, x::Real) = iv.lo <= x <= iv.hi
