"""
Whether `quad`'s zero set is a genuine curve/surface (nonzero `M`) rather
than a flat hyperplane -- determines whether a cut cell built from it needs
its `curve` field set, or can stay `nothing` (a straight edge) exactly like
every G1 cut.
"""
is_curved(quad::Quadric) = maximum(abs, quad.M) > 1e-12

"""
Whether `quad`'s zero set is a genuine pair of two *distinct* real lines
rather than one connected curve -- the case that arises in this codebase
exactly when both compared features are line-like (`K=N-1` vs `K=N-1`,
e.g. two different segments' own interior features): squared-distance-to-
a-line is already `(n·x-d)²`, so the bisector of two of them,
`(n_A·x-d_A)² = (n_B·x-d_B)²`, factors *exactly* into
`(n_A·x-d_A) = ±(n_B·x-d_B)` -- two straight lines through their common
intersection point, not one smooth branch (unlike, say, a point-vs-line
parabola, which is irreducible). Detected the standard way a conic is
degenerate: the augmented 3x3 matrix `[M b; bᵀ c]` is singular. Combined
with `M` being indefinite (one positive and one negative eigenvalue, found
directly from the 2x2 symmetric eigenvalue formula rather than pulling in
a general eigensolver) to rule out the *other* degenerate conics (a single
repeated line, a lone point, or no real solutions at all) -- none of which
arise from this codebase's own quadratic-distance differences, but the
distinction still matters since only the indefinite case is a genuine
pair of two distinct crossing lines.
"""
function is_line_pair(quad::Quadric{2,Float64}; atol=1e-9)
    M, b, c = quad.M, quad.b, quad.c
    det3 = M[1, 1] * (M[2, 2] * c - b[2]^2) - M[1, 2] * (M[1, 2] * c - b[2] * b[1]) + b[1] * (M[1, 2] * b[2] - M[2, 2] * b[1])
    mscale = max(1.0, maximum(abs, M))
    scale = max(1.0, mscale, abs(c), maximum(abs, b))^3
    abs(det3) > atol * scale && return false
    tr = M[1, 1] + M[2, 2]
    half_diff = (M[1, 1] - M[2, 2]) / 2
    spread = sqrt(half_diff^2 + M[1, 2]^2)
    eig_lo, eig_hi = tr / 2 - spread, tr / 2 + spread
    return eig_lo < -atol * mscale && eig_hi > atol * mscale
end

"""
Whether two points already known to lie on `quad` (a confirmed
`is_line_pair` conic) sit on the *same* one of its two lines --
mathematically exact, not a heuristic: two points determine a unique
straight line, and if that line is genuinely one of `quad`'s own two
factors, *every* point on it (not just `p`/`q` themselves) satisfies
`quad`'s equation -- including their midpoint, which is otherwise no more
special than any other point on the chord. If `p`/`q` are on *different*
lines of the pair, the chord between them generally isn't either line, so
its midpoint generally doesn't satisfy the equation. (This test is
specific to the pair-of-lines case -- it would wrongly reject a single
smooth branch like a parabola, whose chord midpoint is essentially never
on the curve itself; only call it once `is_line_pair` has already
confirmed the degenerate case.)
"""
function same_line_arc_endpoints(quad::Quadric{2,Float64}, p::Pt{2,Float64}, q::Pt{2,Float64}; atol=1e-9)
    mid = (p + q) / 2
    scale = max(1.0, maximum(abs, quad.M) * sum(abs2, mid), maximum(abs, quad.b) * norm(mid), abs(quad.c))
    return abs(evaluate(quad, mid)) <= atol * scale
end

"""
The two straight lines a confirmed line-pair conic (`is_line_pair(q)`
already true) factors into, each returned as an `(x0, t̂)` pair ready for
`line_meets_quadric`. Both lines pass through `q`'s own singular point
(where its gradient vanishes -- `M*p0 + b = 0`, a degenerate conic's
defining property, solved directly since `M` is invertible whenever the two
lines genuinely cross rather than run parallel). Their directions come from
`M`'s own indefinite eigendecomposition: writing `y = x - p0`, `q(p0+y)`
reduces to the homogeneous form `y'My` alone (the linear and constant terms
vanish exactly at the singular point), which for a 2x2 symmetric indefinite
`M` with eigenpairs `(μ1,e1)`,`(μ2,e2)` (`μ1>0>μ2`) is
`μ1(e1'y)² - |μ2|(e2'y)²`, a difference of squares factoring into
`(√μ1 e1 ± √|μ2| e2)'y = 0` -- the two lines' normals. Eigenvalues via the
same closed-form 2x2 formula `is_line_pair` already uses; eigenvectors
in the same style rather than pulling in a general eigensolver for a 2x2.
"""
function line_pair_factors(q::Quadric{2,Float64})
    p0 = -(q.M \ q.b)
    a, b12, d = q.M[1, 1], q.M[1, 2], q.M[2, 2]
    tr = a + d
    half_diff = (a - d) / 2
    spread = sqrt(half_diff^2 + b12^2)
    μ1, μ2 = tr / 2 + spread, tr / 2 - spread
    if abs(b12) > 1e-12 * max(1.0, abs(a), abs(d))
        e1 = SVector(b12, μ1 - a)
        e1 = e1 / norm(e1)
    else
        e1 = a >= d ? SVector(1.0, 0.0) : SVector(0.0, 1.0)
    end
    e2 = SVector(-e1[2], e1[1])
    s1, s2 = sqrt(max(μ1, 0.0)), sqrt(max(-μ2, 0.0))
    lines = Tuple{Pt{2,Float64},Pt{2,Float64}}[]
    for n in (s1 * e1 + s2 * e2, s1 * e1 - s2 * e2)
        t̂ = SVector(-n[2], n[1])
        push!(lines, (p0, t̂ / norm(t̂)))
    end
    return lines[1], lines[2]
end

"""
`curve`/`edge_curve` as homogeneous 3x3 symmetric matrices, `[M b; bᵀ c]`,
so that `[x,y,1] Q [x,y,1]ᵀ == evaluate(q, (x,y))` -- the standard conic
representation the pencil-based fallback below (`pencil_line_candidates`)
needs, since only in this form does "the conic through both `curve` and
`edge_curve`'s common points" become the simple linear family `Q1 + t*Q2`.
"""
conic_matrix3(q::Quadric{2,Float64}) = @SMatrix [
    q.M[1, 1] q.M[1, 2] q.b[1]
    q.M[1, 2] q.M[2, 2] q.b[2]
    q.b[1] q.b[2] q.c
]

"""
The real roots of `det(Q1 + t*Q2) = 0` -- where in the pencil of conics
through `Q1`,`Q2` a *degenerate* (rank ≤ 2) member sits (see
`pencil_line_candidates`'s docstring for why that's useful). `det(Q1+t*Q2)`
is a cubic in `t` (a 3x3 determinant, linear in each entry): its
coefficients are found by sampling at 4 points and solving the resulting
Vandermonde system directly (simpler and just as robust as expanding the
determinant symbolically, since this is only ever evaluated at Float64
run-time values, not needed in exact/interval form). A real cubic always
has at least one real root; found here via its companion matrix's
eigenvalues (a 3x3 non-symmetric `eigvals`, which `LinearAlgebra` handles
directly) rather than a hand-rolled Cardano formula. Falls back to treating
it as a quadratic/linear equation when the leading coefficient (`det(Q2)`)
is itself ~0, so a `Q2` that's already degenerate on its own doesn't
silently produce a spurious companion-matrix root at infinity.
"""
function pencil_real_roots(Q1::SMatrix{3,3,Float64}, Q2::SMatrix{3,3,Float64})
    ts = (0.0, 1.0, -1.0, 2.0)
    vals = [det(Q1 + t * Q2) for t in ts]
    V = [ts[i]^p for i in 1:4, p in 0:3]
    c0, c1, c2, c3 = V \ vals
    scale = max(abs(c0), abs(c1), abs(c2), abs(c3), 1.0)
    if abs(c3) < 1e-9 * scale
        if abs(c2) < 1e-9 * scale
            abs(c1) < 1e-12 * scale && return Float64[]
            return [-c0 / c1]
        end
        return quadratic_roots(c2, c1, c0)
    end
    p2, p1, p0 = c2 / c3, c1 / c3, c0 / c3
    companion = [0.0 0.0 -p0; 1.0 0.0 -p1; 0.0 1.0 -p2]
    return [real(λ) for λ in eigvals(companion) if abs(imag(λ)) < 1e-6 * max(1.0, abs(λ))]
end

"""
The line(s) (each an `(x0, t̂)` pair, ready for `line_meets_quadric`) that a
degenerate conic `Qt` (a real 3x3 matrix already known -- or assumed -- to
have `det(Qt) ≈ 0`) factors into: 2 lines for a genuine line pair
(indefinite quadratic part, `is_line_pair`), 1 for a repeated line (the
quadratic part is a rank-1 perfect square, `ℓ(x)²`, so *not* indefinite --
`is_line_pair` alone would wrongly call this "not a line pair" and reject
it), or 0 if `Qt` instead degenerates to a single real point or no real
locus at all (both eigenvalues of the quadratic part nonzero and
same-signed) -- geometrically possible for an arbitrary pencil member, just
not one that contributes any crossing candidates.
"""
function degenerate_conic_lines(Qt::Quadric{2,Float64})
    if !is_curved(Qt)
        return [line_parametrization(Qt)]
    end
    if is_line_pair(Qt)
        (x0a, t̂a), (x0b, t̂b) = line_pair_factors(Qt)
        return [(x0a, t̂a), (x0b, t̂b)]
    end
    M, b, c = Qt.M, Qt.b, Qt.c
    tr = M[1, 1] + M[2, 2]
    half_diff = (M[1, 1] - M[2, 2]) / 2
    spread = sqrt(half_diff^2 + M[1, 2]^2)
    μlo, μhi = tr / 2 - spread, tr / 2 + spread
    mscale = max(1.0, maximum(abs, M))
    # Repeated line: exactly one eigenvalue ~0, the other real and nonzero
    # (a rank-1 M with a definite, not indefinite, sign -- `ℓ(x)² = (n·x-d)²`
    # expands to exactly this: `M = n nᵀ` is PSD of rank 1). Not reached at
    # all unless `det(Qt) ≈ 0` already (guaranteed by construction, since
    # `Qt` is only ever built at an actual root of `pencil_real_roots`), so
    # a nonzero `μ` fully determines the line: its eigenvector `e` is `n`'s
    # own direction (up to scale/sign), and `b = -d*n` pins down `d`.
    if min(abs(μlo), abs(μhi)) < 1e-9 * mscale && max(abs(μlo), abs(μhi)) > 1e-9 * mscale
        μ = abs(μlo) > abs(μhi) ? μlo : μhi
        e = if abs(M[1, 2]) > 1e-12 * mscale
            v = SVector(M[1, 2], μ - M[1, 1])
            v / norm(v)
        else
            M[1, 1] >= M[2, 2] ? SVector(1.0, 0.0) : SVector(0.0, 1.0)
        end
        # `M = μ*e*eᵀ` (rank 1) forces `Qt(x) = μ*(e·x - d)²` for some `d`
        # (matching quadratic parts pins the line's direction to `e`
        # regardless of `μ`'s sign; matching the linear part then pins
        # `d = -dot(b,e)/μ`) -- an `M=0` quadric proportional to `e·x - d`
        # itself (not `Qt` -- that's rank 1, not yet the line alone),
        # ready for the same `line_parametrization` every other linear
        # case here already goes through.
        d = -dot(b, e) / μ
        return [line_parametrization(Quadric{2,Float64}(zeros(SMatrix{2,2,Float64}), e, -2d))]
    end
    return Tuple{Pt{2,Float64},Pt{2,Float64}}[]
end

"""
Real intersection points of two genuinely curved quadrics with different
`M` (`curve.M != edge_curve.M`). The fast paths below (`curve-edge_curve`,
`curve+edge_curve`, `curve`/`edge_curve` alone) all rely on one assumption:
that `edge_curve` and `curve` share a common "third feature" (the
tie-boundary invariant -- see the project notes on it), which makes one of
those four specific combinations collapse to a line or line pair
algebraically. That assumption *can* be violated: an edge's own `.curve`
field is fixed at the moment it's created (recording whichever two features
were being compared then), but a cell's current winner can later change via
a wholesale relabel (`insert_features!`'s `side == :b` branch, which
overwrites the *label* only, on purpose -- the cell's actual shape hasn't
changed, so there's nothing to reclip) without ever touching that edge. A
later insertion comparing the cell's *new* winner against some other new
feature can then hand `quadric_quadric_crossings` a `curve`/`edge_curve`
pair that shares no common feature at all -- confirmed as a real,
reachable failure (previously an outright `error`) via a minimal
random-stress reproduction.

Handled in general via the classical conic-pencil trick instead of relying
on that assumption: every conic `Q1 + t*Q2` in the pencil through `curve`
(`Q1`) and `edge_curve` (`Q2`) passes through all of their real common
points, and `det(Q1+t*Q2)` -- a cubic in `t` -- always has at least one
real root (`pencil_real_roots`), at which `Q1+t*Q2` is a *degenerate* conic
(`degenerate_conic_lines`): a line, a line pair, or (for a `t` that isn't
useful) a single point/no real locus. Every line found this way is
intersected with `edge_curve` (`line_meets_quadric`) to recover the actual
crossing points -- the same reduction the old fast paths above already use
for their own two specific pencil members (`t=-1`,`t=1`), just generalized
to not need `curve`/`edge_curve` to share anything. The original fast paths
are kept as cheap first attempts (avoiding a cubic solve in the common
case); the pencil search is the fallback once they've all failed, trying
every real root in turn (not just the first) since not every root's
conic need be a usable line/line-pair.
"""
function quadric_quadric_crossings(curve::Quadric{2,Float64}, edge_curve::Quadric{2,Float64}, p1::Pt{2,Float64}, p2::Pt{2,Float64}; strict::Bool=false)
    diff = Quadric{2,Float64}(curve.M - edge_curve.M, curve.b - edge_curve.b, curve.c - edge_curve.c)
    sum_ = Quadric{2,Float64}(curve.M + edge_curve.M, curve.b + edge_curve.b, curve.c + edge_curve.c)
    for cand in (diff, sum_)
        if !is_curved(cand)
            x0, t̂ = line_parametrization(cand)
            return line_meets_quadric(x0, t̂, edge_curve, p1, p2; strict=strict)
        end
    end
    dedup(pts) = let out = Pt{2,Float64}[]
        for p in pts
            any(q -> norm(p - q) < 1e-9 * max(1.0, norm(p)), out) || push!(out, p)
        end
        out
    end
    factor_source = if is_line_pair(diff)
        diff
    elseif is_line_pair(sum_)
        sum_
    elseif is_line_pair(curve)
        curve
    elseif is_line_pair(edge_curve)
        edge_curve
    else
        nothing
    end
    if factor_source !== nothing
        (x0a, t̂a), (x0b, t̂b) = line_pair_factors(factor_source)
        return dedup(vcat(line_meets_quadric(x0a, t̂a, edge_curve, p1, p2; strict=strict),
            line_meets_quadric(x0b, t̂b, edge_curve, p1, p2; strict=strict)))
    end

    Q1, Q2 = conic_matrix3(curve), conic_matrix3(edge_curve)
    pts = Pt{2,Float64}[]
    for t in pencil_real_roots(Q1, Q2)
        Qt = Q1 + t * Q2
        qt = Quadric{2,Float64}(SMatrix{2,2,Float64}(Qt[1, 1], Qt[2, 1], Qt[1, 2], Qt[2, 2]), SVector(Qt[1, 3], Qt[2, 3]), Qt[3, 3])
        for (x0, t̂) in degenerate_conic_lines(qt)
            append!(pts, line_meets_quadric(x0, t̂, edge_curve, p1, p2; strict=strict))
        end
    end
    return dedup(pts)
end

"""
Real roots of `A*s^2 + B*s + C = 0`, treating a repeated (or
near-repeated) root as *one* root, not two spuriously-distinct ones a few
ULPs apart. This matters more than it looks: two segments sharing an
endpoint routinely put that shared point exactly *on* the other segment's
own supporting line -- e.g. an L-shaped corner, the single most common
thing two connected segments do. That makes what would otherwise be a
parabola bisector between them degenerate into a repeated linear factor,
`(something)^2 = 0`, whose "two" roots from the general quadratic formula
are the exact same analytic point; the discriminant that should be exactly
zero for a perfect square instead rounds to a tiny positive Float64,
splitting one legitimate crossing into two near-duplicates a few ULPs
apart -- not a rare edge case, but the routine, expected result of drawing
any two segments that share a corner. Also treats a tiny *negative*
discriminant as the same repeated-root case (rather than "no real root at
all"), for the same reason in the other rounding direction.
"""
function quadratic_roots(A, B, C)
    if abs(A) < 1e-12 * max(1.0, abs(B), abs(C))
        abs(B) < 1e-15 && return Float64[]
        return [-C / B]
    end
    disc = B^2 - 4A * C
    scale = max(1.0, B^2, abs(4A * C))
    disc < -1e-9 * scale && return Float64[]
    sq = sqrt(max(disc, 0.0))
    t1, t2 = (-B - sq) / (2A), (-B + sq) / (2A)
    lo, hi = minmax(t1, t2)
    hi - lo < 1e-9 && return [(lo + hi) / 2]
    return [lo, hi]
end

"""
The roots `t ∈ (0,1)` where `quad` crosses the segment from `p1` to `p2`,
sorted ascending -- 0, 1, or 2 of them, since `evaluate(quad, lerp(p1,p2,t))`
is a (possibly degenerate) quadratic in `t`. Falls back to an exact linear
solve when the quadratic coefficient is (numerically) zero, so a genuinely
flat hyperplane quadric -- G1's only case -- is handled exactly as before,
not as a numerically-touchy special case of the general formula.
"""
function edge_crossings(quad::Quadric{N,Float64}, p1::Pt{N,Float64}, p2::Pt{N,Float64}) where {N}
    d = p2 - p1
    # Two ways this edge can be numerically degenerate relative to `quad`,
    # both arising from the same root cause (a vertex that's exactly some
    # multi-point circumcenter -- e.g. a quadruple point in 3D -- computed
    # via two different clip sequences that don't round identically):
    # either the edge itself is near-zero-length (both endpoints are
    # independent approximations of the *same* point), or the edge has real
    # length but both endpoints separately land within float noise of
    # `quad` (the whole edge lies almost exactly *in* the bisector, making
    # both the quadratic and linear coefficients below numerically
    # meaningless -- not just the direction vector). Either way there's no
    # geometrically meaningful precise crossing point to compute, so fall
    # back to the edge's own midpoint.
    v1, v2 = evaluate(quad, p1), evaluate(quad, p2)
    if norm(d) < 1e-9 || (abs(v1) < 1e-9 && abs(v2) < 1e-9)
        return [0.5]
    end
    Md = quad.M * d
    A = dot(d, Md)
    B = 2 * dot(p1, Md) + 2 * dot(quad.b, d)
    C = v1
    # Bounds allow a small epsilon past 0/1, not just inclusive at exactly
    # 0.0/1.0: a vertex can land (in floating point) exactly on a *later*
    # bisector even though the exact-predicate kernel correctly resolved it
    # as off-bisector when it was originally classified -- e.g. a triple
    # point (where 3 cells meet, the circumcenter of 3 points) is, by
    # construction, exactly on all 3 pairwise bisectors at once, so a vertex
    # created from one of those bisectors generically lands (in float)
    # exactly on the crossing point of another -- but the general quadratic
    # formula's own rounding can then put the computed root a few ULPs on
    # the *wrong* side of 0/1 even when the true root is exactly at the
    # boundary, so a hard `0.0 <= t <= 1.0` is too strict. Matches (and
    # slightly extends) the tolerance the old purely-linear
    # `hyperplane_crossing_point` had implicitly (it never checked `t` was
    # strictly inside `(0,1)` at all). The result is clamped back into
    # `[0,1]` so `quadric_crossing_point` never extrapolates past the edge.
    ε = 1e-9
    ts = sort(quadratic_roots(A, B, C))
    return [clamp(t, 0.0, 1.0) for t in ts if -ε <= t <= 1 + ε]
end

quadric_crossing_point(p1::Pt{N,Float64}, p2::Pt{N,Float64}, t::Float64) where {N} = p1 + t * (p2 - p1)

"""
An anchor point on `line`'s own zero set and a unit vector spanning it --
`line` must be linear (`line.M == 0`, checked by the caller via
`is_curved`). At `N=2` a linear quadric's zero set is a single line, fully
parametrized by one point and one direction.
"""
function line_parametrization(line::Quadric{2,Float64})
    n = line.b
    nn = norm(n)
    n̂ = n / nn
    d = -line.c / (2nn)
    x0 = d * n̂
    t̂ = SVector(-n̂[2], n̂[1])
    return x0, t̂
end

"""
Points where the line through `x0` in direction `t̂` crosses `Q` (any
`Quadric{2,Float64}`) -- `Q(x0 + s*t̂) = 0` is a quadratic in `s`, solved
directly. Filtered to those lying between `p1` and `p2` along `Q`'s own
natural single-valued axis (`curve_natural_axis`, `plot2d.jl`) -- *not* a
raw x/y coordinate range (an earlier version of this function used
whichever coordinate varies more between `p1` and `p2`, matching
`tessellate_curve`'s own old convention, and inherited the exact same
blind spot: a genuine crossing can have a raw coordinate outside
`p1`/`p2`'s own range whenever `Q`'s arc has a turning point between them,
getting wrongly filtered out here -- confirmed as the cause of a real
"edge has 0 crossings" construction failure). `strict` switches between
`edge_curve_crossings`'s inclusive epsilon-past-the-boundary tolerance and
`edge_curve_has_interior_crossing`'s tight non-inclusive one (see their
own docstrings for why each is needed). The shared core both of them
reduce to once they've each set up their own line and quadric to feed in
here.
"""
function line_meets_quadric(x0::Pt{2,Float64}, t̂::Pt{2,Float64}, Q::Quadric{2,Float64}, p1::Pt{2,Float64}, p2::Pt{2,Float64}; strict::Bool=false)
    axis = curve_natural_axis(Q)
    lo, hi = minmax(dot(axis, p1), dot(axis, p2))
    margin = 1e-9 * max(1.0, hi - lo)
    in_range(x) = let t = dot(axis, x)
        strict ? (lo + margin < t < hi - margin) : (lo - margin <= t <= hi + margin)
    end

    Mt = Q.M * t̂
    A = dot(t̂, Mt)
    B = 2 * dot(x0, Mt) + 2 * dot(Q.b, t̂)
    C = evaluate(Q, x0)

    ss = quadratic_roots(A, B, C)
    return [x0 + s * t̂ for s in ss if in_range(x0 + s * t̂)]
end

"""
The points where `curve` crosses the edge from `p1` to `p2`, where the edge
itself is the arc of `edge_curve` between them (`edge_curve === nothing`
means the edge is simply the straight chord, as in every G1 edge). This is
`edge_crossings` generalized to a possibly-curved *edge*, not just a
possibly-curved clip: `edge_crossings` alone parametrizes the edge as the
straight line through `p1`,`p2`, which silently gives the wrong answer once
an edge is genuinely curved (introduced by G2) -- the crossing needs to be
found on the edge's *true* shape instead.

Exact in two situations:
  - At least one of `curve`/`edge_curve` is linear: parametrize that one's
    own zero set (a line) with one parameter and substitute into the other
    (possibly curved) one's equation, giving one quadratic in that single
    parameter, solved directly.
  - Both are curved but share the *same* quadratic part (`M`): this happens
    routinely, not just by luck -- if `curve = bisector(f, new)` and
    `edge_curve = bisector(f, g)` (an edge always ties the cell's own
    winner `f` against its neighbor `g`, per the project's tie-boundary
    invariant), their difference is `bisector(g, new)`, and that shares
    `curve`/`edge_curve`'s `M` exactly whenever `g` and `new` do -- e.g.
    whenever both are point features, since every point feature has
    `M = I` regardless of *which* point, independent of what `f` is. In
    that case the difference is linear (the `M`s cancel), so the same
    parametrize-and-substitute trick applies to *that* line instead.

Otherwise (both curved, genuinely different `M`) is, in general, a harder
quadric-quadric intersection (up to 4 points via Bezout) -- but reached
here only in the one specific configuration this codebase's own feature
model can actually produce it in (two point-vs-line bisectors sharing only
their line, not their point), which reduces exactly to two line-vs-quadric
calls instead; see `quadric_quadric_crossings`'s own docstring for the
reduction.

Returns actual points (not `t`-parameters, since a curved edge's own
parametrization by `t ∈ [0,1]` along the chord doesn't apply), and, for the
near-degenerate cases, given the same epsilon tolerance and midpoint-
fallback treatment as `edge_crossings` (see its own docstring for why --
the identical triple-point/near-zero-length situations arise here too).
"""
function edge_curve_crossings(curve::Quadric{2,Float64}, p1::Pt{2,Float64}, p2::Pt{2,Float64}, edge_curve::Union{Nothing,Quadric{2,Float64}})
    (edge_curve === nothing || !is_curved(edge_curve)) && return quadric_crossing_point.(Ref(p1), Ref(p2), edge_crossings(curve, p1, p2))

    d = p2 - p1
    v1, v2 = evaluate(curve, p1), evaluate(curve, p2)
    if norm(d) < 1e-9 || (abs(v1) < 1e-9 && abs(v2) < 1e-9)
        return [(p1 + p2) / 2]
    end

    if !is_curved(curve)
        x0, t̂ = line_parametrization(curve)
        return line_meets_quadric(x0, t̂, edge_curve, p1, p2)
    elseif maximum(abs, curve.M - edge_curve.M) < 1e-9
        diff = Quadric{2,Float64}(curve.M - edge_curve.M, curve.b - edge_curve.b, curve.c - edge_curve.c)
        x0, t̂ = line_parametrization(diff)
        return line_meets_quadric(x0, t̂, edge_curve, p1, p2)
    else
        return quadric_quadric_crossings(curve, edge_curve, p1, p2)
    end
end

"""
Whether `curve` crosses the *interior* of the edge from `p1` to `p2` (whose
true shape is `edge_curve`, or the straight chord if `nothing`) -- the
`edge_curve_crossings` counterpart to `has_interior_crossing`, for the same
defensive "this edge looked unsplit, is it really?" check, generalized to a
possibly-curved edge (including the same-`M` elimination case -- see
`edge_curve_crossings`'s docstring). Strict (non-inclusive) bounds, unlike
`edge_curve_crossings`, so an endpoint landing exactly on `curve` doesn't
trip it.
"""
function edge_curve_has_interior_crossing(curve::Quadric{2,Float64}, p1::Pt{2,Float64}, p2::Pt{2,Float64}, edge_curve::Union{Nothing,Quadric{2,Float64}})
    (edge_curve === nothing || !is_curved(edge_curve)) && return has_interior_crossing(curve, p1, p2)

    d = p2 - p1
    (norm(d) < 1e-9 || (abs(evaluate(curve, p1)) < 1e-9 && abs(evaluate(curve, p2)) < 1e-9)) && return false

    if !is_curved(curve)
        x0, t̂ = line_parametrization(curve)
        return !isempty(line_meets_quadric(x0, t̂, edge_curve, p1, p2; strict=true))
    elseif maximum(abs, curve.M - edge_curve.M) < 1e-9
        diff = Quadric{2,Float64}(curve.M - edge_curve.M, curve.b - edge_curve.b, curve.c - edge_curve.c)
        x0, t̂ = line_parametrization(diff)
        return !isempty(line_meets_quadric(x0, t̂, edge_curve, p1, p2; strict=true))
    else
        return !isempty(quadric_quadric_crossings(curve, edge_curve, p1, p2; strict=true))
    end
end

"""
Dispatch wrapper `clip_by_hyperplane!` actually calls: at `N=2`, uses the
curve-aware `edge_curve_crossings`/`edge_curve_has_interior_crossing`
(handling both straight and curved existing edges correctly); at any other
`N`, falls back to the original straight-edge-only logic, which is exactly
correct there since G1's points-only construction -- the only thing that
runs at `N != 2` so far -- never produces a curved edge in the first place
(`edge_curve` is always `nothing`), so erroring if it somehow weren't is
the right defensive move rather than silently mishandling it.
"""
function edge_points_at_crossings(quad::Quadric{2,Float64}, p1::Pt{2,Float64}, p2::Pt{2,Float64}, edge_curve::Union{Nothing,Quadric{2,Float64}})
    return edge_curve_crossings(quad, p1, p2, edge_curve)
end
function edge_points_at_crossings(quad::Quadric{N,Float64}, p1::Pt{N,Float64}, p2::Pt{N,Float64}, edge_curve) where {N}
    edge_curve !== nothing && error("edge_points_at_crossings: curved edges are only supported at N=2 so far")
    return quadric_crossing_point.(Ref(p1), Ref(p2), edge_crossings(quad, p1, p2))
end

function edge_has_crossing_inside(quad::Quadric{2,Float64}, p1::Pt{2,Float64}, p2::Pt{2,Float64}, edge_curve::Union{Nothing,Quadric{2,Float64}})
    return edge_curve_has_interior_crossing(quad, p1, p2, edge_curve)
end
function edge_has_crossing_inside(quad::Quadric{N,Float64}, p1::Pt{N,Float64}, p2::Pt{N,Float64}, edge_curve) where {N}
    edge_curve !== nothing && error("edge_has_crossing_inside: curved edges are only supported at N=2 so far")
    return has_interior_crossing(quad, p1, p2)
end

"""
Whether `quad` genuinely separates `cell_id`'s interior into both signs --
`descendant_points`' vertex-only sign check alone answers this correctly
for a flat (hyperplane) `quad`, but not for a curved one: a curved bisector
can dip into and back out of a single edge (this file's whole multi-
crossing story -- see `clip_by_hyperplane!`'s docstring) while every one of
the cell's *vertices* still agrees on one side, since the sign only
actually flips partway along that edge's own interior. A curved bisector
can only ever enter a bounded simple polygon by crossing its boundary
somewhere (it's an open, unbounded curve, so it can't produce a sign
change purely in the interior without touching the boundary first) -- so
checking every boundary edge for an interior crossing, not just every
vertex, is enough to catch every case a pure vertex check would miss.
Always agrees with a vertex-only check when `quad` is flat or `N != 2`
(neither can produce a curved edge in the first place), so this is safe to
use as the one true test everywhere, not just as an extra curved-only
fallback.
"""
function cell_uniformly_signed(cx::CellComplex{N}, cell_id::Int, quad, sign_of) where {N}
    pts = descendant_points(cx, cell_id)
    signs = [sign_of(p) for p in pts]
    all_a = all(<=(0), signs)
    all_b = all(>=(0), signs)
    (all_a || all_b) || return :mixed
    if N == 2 && is_curved(quad)
        node = cx.nodes[cell_id]
        if node.dim == 2
            for e in node.subcells
                en = cx.nodes[e]
                v1, v2 = en.subcells
                edge_has_crossing_inside(quad, cx.nodes[v1].point, cx.nodes[v2].point, en.curve) && return :mixed
            end
        end
    end
    return all_a ? :a : :b
end

"""
Whether `quad` crosses the *interior* of the segment from `p1` to `p2`
(strict `0 < t < 1`, unlike `edge_crossings`'s inclusive bounds) -- used
only for the defensive "this edge looked unsplit, is it really?" check.
Deliberately stricter than `edge_crossings`: an edge endpoint landing
exactly on `quad` (the common triple-point situation `edge_crossings`
tolerates) must *not* trip this check, since that's not a second crossing
through the edge's middle, just the already-known endpoint coinciding with
`quad`.
"""
function has_interior_crossing(quad::Quadric{N,Float64}, p1::Pt{N,Float64}, p2::Pt{N,Float64}) where {N}
    d = p2 - p1
    # see edge_crossings' matching guard for why both of these are treated
    # as "no meaningful crossing to report" rather than run through the
    # numerically-unstable general solve below
    (norm(d) < 1e-9 || (abs(evaluate(quad, p1)) < 1e-9 && abs(evaluate(quad, p2)) < 1e-9)) && return false
    Md = quad.M * d
    A = dot(d, Md)
    B = 2 * dot(p1, Md) + 2 * dot(quad.b, d)
    C = evaluate(quad, p1)
    return any(t -> 1e-9 < t < 1 - 1e-9, quadratic_roots(A, B, C))
end

"""
The points (not just the yes/no of `edge_curve_has_interior_crossing`)
where `quad` crosses the strict *interior* of edge `(p1,p2,edge_curve)`,
sorted `p1`->`p2` -- used by `clip_top_cell_2d!` only for edges whose two
endpoints already agree on their side (the old code's `all_a`/`all_b`
shortcut, where an interior crossing meant "outside the generic case,
error"; here it instead means a genuine 0-or-2-crossing "bite", see
`clip_by_hyperplane!`'s docstring). Deliberately mirrors
`edge_curve_has_interior_crossing`/`has_interior_crossing`'s own strict,
non-inclusive tolerance (not `edge_points_at_crossings`'s inclusive one):
when the two endpoints agree, a crossing landing right at (or within noise
of) one of them isn't a real sign change at all, just numerical wobble
around a value that's already unambiguous -- unlike the *disagreeing*-
endpoints case (still handled the old way, byte for byte, in
`clip_top_cell_2d!` below), where an inclusive root landing near a vertex
tie-broken onto the "wrong" side is exactly the mechanism
`weld_near_duplicate_vertices!` exists to clean up after the fact, and
must be left alone here.
"""
function edge_strict_interior_crossings(quad::Quadric{2,Float64}, p1::Pt{2,Float64}, p2::Pt{2,Float64}, edge_curve::Union{Nothing,Quadric{2,Float64}})
    d = p2 - p1
    use_x = abs(d[1]) >= abs(d[2])
    key(x) = use_x ? x[1] : x[2]
    sgn = key(p2) >= key(p1) ? 1 : -1
    dosort(pts) = sort(pts, by=x -> sgn * key(x))

    if edge_curve === nothing || !is_curved(edge_curve)
        (norm(d) < 1e-9 || (abs(evaluate(quad, p1)) < 1e-9 && abs(evaluate(quad, p2)) < 1e-9)) && return Pt{2,Float64}[]
        Md = quad.M * d
        A = dot(d, Md)
        B = 2 * dot(p1, Md) + 2 * dot(quad.b, d)
        C = evaluate(quad, p1)
        ts = [t for t in quadratic_roots(A, B, C) if 1e-9 < t < 1 - 1e-9]
        return dosort([p1 + t * d for t in ts])
    end

    (norm(d) < 1e-9 || (abs(evaluate(quad, p1)) < 1e-9 && abs(evaluate(quad, p2)) < 1e-9)) && return Pt{2,Float64}[]
    if !is_curved(quad)
        x0, t̂ = line_parametrization(quad)
        return dosort(line_meets_quadric(x0, t̂, edge_curve, p1, p2; strict=true))
    elseif maximum(abs, quad.M - edge_curve.M) < 1e-9
        diff = Quadric{2,Float64}(quad.M - edge_curve.M, quad.b - edge_curve.b, quad.c - edge_curve.c)
        x0, t̂ = line_parametrization(diff)
        return dosort(line_meets_quadric(x0, t̂, edge_curve, p1, p2; strict=true))
    else
        return dosort(quadric_quadric_crossings(quad, edge_curve, p1, p2; strict=true))
    end
end

"""
Decomposes a 2D cell's `edges` (its full `subcells` list) into its separate
simple cycles, each in cyclic order as `(edge_id, from_vertex_id,
to_vertex_id)` triples -- the vertex-adjacency graph walk at the core of
`polygon_vertices_2d` (`plot2d.jl`), factored out here too since
`clip_top_cell_2d!` needs the same cyclic edge structure but not
`polygon_vertices_2d`'s further step of tessellating curves into sampled
points.

Almost always returns a single cycle (the cell's own outer boundary), but a
cell whose winning feature's territory has a *hole* -- another live cell's
region fully enclosed inside it, sharing no boundary with anything else --
has `edges` containing that hole's own boundary too, as a second, entirely
disjoint cycle. Nothing in this codebase's per-cell winner comparison rules
out that topology (a generalized-Voronoi region need not be simply
connected once curved bisectors are in play), so rather than assume one
cycle and fail on the rest (a real, previously-uncaught construction bug --
see the malformed-cycle investigation), this repeats the single-cycle walk
on whatever edges remain unused after each cycle closes, collecting every
one it finds.

`edges` can also contain a genuine "dangling tip" -- a chain of one or more
edges hanging off an otherwise-closed cycle at a vertex with no other live
incident edge in this set at all (confirmed, via a minimal random-stress
reproduction, to be reachable: a construction-history detail this file
doesn't fully trace elsewhere in the complex leaves such a stub referenced
by a cell's own `subcells` even though nothing else ever closes the loop
back through it). Such a spike is dropped before walking, not treated as an
error: a dangling appendage encloses no area on its own (it can't
contribute to a simple cycle by definition -- a true simple cycle visits
every vertex exactly twice), so silently excluding it from the returned
loops is the geometrically correct answer, not merely a tolerated
approximation of one. Pruning is iterative (removing one spike's own edge
can unmask another, shorter one behind it, e.g. a two-edge dangling chain)
and applied before the single-cycle walk below, which is otherwise
unchanged and still errors on any *other* reason a walk might fail to
close (a genuine topology bug elsewhere, not a plain dangling tip).
"""
function cyclic_boundary_walks(cx::CellComplex{2}, edges::Vector{Int})
    live = collect(edges)
    while true
        degree = Dict{Int,Int}()
        for e in live
            v1, v2 = cx.nodes[e].subcells
            degree[v1] = get(degree, v1, 0) + 1
            degree[v2] = get(degree, v2, 0) + 1
        end
        dangling = [e for e in live if any(v -> degree[v] == 1, cx.nodes[e].subcells)]
        isempty(dangling) && break
        live = [e for e in live if e ∉ dangling]
    end

    n = length(live)
    edge_verts = [(cx.nodes[e].subcells[1], cx.nodes[e].subcells[2]) for e in live]
    used = falses(n)
    loops = Vector{Tuple{Int,Int,Int}}[]
    for start in 1:n
        used[start] && continue
        used[start] = true
        order = Int[start]
        entry_vertex = edge_verts[start][1]
        cur_vertex = edge_verts[start][2]
        while cur_vertex != entry_vertex
            found_idx = 0
            for i in 1:n
                used[i] && continue
                a, b = edge_verts[i]
                if a == cur_vertex || b == cur_vertex
                    found_idx = i
                    break
                end
            end
            found_idx == 0 && error("cyclic_boundary_walks: boundary edges don't form a union of simple cycles")
            used[found_idx] = true
            push!(order, found_idx)
            a, b = edge_verts[found_idx]
            cur_vertex = (a == cur_vertex) ? b : a
        end
        out = Tuple{Int,Int,Int}[]
        walk_vertex = entry_vertex
        for idx in order
            a, b = edge_verts[idx]
            from, to = (a == walk_vertex) ? (a, b) : (b, a)
            push!(out, (live[idx], from, to))
            walk_vertex = to
        end
        push!(loops, out)
    end
    return loops
end

"""
Whether `pt` is inside the closed region bounded by `edges` (any list of
dim=1 nodes already known to form one simple cycle) -- `point_in_cell_2d`
(`plot2d.jl`)'s own even-odd ray-crossing core, factored out so
`clip_top_cell_2d!` can reuse it for a *not-yet-a-cell* boundary (a run's
own edges plus its not-yet-final closing arc) to decide which resulting
piece a hole belongs in, without needing a real stored `CellNode` to query
it through.
"""
function point_in_edge_loop(cx::CellComplex{2}, edges::Vector{Int}, pt::Pt{2,Float64})
    y0 = pt[2]
    crossings = 0
    for e in edges
        en = cx.nodes[e]
        v1, v2 = en.subcells
        p1, p2 = cx.nodes[v1].point, cx.nodes[v2].point
        if en.curve === nothing
            (p1[2] > y0) == (p2[2] > y0) && continue
            x_at_y = p1[1] + (p2[1] - p1[1]) * (y0 - p1[2]) / (p2[2] - p1[2])
            x_at_y > pt[1] && (crossings += 1)
        else
            for x in curve_crossings_at_y(en.curve, y0)
                x > pt[1] || continue
                on_arc_between(en.curve, p1, p2, SVector(x, y0)) && (crossings += 1)
            end
        end
    end
    return isodd(crossings)
end

"""
Which of `loops` (`cyclic_boundary_walks`' output, 2 or more -- a cell with
at least one hole) is the *outer* boundary rather than a hole: the one with
the largest bounding-box area, since a hole is by definition strictly
enclosed within its containing loop's own extent, hence strictly smaller.

Returns `nothing` if `loops` is empty -- reachable when every one of a
cell's own edges turns out to be part of a dangling spike with no genuine
closed cycle at all (`cyclic_boundary_walks` prunes all of them), meaning
this "cell" encloses no real area. Callers that assume a cell always has an
outer loop must check for this explicitly.
"""
function find_outer_loop(cx::CellComplex{2}, loops::Vector{Vector{Tuple{Int,Int,Int}}})
    isempty(loops) && return nothing
    function loop_bbox_area(w)
        pts = [cx.nodes[v].point for (_, v, _) in w]
        lo = reduce((a, b) -> min.(a, b), pts)
        hi = reduce((a, b) -> max.(a, b), pts)
        d = hi - lo
        return d[1] * d[2]
    end
    return argmax([loop_bbox_area(w) for w in loops])
end

"""
N=2 replacement for `clip_by_hyperplane!`'s general-purpose per-subcell
dictionary assembly, handling `start_id`'s own edges (dim=1) and its own
top-level split (dim=2) together in one pass -- see `clip_by_hyperplane!`'s
docstring for why multi-crossing needs this rather than the general-purpose
dim-by-dim loop.

Walks `start_id`'s boundary in cyclic order (`cyclic_boundary_walks`);
splits each edge at its own interior crossings with `quad` (0, 1, or 2 of
them -- the true ceiling here, since every crossing-finding routine this
codebase has solves at most a quadratic in one parameter); flattens the
whole boundary, now expressed edge-piece by edge-piece, into one cyclic
sequence of (side, piece) tuples; and cuts that sequence into maximal
same-side runs. Each run becomes its own new top-level cell, closed by a
fresh arc of `quad` connecting the run's own two cut vertices -- *not*
shared with any other run, unlike the old single-crossing case's one cut
edge shared by exactly two new cells, since a run's own closing arc
connects only *its* two endpoints. With no crossings anywhere, there's
exactly one run spanning the whole boundary, which reduces to leaving
`start_id` untouched -- the same outcome the old `all_a`/`all_b` shortcut
gave, just reached as the base case of this more general walk rather than
as a separately special-cased branch.

More than two runs (multiple same-side pieces, e.g. a curve dipping into
and back out of a single edge and carving an isolated "bite" cell) is
exactly the case the label/bbox infrastructure (`label_index`,
`cell_bbox`, `assert_label_bbox_invariant`) exists to support: several
live cells legitimately sharing one label, distinguished by disjoint
bounding boxes. Whether that disjointness assumption actually always holds
for every shape multi-crossing can produce hasn't been proven -- if a
future case violates it, `assert_label_bbox_invariant`/
`find_containing_cell`'s bbox fast-path will need a different mechanism;
this is a first pass at the general problem, not a closed case.

`orig_edges` occasionally decomposes (`cyclic_boundary_walks`) into more
than one loop -- `start_id` has a hole, another live cell's region fully
enclosed inside it. Only the *outer* loop (`find_outer_loop`) is walked for
crossings here: a hole's own boundary is exactly the enclosed cell's own
boundary too, so if `quad` actually cuts into it, that's already being
discovered independently by *that* cell's own `clip_top_cell_2d!`/
`clip_by_hyperplane!` call in this same insertion pass (every live cell is
clipped against `quad` on its own, per `insert_features!`'s own per-cell
loop) -- reprocessing it here would be redundant at best, and inconsistent
at worst if the two independent crossing computations don't round
identically. So each hole is instead carried through unchanged, attached
whole to whichever of the outer loop's resulting runs geometrically
contains it (`point_in_edge_loop`, checked against each run's own vertices
plus its now-created closing arc) -- if the outer loop doesn't split at
all, that's every hole staying with `start_id`, unchanged, exactly as
before this case was handled at all.

Returns `(results, extra_a, extra_b, extra_cut, pending)`: `results` is a
`Vector{Tuple{Bool,Int}}` of (is-A-side, cell-id) for every resulting
top-level cell (length 1, reusing `start_id` itself, when nothing
changed); `extra_a`/`extra_b`/`extra_cut` are every edge or top-cell id
(new or reused) that needs the corresponding label applied, exactly like
`a_piece`/`b_piece`/`cut_piece`'s values in the general-purpose code, just
not shoehorned into a one-id-per-original-id dict since a single original
edge can now expand into up to three pieces; `pending` is this call's own
`(old_id, new_ids)` supersessions, for the caller to append to its own
list (same deferred-until-the-whole-clip-finishes discipline as the
general-purpose code's `pending_supersessions`).
"""
function clip_top_cell_2d!(cx::CellComplex{2}, start_id::Int, quad::Quadric{2,Float64}, a_piece::Dict{Int,Int}, b_piece::Dict{Int,Int})
    orig_edges = cx.nodes[start_id].subcells
    loops = cyclic_boundary_walks(cx, orig_edges)
    if length(loops) == 1
        walk = loops[1]
        hole_walks = Vector{Tuple{Int,Int,Int}}[]
    else
        outer_idx = find_outer_loop(cx, loops)
        walk = loops[outer_idx]
        hole_walks = [loops[i] for i in eachindex(loops) if i != outer_idx]
    end

    extra_a = Int[]
    extra_b = Int[]
    extra_cut = Int[]
    pending = Tuple{Int,Vector{Int}}[]
    flat = Tuple{Bool,Int,Int,Int}[]   # (is_a, piece_edge_id, from_vertex, to_vertex), in cyclic walk order

    for (e, fv, tv) in walk
        enode = cx.nodes[e]
        p1id, p2id = enode.subcells[1], enode.subcells[2]
        p1, p2 = cx.nodes[p1id].point, cx.nodes[p2id].point
        s1, s2 = haskey(a_piece, p1id), haskey(a_piece, p2id)

        # Every interior crossing of `quad` along this edge flips the side,
        # so `s1` alone (alternating once per crossing) determines every
        # resulting piece's side -- for *any* number of crossings, not just
        # 0 or 2: this is just the intermediate value theorem applied
        # piecewise, true regardless of how many times a curved bisector
        # dips in and out (up to 4, by Bezout, for two genuine conics).
        # `s2` itself is never consulted for this: it's `a_piece`/`b_piece`'s
        # own, possibly-tie-broken (`symbolic_tiebreak`) side for `p2id`,
        # which can legitimately disagree with the alternation's own final
        # piece when `p2id` sits (numerically) exactly on `quad` itself --
        # a genuine other-tie there, arbitrarily broken one way, doesn't
        # change which side the strictly-interior geometry actually
        # occupies. Using strict crossings here (not `s1==s2`) to decide how
        # many cuts to make is what makes this robust to exactly that case,
        # which is confirmed the real cause of the previously-open
        # "N interior crossings while both endpoints agree" failures: not
        # literally more crossings than anticipated, but a tie-broken
        # endpoint whose arbitrary side doesn't match the true local parity.
        xs = edge_strict_interior_crossings(quad, p1, p2, enode.curve)
        if isempty(xs) && s1 != s2
            # The one case strict crossings can genuinely miss: the true
            # crossing sits so close to one endpoint that the strict
            # (non-inclusive) tolerance filters it out entirely. Falls back
            # to the inclusive finder, which still locates it (right where
            # `weld_near_duplicate_vertices!` expects to reconcile it
            # afterward) -- unchanged from this function's original
            # single-crossing behavior.
            crossings = edge_points_at_crossings(quad, p1, p2, enode.curve)
            if length(crossings) == 1
                xs = crossings
            else
                # No genuine interior crossing at all, inclusive tolerance
                # included -- by the intermediate value theorem, `s1 != s2`
                # despite that can only mean one endpoint's own *recorded*
                # side is itself an arbitrary tie-break (`symbolic_tiebreak`,
                # from the vertex-labeling loop above), not a reflection of
                # `quad`'s true local sign there: a genuine sign change
                # would force an actual crossing to exist somewhere
                # strictly between them, which was just confirmed absent.
                # The endpoint with the larger |value| is the one whose
                # side is trustworthy (further from the zero set, so its
                # sign isn't a coin flip); use that one for the whole
                # (unsplit) edge instead of erroring over the other's own
                # arbitrary tie-break disagreeing with it.
                v1, v2 = evaluate(quad, p1), evaluate(quad, p2)
                s1 = (abs(v1) >= abs(v2) ? v1 : v2) < 0
            end
        end

        if isempty(xs)
            pieces = [(s1, e, p1id, p2id)]
        else
            length(xs) <= 4 || error("clip_by_hyperplane!: edge $e has $(length(xs)) interior crossings -- two genuine quadrics can cross at most 4 times (Bezout), so this many indicates a bug in the crossing finder, not a valid configuration")
            cuts = [add_cell!(cx, 0, Label(), Int[], x) for x in xs]
            verts = vcat(p1id, cuts, p2id)
            pieces = Tuple{Bool,Int,Int,Int}[]
            for i in 1:length(verts)-1
                side = isodd(i) ? s1 : !s1
                pid = add_cell!(cx, 1, Label(), [verts[i], verts[i+1]]; curve=enode.curve)
                push!(pieces, (side, pid, verts[i], verts[i+1]))
            end
            push!(pending, (e, [p[2] for p in pieces]))
        end

        for (is_a, pid, _, _) in pieces
            push!(is_a ? extra_a : extra_b, pid)
        end
        append!(flat, fv == p1id ? pieces : [(is_a, pid, to, from) for (is_a, pid, from, to) in reverse(pieces)])
    end

    if all(p -> p[1] == flat[1][1], flat)
        side = flat[1][1]
        push!(side ? extra_a : extra_b, start_id)
        return [(side, start_id)], extra_a, extra_b, extra_cut, pending
    end

    n = length(flat)
    seam = findfirst(i -> flat[i][1] != flat[mod1(i - 1, n)][1], 1:n)
    runs = Vector{eltype(flat)}[]
    cur = [flat[seam]]
    for k in 1:n-1
        i = mod1(seam + k, n)
        if flat[i][1] == flat[mod1(i - 1, n)][1]
            push!(cur, flat[i])
        else
            push!(runs, cur)
            cur = [flat[i]]
        end
    end
    push!(runs, cur)

    arc_ids = Int[]
    for run in runs
        first_v, last_v = run[1][3], run[end][4]
        arc_curve = is_curved(quad) ? quad : nothing
        if arc_curve !== nothing && is_line_pair(quad)
            p_first, p_last = cx.nodes[first_v].point, cx.nodes[last_v].point
            same_line_arc_endpoints(quad, p_first, p_last) ||
                error("clip_by_hyperplane!: bisector is a genuine pair of two distinct lines (a line-vs-line comparison), and this run's own two cut points lie on *different* lines of that pair -- connecting them with a single curved edge would be wrong (there is no single line through them that stays on the bisector). This should be structurally unreachable: `clip_by_hyperplane!` dispatches every genuine K=N-1 vs K=N-1 (line-vs-line) comparison to `clip_by_line_pair!` before `quad` is ever computed here (confirmed never hit across the full test suite or extensive random-stress testing), so reaching this line at all means that dispatch invariant has been violated elsewhere -- an internal bug to fix at the source, not a case to special-case here.")
        end
        arc_id = add_cell!(cx, 1, Label(), [last_v, first_v]; curve=arc_curve)
        push!(extra_cut, arc_id)
        push!(arc_ids, arc_id)
    end

    run_extra_edges = [Int[] for _ in runs]
    for hw in hole_walks
        sample_pt = loop_interior_point(cx, hw)
        ri = findfirst(i -> point_in_edge_loop(cx, vcat([p[2] for p in runs[i]], arc_ids[i]), sample_pt), eachindex(runs))
        if ri === nothing
            # Can genuinely happen, not just an exact-predicate/rounding
            # near-miss: a hole is itself a separate live top cell, so it
            # can (in the same insertion round) be independently cut by
            # the very same bisector splitting its container here -- the
            # "carried through unchanged" assumption this function's own
            # docstring already flags as unproven. When that happens,
            # `hw` reflects the hole's *pre*-cut boundary, which may no
            # longer land cleanly inside any post-cut piece. Rather than
            # abort construction over a topology question this function
            # isn't set up to resolve properly (that needs deferring
            # reattachment until every cell this round has finished being
            # clipped, a larger restructuring than a local fix), fall back
            # to the run whose bounding-box center is nearest the hole's
            # own -- a reasonable best-effort placement, not a guess in
            # the dark, and warn so this is visible rather than silent.
            hole_pts = [cx.nodes[v].point for (_, v, _) in hw]
            hole_center = sum(hole_pts) / length(hole_pts)
            run_center(run) = sum(vcat([cx.nodes[p[3]].point for p in run], [cx.nodes[p[4]].point for p in run])) / (2 * length(run))
            ri = argmin([norm(run_center(run) - hole_center) for run in runs])
            @warn "clip_top_cell_2d!: a hole in start_id=$start_id's territory didn't land cleanly inside any of the $(length(runs)) resulting piece(s) (likely independently cut by this same insertion -- see the hole-placement investigation notes) -- falling back to the nearest piece by bounding-box center"
        end
        append!(run_extra_edges[ri], [e for (e, _, _) in hw])
    end

    results = Tuple{Bool,Int}[]
    for (ri, run) in enumerate(runs)
        cell_id = add_cell!(cx, 2, Label(), vcat([p[2] for p in run], arc_ids[ri], run_extra_edges[ri]))
        is_a = run[1][1]
        push!(is_a ? extra_a : extra_b, cell_id)
        push!(results, (is_a, cell_id))
    end
    push!(pending, (start_id, [cid for (_, cid) in results]))
    return results, extra_a, extra_b, extra_cut, pending
end

"""
All descendants of `start_id`, grouped by dimension (index `dim+1`),
including `start_id` itself -- the scope a single hyperplane clip should
touch, as opposed to the whole complex. Generalizes the pristine-bbox case
(where "the whole complex" and "descendants of the top cell" coincide) to
clipping one specific, already-existing cell within a larger complex.
"""
function descendant_nodes_by_dim(cx::CellComplex{N}, start_id::Int) where {N}
    start_dim = cx.nodes[start_id].dim
    by_dim = [Int[] for _ in 0:start_dim]
    seen = Set{Int}()
    stack = [start_id]
    while !isempty(stack)
        cur = pop!(stack)
        cur in seen && continue
        push!(seen, cur)
        node = cx.nodes[cur]
        push!(by_dim[node.dim+1], cur)
        append!(stack, node.subcells)
    end
    return by_dim
end

"""
Clip `start_id` (and everything below it) by the bisector of two features
`A` (vertex/sub-simplex `idxA`) and `B` (`idxB`): every descendant cell
ends up labeled by whichever of `A`/`B` is closer throughout it (or, on the
shared boundary, by both -- a genuine tie). `A`/`B` can be any
`AffineQuadratic` (not just points, i.e. `K=0`) -- G1 only ever compared
two points, giving a flat hyperplane bisector, but the exact same
dimension-by-dimension assembly below works unchanged for a genuinely
curved bisector (e.g. point-vs-segment-interior, a parabola in 2D): the
only things that actually depend on the bisector's shape are factored out
into `edge_crossings`/`quadric_crossing_point`, and the new cut cell this
produces at each split records that curved shape (`curve`) rather than
assuming a straight chord. `start_id` need not be the whole complex's only
top cell -- this clips just its own descendants, scoped via
`descendant_nodes_by_dim`, which is what makes it safe to call repeatedly
against different, already-partially-built cells of a larger complex (the
multi-point/multi-segment incremental case) rather than only once against
a pristine bounding box.

Processes dimension `0, 1, ..., dim(start_id)` in order, reusing each
dimension's already-computed split to determine the next: a cell is left
unsplit (keeping its id) if all its immediate subcells ended up entirely
on the same side; otherwise it's replaced by two new cells (one per side)
sharing a newly-created cut cell one dimension down. Every split node is
immediately `supersede!`d, which patches any *other*, pre-existing cell
that also referenced it as a subcell (e.g. a neighboring cell sharing that
boundary) -- see `supersede!`'s own docstring for why this keeps the
complex locally consistent without a separate reconciliation pass.

Assumes the generic case in one remaining sense: no existing vertex lies
exactly on the bisector (a real but currently unhandled degenerate case,
deliberately deferred -- see project notes on symbolic tie-breaking for how
it should eventually be resolved). A single edge of the existing skeleton
being crossed by the bisector more than once -- a real configuration for a
genuinely curved bisector -- *is* handled, at `N=2`, via
`clip_top_cell_2d!`; see its own docstring for how. Any other `N`, or a
crossing count `clip_top_cell_2d!` itself can't make sense of (more than
2, or inconsistent with the edge's own endpoint sides), still falls back to
this function's own general-purpose dimension-by-dimension assembly below,
which keeps its original one-crossing-only limit.
"""
function clip_by_hyperplane!(cx::CellComplex{N}, start_id::Int, A::AffineQuadratic{N,KA,Float64}, B::AffineQuadratic{N,KB,Float64}, idxA::Set{VertexIdx}, idxB::Set{VertexIdx}; preserve_label::Union{Nothing,Label}=nothing) where {N,KA,KB}
    # Two line-like features (K=N-1, e.g. two different segments' own
    # interior features) never actually have a single-branch bisector --
    # see `clip_by_line_pair!`'s own docstring for why it's always exactly
    # two straight lines -- so this whole function's "one curved bisector"
    # assumption doesn't apply; dispatch to the dedicated two-flat-clips
    # handling instead. Never fires under `preserve_label` (pure geometric
    # refinement, e.g. `insert_own_lines!`'s own endpoint cuts, never
    # compares two real line features against each other in the first
    # place).
    if N == 2 && KA == N - 1 && KB == N - 1 && preserve_label === nothing
        return clip_by_line_pair!(cx, start_id, A, B, idxA, idxB)
    end
    quad = bisector(A, B)
    a_piece = Dict{Int,Int}()   # original node id -> id of the (possibly new) node entirely on A's side
    b_piece = Dict{Int,Int}()
    cut_piece = Dict{Int,Int}() # original node id -> id of the new (dim-1) shared-boundary cell, only for split nodes
    # (old_id, [a_id,b_id]) pairs to `supersede!` -- deferred to *after* the
    # whole clip finishes (see the docstring of `supersede!` calls below for
    # why: this operation's own not-yet-processed higher-dimension nodes are
    # themselves pre-existing parents of the lower-dimension ones being split
    # right now, and patching them mid-loop would rewrite their `subcells`
    # out from under this same function's own later reads of them, before
    # its `a_piece`/`b_piece` bookkeeping (keyed by the *original* ids) ever
    # gets a chance to look them up).
    pending_supersessions = Tuple{Int,Vector{Int}}[]
    # `a_piece`/`b_piece`/`cut_piece` above are keyed one-id-per-original-id,
    # which the N=2 multi-crossing path below can't respect (a single edge
    # can expand into up to three new pieces, and a single top cell into
    # more than two) -- these three collect every id from that path needing
    # the corresponding label, alongside (not instead of) the dicts above.
    extra_a = Int[]
    extra_b = Int[]
    extra_cut = Int[]

    by_dim = descendant_nodes_by_dim(cx, start_id)

    for id in by_dim[1]
        node = cx.nodes[id]
        s = exact_sign((q, x) -> evaluate(q, x), quad, node.point)
        # A genuine exact tie (not just numerically close -- `exact_sign`'s
        # own exact fallback already ruled that out): both "this vertex
        # belongs to A" and "...to B" are equally valid at a single,
        # measure-zero point, so pick one *consistently* via the same
        # deterministic rule every time this exact configuration is seen
        # (whether from this call or a different one touching the same
        # vertex against the same two faces) -- the standard robust-
        # geometry answer (symbolic perturbation), not an error. The
        # resulting split is a fully valid, self-consistent complex; the
        # only cost is that this single vertex's own label reflects just
        # the winning side, not the tie.
        s = s == 0 ? symbolic_tiebreak(sort(collect(idxA)), sort(collect(idxB))) : s
        if s < 0
            a_piece[id] = id
        else
            b_piece[id] = id
        end
    end

    if N == 2 && cx.nodes[start_id].dim == 2
        results, ea, eb, ec, pending = clip_top_cell_2d!(cx, start_id, quad, a_piece, b_piece)
        append!(pending_supersessions, pending)
        append!(extra_a, ea)
        append!(extra_b, eb)
        append!(extra_cut, ec)
        # Only used for the legacy scalar return value below -- when
        # multiple runs land on the same side, whichever is processed last
        # wins the dict slot; every genuine result (including the others)
        # is still labeled correctly via `extra_a`/`extra_b` above.
        for (is_a, cid) in results
            if is_a
                a_piece[start_id] = cid
            else
                b_piece[start_id] = cid
            end
        end
        isempty(extra_cut) || (cut_piece[start_id] = first(extra_cut))
    else
        for dim in 1:(cx.nodes[start_id].dim)
            for id in by_dim[dim+1]
                node = cx.nodes[id]
                subs = node.subcells
                split_subs = [s for s in subs if haskey(cut_piece, s)]
                all_a = all(s -> haskey(a_piece, s) && !haskey(cut_piece, s), subs)
                all_b = all(s -> haskey(b_piece, s) && !haskey(cut_piece, s), subs)

                if isempty(split_subs) && all_a
                    dim == 1 && edge_has_crossing_inside(quad, cx.nodes[subs[1]].point, cx.nodes[subs[2]].point, node.curve) &&
                        error("clip_by_hyperplane!: an edge with both endpoints on A's side is nonetheless crossed by the bisector internally -- this general-purpose fallback only supports at most one crossing per edge (a curved bisector crossing one edge twice is a genuinely generic configuration, just not one this path handles -- see `clip_top_cell_2d!` at N=2)")
                    a_piece[id] = id
                elseif isempty(split_subs) && all_b
                    dim == 1 && edge_has_crossing_inside(quad, cx.nodes[subs[1]].point, cx.nodes[subs[2]].point, node.curve) &&
                        error("clip_by_hyperplane!: an edge with both endpoints on B's side is nonetheless crossed by the bisector internally -- this general-purpose fallback only supports at most one crossing per edge (a curved bisector crossing one edge twice is a genuinely generic configuration, just not one this path handles -- see `clip_top_cell_2d!` at N=2)")
                    b_piece[id] = id
                else
                    # The new (dim-1)-dimensional cut cell within `id` itself,
                    # computed *before* assembling `id`'s own two new pieces --
                    # it's a subcell of *both* of them (their shared boundary).
                    if dim == 1
                        p1, p2 = cx.nodes[subs[1]].point, cx.nodes[subs[2]].point
                        crossings = edge_points_at_crossings(quad, p1, p2, node.curve)
                        length(crossings) == 1 || error("clip_by_hyperplane!: edge has $(length(crossings)) crossings (expected exactly 1) -- this general-purpose fallback doesn't handle more")
                        x = only(crossings)
                        cut_id = add_cell!(cx, 0, Label(), Int[], x)
                    else
                        cut_id = add_cell!(cx, dim - 1, Label(), split_subs_to_cut_boundary(cut_piece, split_subs); curve=(is_curved(quad) ? quad : nothing))
                    end
                    cut_piece[id] = cut_id

                    # Each of `id`'s own subcells contributes to `id`'s new
                    # pieces via its *own* same-dimension a/b split (not its
                    # lower-dimensional cut trace, which is a different thing
                    # entirely -- a subcell that was itself split contributes
                    # both of its own two new pieces, one to each side).
                    a_subs = Int[cut_id]
                    b_subs = Int[cut_id]
                    for s in subs
                        if haskey(cut_piece, s)
                            push!(a_subs, a_piece[s])
                            push!(b_subs, b_piece[s])
                        elseif haskey(a_piece, s)
                            push!(a_subs, a_piece[s])
                        else
                            push!(b_subs, b_piece[s])
                        end
                    end
                    a_id = add_cell!(cx, dim, Label(), a_subs)
                    b_id = add_cell!(cx, dim, Label(), b_subs)
                    a_piece[id] = a_id
                    b_piece[id] = b_id
                    push!(pending_supersessions, (id, [a_id, b_id]))
                end
            end
        end
    end

    for (old_id, new_ids) in pending_supersessions
        supersede!(cx, old_id, new_ids)
    end

    # `a_piece`/`b_piece`'s *values* mix two genuinely different things that
    # happen to share one dict: brand-new dim>=1 pieces this call just
    # created (which genuinely do belong under `label_a`/`label_b`/
    # `preserve_label`), and dim=0 *pre-existing* vertices, self-mapped
    # (`a_piece[id] = id`) purely as side-of-`quad` bookkeeping for
    # assembling those new pieces. The label this loop gives such a vertex
    # is therefore only ever provisional -- correct in the common case
    # (nothing else was ever tied there), but a vertex that's also, say, a
    # neighbor's own boundary tie needs its *true* label recomputed from its
    # actual position, not just this one pairwise comparison's own verdict;
    # see `insert_point!`/`insert_features!`'s own `extra_verts` argument to
    # `weld_near_duplicate_vertices!`, which is what actually guarantees
    # every touched vertex ends up correct, not this assignment alone
    # (confirmed as a real, reachable gap via random-stress testing, not
    # just a theoretical one).
    if preserve_label === nothing
        label_a = Label([idxA])
        label_b = Label([idxB])
        label_cut = Label([idxA, idxB])
        for id in Set(values(a_piece)) ∪ Set(extra_a)
            set_label!(cx, id, label_a)
        end
        for id in Set(values(b_piece)) ∪ Set(extra_b)
            set_label!(cx, id, label_b)
        end
        for id in Set(values(cut_piece)) ∪ Set(extra_cut)
            set_label!(cx, id, cx.nodes[id].label ∪ label_cut)
        end
    else
        # Pure refinement (see `clip_by_plane_preserving_label!`): every
        # piece this call creates, at *every* dimension level (not just the
        # top one) genuinely belongs to `start_id`'s own original label --
        # `idxA`/`idxB` are just placeholders for defining the cut plane
        # geometrically and carry no real tie meaning here, so deriving a
        # label from them (as the branch above does) would taint
        # intermediate substructure with a spurious tie that nothing ever
        # corrects.
        for id in Set(values(a_piece)) ∪ Set(values(b_piece)) ∪ Set(values(cut_piece)) ∪ Set(extra_a) ∪ Set(extra_b) ∪ Set(extra_cut)
            set_label!(cx, id, preserve_label)
        end
    end

    return get(a_piece, start_id, nothing), get(b_piece, start_id, nothing), get(cut_piece, start_id, nothing)
end

"""
The subcells of a newly-created cut cell at dimension `dim-1 >= 1`: just
the cut cells of whichever of the current cell's own subcells were split
-- i.e. `split_subs` mapped through `cut_piece`, which is exactly what's
already computed by the time this is needed. Kept as a tiny named helper
only so the call site above reads clearly.
"""
split_subs_to_cut_boundary(cut_piece, split_subs) = Int[cut_piece[s] for s in split_subs]
