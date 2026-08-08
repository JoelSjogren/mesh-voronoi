"""Whether `quad`'s zero set is a genuine curve (nonzero `M`) rather than a flat hyperplane."""
is_curved(quad::Quadric) = maximum(abs, quad.M) > 1e-12

"""
Whether `quad`'s zero set is a genuine pair of two distinct real lines
rather than one connected curve -- arises when both compared features are
line-like: `(n_A·x-d_A)² = (n_B·x-d_B)²` factors exactly into
`(n_A·x-d_A) = ±(n_B·x-d_B)`, two lines through their intersection.
Detected via the augmented 3x3 matrix `[M b; bᵀ c]` being singular,
combined with `M` indefinite (one positive, one negative eigenvalue) to
rule out the other degenerate conics (repeated line, single point).
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
mathematically exact: if their chord is genuinely one of `quad`'s two
factors, every point on it (including the midpoint) satisfies `quad`'s
equation; if `p`/`q` are on different lines, the chord's midpoint
generally doesn't. Specific to the pair-of-lines case -- only call once
`is_line_pair` has confirmed it.
"""
function same_line_arc_endpoints(quad::Quadric{2,Float64}, p::Pt{2,Float64}, q::Pt{2,Float64}; atol=1e-9)
    mid = (p + q) / 2
    scale = max(1.0, maximum(abs, quad.M) * sum(abs2, mid), maximum(abs, quad.b) * norm(mid), abs(quad.c))
    return abs(evaluate(quad, mid)) <= atol * scale
end

"""
The two straight lines a confirmed line-pair conic (`is_line_pair(q)`
true) factors into, each `(x0, t̂)` ready for `line_meets_quadric`. Both
pass through `q`'s singular point (`M*p0+b=0`, solved directly since `M`
is invertible when the lines genuinely cross). Directions come from `M`'s
indefinite eigendecomposition: `y=x-p0` reduces `q(p0+y)` to the
homogeneous `μ1(e1'y)² - |μ2|(e2'y)²`, factoring into
`(√μ1 e1 ± √|μ2| e2)'y = 0` -- the two lines' normals.
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
`q` as a homogeneous 3x3 symmetric matrix `[M b; bᵀ c]`, so
`[x,y,1]Q[x,y,1]ᵀ == evaluate(q,(x,y))` -- the standard conic
representation the pencil-based fallback needs, since only in this form
is "the conic through both curves' common points" the linear family
`Q1 + t*Q2`.
"""
conic_matrix3(q::Quadric{2,Float64}) = @SMatrix [
    q.M[1, 1] q.M[1, 2] q.b[1]
    q.M[1, 2] q.M[2, 2] q.b[2]
    q.b[1] q.b[2] q.c
]

"""
Real roots of `det(Q1 + t*Q2) = 0` -- where in the pencil of conics
through `Q1`,`Q2` a degenerate (rank<=2) member sits. Cubic in `t`;
coefficients found by sampling 4 points and solving the Vandermonde
system directly. Real root found via the companion matrix's eigenvalues.
Falls back to quadratic/linear when the leading coefficient (`det(Q2)`)
is itself ~0.
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
The line(s) a degenerate conic `Qt` (`det(Qt)≈0`) factors into: 2 for a
genuine line pair (`is_line_pair`), 1 for a repeated line (rank-1 perfect
square, not indefinite, so `is_line_pair` alone would wrongly reject it),
or 0 if `Qt` degenerates to a single point or no real locus (both
eigenvalues nonzero, same-signed) -- possible for an arbitrary pencil
member, just contributes no crossing candidates.
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
    # Repeated line: exactly one eigenvalue ~0, the other nonzero (rank-1
    # M with a definite sign -- `ℓ(x)²=(n·x-d)²` expands to `M=n nᵀ`, PSD
    # rank 1). A nonzero `μ` fully determines the line: eigenvector `e` is
    # `n`'s own direction, `b=-d*n` pins down `d`.
    if min(abs(μlo), abs(μhi)) < 1e-9 * mscale && max(abs(μlo), abs(μhi)) > 1e-9 * mscale
        μ = abs(μlo) > abs(μhi) ? μlo : μhi
        e = if abs(M[1, 2]) > 1e-12 * mscale
            v = SVector(M[1, 2], μ - M[1, 1])
            v / norm(v)
        else
            M[1, 1] >= M[2, 2] ? SVector(1.0, 0.0) : SVector(0.0, 1.0)
        end
        # `M = μ*e*eᵀ` forces `Qt(x) = μ*(e·x-d)²`; matching parts pins
        # `d = -dot(b,e)/μ`, giving an `M=0` quadric proportional to `e·x-d`.
        d = -dot(b, e) / μ
        return [line_parametrization(Quadric{2,Float64}(zeros(SMatrix{2,2,Float64}), e, -2d))]
    end
    return Tuple{Pt{2,Float64},Pt{2,Float64}}[]
end

"""
Real intersection points of two genuinely curved quadrics with different
`M`. The fast paths below assume `edge_curve`/`curve` share a common
"third feature" (the tie-boundary invariant), collapsing one of a few
specific combinations to a line/line pair. That can be violated -- an
edge's own `.curve` is fixed at creation, but a cell's current winner can
later change via a wholesale relabel without touching that edge, so a
later comparison can hand this a pair sharing no common feature at all
(confirmed reachable, previously an outright error).

Handled in general via the conic-pencil trick: every conic `Q1+t*Q2`
through `curve`,`edge_curve` passes through their real common points, and
`det(Q1+t*Q2)` (cubic in `t`) always has a real root (`pencil_real_roots`)
where it's degenerate (`degenerate_conic_lines`) -- a line or line pair,
intersected with `edge_curve` to recover crossings. The fast paths are
kept as cheap first attempts; the pencil search is the fallback, trying
every real root since not every one gives a usable line.
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
sorted ascending -- 0, 1, or 2, since `evaluate(quad, lerp(p1,p2,t))` is a
(possibly degenerate) quadratic in `t`. Falls back to an exact linear
solve when the quadratic coefficient is numerically zero.
"""
function edge_crossings(quad::Quadric{N,Float64}, p1::Pt{N,Float64}, p2::Pt{N,Float64}) where {N}
    d = p2 - p1
    # Two ways this edge can be numerically degenerate relative to `quad`
    # (both from a vertex that's exactly some multi-point circumcenter,
    # computed via two clip sequences that don't round identically):
    # near-zero-length, or real length but both endpoints land within
    # float noise of `quad` (the edge lies almost exactly in the
    # bisector). Either way, fall back to the edge's own midpoint.
    v1, v2 = evaluate(quad, p1), evaluate(quad, p2)
    if norm(d) < 1e-9 || (abs(v1) < 1e-9 && abs(v2) < 1e-9)
        return [0.5]
    end
    Md = quad.M * d
    A = dot(d, Md)
    B = 2 * dot(p1, Md) + 2 * dot(quad.b, d)
    C = v1
    # Bounds allow a small epsilon past 0/1, not just inclusive at exactly
    # 0.0/1.0: a vertex can land (in float) exactly on a *later* bisector
    # (e.g. a triple point sits exactly on 3 pairwise bisectors at once),
    # and the quadratic formula's own rounding can put the computed root a
    # few ULPs on the wrong side of 0/1 even when the true root is exactly
    # at the boundary. Clamped back into `[0,1]` afterward so
    # `quadric_crossing_point` never extrapolates past the edge.
    ε = 1e-9
    ts = sort(quadratic_roots(A, B, C))
    return [clamp(t, 0.0, 1.0) for t in ts if -ε <= t <= 1 + ε]
end

quadric_crossing_point(p1::Pt{N,Float64}, p2::Pt{N,Float64}, t::Float64) where {N} = p1 + t * (p2 - p1)

"""Anchor point and unit direction spanning `line`'s zero set (must be linear, `line.M == 0`)."""
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
Points where the line through `x0` in direction `t̂` crosses `Q` --
`Q(x0+s*t̂)=0` is a quadratic in `s`, solved directly. Filtered to those
between `p1`,`p2` along `Q`'s own natural single-valued axis
(`curve_natural_axis`), not a raw x/y range (which wrongly filters a
crossing whenever `Q`'s arc has a turning point between `p1`,`p2` --
confirmed as a real "edge has 0 crossings" failure cause). `strict`
switches between `edge_curve_crossings`'s inclusive tolerance and
`edge_curve_has_interior_crossing`'s tight non-inclusive one.
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
The points where `curve` crosses the edge from `p1` to `p2`, where the
edge itself is the arc of `edge_curve` (`nothing` means a straight chord).
`edge_crossings` generalized to a possibly-curved *edge*, not just a
possibly-curved clip.

Exact in two situations: at least one of `curve`/`edge_curve` is linear
(parametrize that line, substitute into the other's equation, one
quadratic); or both are curved but share the same `M` (routine, not
lucky: if `curve=bisector(f,new)` and `edge_curve=bisector(f,g)`, their
difference is `bisector(g,new)`, sharing `M` whenever `g`,`new` do, e.g.
both point features with `M=I`) -- the difference is then linear and the
same trick applies.

Otherwise (genuinely different `M`) reaches `quadric_quadric_crossings`'s
own general pencil reduction.

Returns actual points, not `t`-parameters, with the same epsilon/midpoint
fallback treatment as `edge_crossings` for near-degenerate cases.
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
Whether `curve` crosses the *interior* of the edge from `p1` to `p2`
(true shape `edge_curve`, or straight chord if `nothing`) -- the
`edge_curve_crossings` counterpart to `has_interior_crossing`, for the
"this edge looked unsplit, is it really?" check. Strict (non-inclusive)
bounds, so an endpoint landing exactly on `curve` doesn't trip it.
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
Dispatch wrapper `clip_by_hyperplane!` calls: at `N=2`, the curve-aware
`edge_curve_crossings`/`edge_curve_has_interior_crossing`; at other `N`,
the straight-edge-only logic (errors if a curved edge somehow reached it,
since points-only construction never produces one).
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
a vertex-only sign check alone is correct for a flat `quad`, but not a
curved one: it can dip into and back out of a single edge while every
vertex still agrees on one side. A curved bisector can only enter a
bounded simple polygon by crossing its boundary, so checking every
boundary edge for an interior crossing (not just vertices) catches every
case a vertex-only check would miss. Always agrees with a vertex-only
check when `quad` is flat or `N != 2`, so safe to use everywhere.
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
Shared cleanup for both `cyclic_boundary_walks` (1D boundary -> simple
cycles) and `trace_surface_faces` (2D surface boundary -> half-edge
tracing -- a different, not interchangeable question, see its own
docstring). Both need the same two passes first:

Iteratively prunes "dangling tips" -- edges hanging off an otherwise-
closed structure with no other live incident edge (confirmed reachable: a
construction-history detail neither caller fully traces can leave such a
stub). Encloses no area, can't be part of any cycle/face, so dropping it
is exact, not approximate. Iterative since removing one spike can unmask
a shorter one behind it.

Then, only if some vertex has degree > 2: canonicalizes near-duplicate
vertices -- ids a few ULPs apart that are really the same point, not yet
welded (this codebase's normal weld can run too late for structure a
single clip needs mid-construction). Proper union-find, since duplicates
can chain transitively. Canonicalizing can turn an edge into a degenerate
`[v,v]` self-loop, dropped, re-stranding something else, so dangling-
pruning repeats. `cx` itself is never mutated (both callers are shared,
read-only-ish utilities).

Returns `(live, canon)`: `edges` with dangling/degenerate pieces removed,
and the near-duplicate-merged vertex map (identity in the common case).
"""
function prepare_boundary_edges(cx::CellComplex{N}, edges::Vector{Int}) where {N}
    pt(v) = cx.nodes[v].point
    # A genuine self-loop (both raw endpoints already the same id, not
    # merely near-duplicate) carries no real topology and, left in,
    # corrupts a branch vertex's rotation list (inserted twice, silently
    # discarding one position), sending `trace_surface_faces`'s tracer
    # into a runaway loop (confirmed via a real repro). The near-duplicate
    # canonicalization below only drops a self-loop it *creates* --
    # this drops one that already existed, unconditionally.
    live = [e for e in edges if cx.nodes[e].subcells[1] != cx.nodes[e].subcells[2]]
    degree = Dict{Int,Int}()
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

    canon = Dict(v => v for v in keys(degree))
    if any(>(2), values(degree))
        verts = collect(keys(degree))
        function find_root(x)
            while canon[x] != x
                x = canon[x]
            end
            return x
        end
        for i in eachindex(verts), j in (i+1):length(verts)
            a, b = verts[i], verts[j]
            pa, pb = pt(a), pt(b)
            norm(pa - pb) <= 1e-9 * max(1.0, norm(pa), norm(pb)) || continue
            ra, rb = find_root(a), find_root(b)
            ra != rb && (canon[max(ra, rb)] = min(ra, rb))
        end
        for v in verts
            canon[v] = find_root(v)
        end
        if any(v -> canon[v] != v, verts)
            cend(e) = (canon[cx.nodes[e].subcells[1]], canon[cx.nodes[e].subcells[2]])
            live = [e for e in live if cend(e)[1] != cend(e)[2]]
            while true
                degree2 = Dict{Int,Int}()
                for e in live
                    v1, v2 = cend(e)
                    degree2[v1] = get(degree2, v1, 0) + 1
                    degree2[v2] = get(degree2, v2, 0) + 1
                end
                dangling = [e for e in live if any(v -> degree2[v] == 1, cend(e))]
                isempty(dangling) && break
                live = [e for e in live if e ∉ dangling]
            end
        end
    end
    return live, canon
end

"""
Decomposes a 2D *cell*'s `edges` into its separate simple cycles, each in
cyclic order as `(edge_id, from_vertex_id, to_vertex_id)` triples --
factored out of `polygon_vertices_2d` (`plot2d.jl`) since
`clip_top_cell_2d!` needs the same cyclic structure without tessellation.

Valid only for a genuinely 1-dimensional boundary (a 2D cell's own
boundary at any `N`, or an `N=2` top cell), which must decompose into
simple closed curves. An `N=3` *face*'s own boundary is different -- it
can develop a genuine branch point after enough clips, where no such
decomposition exists at all (see `trace_surface_faces`). Callers walking
a face's own `subcells` should use that instead.

Almost always returns a single cycle (the cell's own outer boundary), but a
cell whose winning feature's territory has a *hole* -- another live cell's
region fully enclosed inside it, sharing no boundary with anything else --
has `edges` containing that hole's own boundary too, as a second, entirely
disjoint cycle. Dangling tips and near-duplicate vertices are cleaned up
first by `prepare_boundary_edges` (see its own docstring); a genuine
branch point past that (rare here, since it usually means a cell boundary
that isn't actually 1D after all) is disambiguated by whichever unused
edge turns *least* from the direction just arrived by, the standard way
to continue straight through a self-crossing of smooth curves rather than
bounce onto the other one.
"""
function cyclic_boundary_walks(cx::CellComplex{N}, edges::Vector{Int}) where {N}
    pt(v) = cx.nodes[v].point
    live, canon = prepare_boundary_edges(cx, edges)
    n = length(live)
    edge_verts = [(canon[cx.nodes[e].subcells[1]], canon[cx.nodes[e].subcells[2]]) for e in live]
    # Canonicalized ids drive connectivity below, but *returned* tuples
    # still report each edge's own real endpoints -- a caller re-deriving
    # `subcells` independently needs to see the same id it already knows.
    # Identical to `edge_verts` in the common no-near-duplicates case.
    edge_verts_raw = [(cx.nodes[e].subcells[1], cx.nodes[e].subcells[2]) for e in live]

    # Only vertices with >2 incident live edges need a disambiguation
    # rule; elsewhere there's at most one unused edge to continue to
    # regardless. `rotation[v]` is `v`'s incident edges in cyclic order,
    # via the best-fit 2-plane (SVD) through their directions -- works
    # unchanged whichever surface/dimension `v` sits on, since edges
    # touching one vertex are coplanar up to numerical noise.
    by_vertex = Dict{Int,Vector{Int}}()
    for i in 1:n
        a, b = edge_verts[i]
        push!(get!(() -> Int[], by_vertex, a), i)
        push!(get!(() -> Int[], by_vertex, b), i)
    end
    rotation = Dict{Int,Vector{Int}}()
    rot_pos = Dict{Int,Dict{Int,Int}}()
    for (v, incident) in by_vertex
        length(incident) <= 2 && continue
        dirs = [pt(edge_verts[i][1] == v ? edge_verts[i][2] : edge_verts[i][1]) - pt(v) for i in incident]
        basis = svd(reduce(hcat, dirs)).U
        u1, u2 = basis[:, 1], basis[:, 2]
        angles = [atan(dot(d, u2), dot(d, u1)) for d in dirs]
        order_by_angle = sortperm(angles)
        rotation[v] = incident[order_by_angle]
        rot_pos[v] = Dict(e => k for (k, e) in enumerate(rotation[v]))
    end

    used = falses(n)
    loops = Vector{Tuple{Int,Int,Int}}[]
    for start in 1:n
        used[start] && continue
        used[start] = true
        order = Int[start]
        entry_vertex = edge_verts[start][1]
        cur_vertex = edge_verts[start][2]
        prev_edge = start
        while cur_vertex != entry_vertex
            found_idx = 0
            if haskey(rotation, cur_vertex)
                rot = rotation[cur_vertex]
                m = length(rot)
                p = rot_pos[cur_vertex][prev_edge]
                for k in 1:m-1
                    cand = rot[mod1(p + k, m)]
                    if !used[cand]
                        found_idx = cand
                        break
                    end
                end
            else
                for i in 1:n
                    used[i] && continue
                    a, b = edge_verts[i]
                    if a == cur_vertex || b == cur_vertex
                        found_idx = i
                        break
                    end
                end
            end
            found_idx == 0 &&
                error("cyclic_boundary_walks: boundary edges don't form a union of simple cycles")
            used[found_idx] = true
            push!(order, found_idx)
            prev_edge = found_idx
            a, b = edge_verts[found_idx]
            cur_vertex = (a == cur_vertex) ? b : a
        end
        out = Tuple{Int,Int,Int}[]
        walk_vertex = entry_vertex
        for idx in order
            a, b = edge_verts[idx]
            ra, rb = edge_verts_raw[idx]
            to = (a == walk_vertex) ? b : a
            from_raw, to_raw = (a == walk_vertex) ? (ra, rb) : (rb, ra)
            push!(out, (live[idx], from_raw, to_raw))
            walk_vertex = to
        end
        push!(loops, out)
    end
    return loops
end

"""
Decomposes boundary edges lying on a *2D surface* into that surface's own
embedded faces via half-edge tracing -- the general form of
`trace_cap_faces` (a thin wrapper fixing `normal_at` to one quadric's own
gradient), reused for an existing `N=3` face's own accumulated boundary,
which can develop the same kind of genuine branch point over several
clips that a brand-new cap develops in one.

Needs a different algorithm from `cyclic_boundary_walks`: a 1D boundary
must decompose into simple cycles (dimension counting), but a 2D
surface's boundary graph has no such guarantee once a branch point exists
-- closer to a polyhedron's edge skeleton than a simple polygon. The
right question is which *faces* the embedding has, not which cycles the
edges form (a vertex's total degree only constrains an edge-disjoint
cycle *cover*, a stricter and sometimes provably-impossible question --
confirmed via a parity check on real failing input: some branch vertices
have odd total degree).

`normal_at(v)` must return a *globally* consistent normal direction --
a curved quadric's own gradient, or a constant vector for a flat surface.
An inconsistently-oriented (e.g. independent per-vertex best-fit) normal
was confirmed to merge everything into one degenerate mega-trace instead
of cleanly separating faces.

Standard combinatorial-map face tracing: every undirected edge is two
half-edges; from each not-yet-used one, continue to the next edge in the
arrival vertex's own rotation order after the one just arrived by, until
back at the start. A branch vertex can make a single trace revisit it
without that being a second face.

Returns one `(edge_id, from_vertex, to_vertex)` triple list per distinct
face -- same format and "outer loop by bounding box, rest holes"
convention as `cyclic_boundary_walks`, a drop-in replacement wherever a
boundary can develop a genuine branch point.
"""
function trace_surface_faces(cx::CellComplex{N}, edges::Vector{Int}, normal_at) where {N}
    live, canon = prepare_boundary_edges(cx, edges)
    isempty(live) && return Vector{Tuple{Int,Int,Int}}[]

    pt(v) = cx.nodes[v].point
    n = length(live)
    edge_verts = [(canon[cx.nodes[e].subcells[1]], canon[cx.nodes[e].subcells[2]]) for e in live]
    edge_verts_raw = [(cx.nodes[e].subcells[1], cx.nodes[e].subcells[2]) for e in live]
    other_end(i, v) = edge_verts[i][1] == v ? edge_verts[i][2] : edge_verts[i][1]

    by_vertex = Dict{Int,Vector{Int}}()
    for i in 1:n
        a, b = edge_verts[i]
        push!(get!(() -> Int[], by_vertex, a), i)
        push!(get!(() -> Int[], by_vertex, b), i)
    end

    rotation = Dict{Int,Vector{Int}}()
    rot_pos = Dict{Int,Dict{Int,Int}}()
    for (v, incident) in by_vertex
        if length(incident) <= 2
            rotation[v] = incident
        else
            g = normal_at(v)
            ng = norm(g)
            ng < 1e-9 && error("trace_surface_faces: normal vanishes at branch vertex $v -- can't orient the local rotation there")
            g = g / ng
            d0 = pt(other_end(incident[1], v)) - pt(v)
            u1 = normalize(d0 - dot(d0, g) * g)
            u2 = cross(g, u1)
            dirs = [pt(other_end(i, v)) - pt(v) for i in incident]
            angles = [atan(dot(d, u2), dot(d, u1)) for d in dirs]
            rotation[v] = incident[sortperm(angles)]
        end
        rot_pos[v] = Dict(e => k for (k, e) in enumerate(rotation[v]))
    end

    used_fwd = falses(n)
    used_bwd = falses(n)
    faces = Vector{Vector{Tuple{Int,Int,Int}}}()
    for start_i in 1:n, start_dir in (:fwd, :bwd)
        already = start_dir == :fwd ? used_fwd[start_i] : used_bwd[start_i]
        already && continue
        face = Tuple{Int,Int,Int}[]
        cur_i, cur_dir = start_i, start_dir
        steps = 0
        while true
            steps += 1
            steps > 4n + 5 && error("trace_surface_faces: runaway trace after $(steps) steps -- rotation system isn't closing correctly")
            a, b = edge_verts[cur_i]
            ra, rb = edge_verts_raw[cur_i]
            from_c, to_c = cur_dir == :fwd ? (a, b) : (b, a)
            from_raw, to_raw = cur_dir == :fwd ? (ra, rb) : (rb, ra)
            if cur_dir == :fwd
                used_fwd[cur_i] = true
            else
                used_bwd[cur_i] = true
            end
            push!(face, (live[cur_i], from_raw, to_raw))
            to = to_c
            rot = rotation[to]
            next_i = rot[mod1(rot_pos[to][cur_i] + 1, length(rot))]
            next_dir = edge_verts[next_i][1] == to ? :fwd : :bwd
            (next_i, next_dir) == (start_i, start_dir) && break
            cur_i, cur_dir = next_i, next_dir
        end
        push!(faces, face)
    end

    seen = Set{Set{Int}}()
    kept = Vector{Vector{Tuple{Int,Int,Int}}}()
    for f in faces
        es = Set(e for (e, _, _) in f)
        es in seen && continue
        push!(seen, es)
        push!(kept, f)
    end
    return kept
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
Restricts `Quadric{N,Float64}` `q` to the 2D plane `{origin + u*e1 + v*e2}`
(`e1`,`e2` orthonormal), returning the `Quadric{2,Float64}` such that
`evaluate(restrict_to_plane(q,...), (u,v)) == evaluate(q, origin+u*e1+v*e2)`
exactly, for every `(u,v)`.

This is the one new primitive `N=3` curved bisectors need that `N=2` never
did: a 2D "edge" is already, by itself, a 1-dimensional subset of the
*same* 2D ambient space `Quadric{2}` describes, so a single quadric
equation fully pins down its shape. A 3D *face* is a 2-dimensional subset
of 3D space -- one `Quadric{3}` equation alone describes a *surface*, not
the face's own bounded 2D shape -- so finding where another bisector
crosses a (flat) face means first re-expressing that bisector *within*
the face's own 2D plane, where the existing, already-proven 2D
crossing-finding machinery (`edge_crossings`, `edge_strict_interior_crossings`,
etc.) applies unchanged.

Derived by substituting `x = origin + u*e1 + v*e2` into
`evaluate(q,x) = xᵀMx + 2bᵀx + c` and collecting terms in `u`,`v`: the
quadratic part comes from `e_i·(M·e_j)`, the linear part from
`e_i·(M·origin + b)`, and the constant term is exactly `evaluate(q, origin)`
(since setting `u=v=0` must reproduce `q` at `origin` itself).
"""
function restrict_to_plane(q::Quadric{N,Float64}, origin::Pt{N,Float64}, e1::Pt{N,Float64}, e2::Pt{N,Float64}) where {N}
    Mo_plus_b = q.M * origin + q.b
    M2 = SMatrix{2,2,Float64}(dot(e1, q.M * e1), dot(e1, q.M * e2), dot(e1, q.M * e2), dot(e2, q.M * e2))
    b2 = SVector(dot(e1, Mo_plus_b), dot(e2, Mo_plus_b))
    c2 = evaluate(q, origin)
    return Quadric{2,Float64}(M2, b2, c2)
end

"""
`point_in_edge_loop`'s own 3D-face-local counterpart, for `clip_flat_face_3d!`/
`clip_curved_face_3d!`'s own hole-placement ("which resulting piece does
this hole belong in?", mirroring `clip_top_cell_2d!`'s identical question
one dimension down). `pt_local` and every edge's own endpoints are
projected into `(origin,e1,e2)`'s own local 2D frame first via
`restrict_to_plane`'s same substitution, and a curved edge's own `curve`
(if any) is restricted into that frame too rather than assumed already
2D -- otherwise byte-for-byte `point_in_edge_loop`'s own even-odd
ray-crossing test.

For a genuinely flat face, `(origin,e1,e2)` is *the* exact plane every
point on the face lies in, so this is exact. For a curved face there's no
single exact flat frame (see `curved_face_local_frame`'s own docstring)
-- a locally-approximate one is used there instead, matching this
codebase's existing acceptance of a best-effort (not exact) answer for
hole placement specifically (see the fallback in `clip_flat_face_3d!`'s
own hole loop, inherited unchanged from `clip_top_cell_2d!`).
"""
function point_in_edge_loop_3d(cx::CellComplex{3}, edges::Vector{Int}, origin::Pt{3,Float64}, e1::Pt{3,Float64}, e2::Pt{3,Float64}, pt_local::Pt{2,Float64})
    to_local(p) = SVector(dot(p - origin, e1), dot(p - origin, e2))
    y0 = pt_local[2]
    crossings = 0
    for e in edges
        en = cx.nodes[e]
        v1, v2 = en.subcells
        p1, p2 = to_local(cx.nodes[v1].point), to_local(cx.nodes[v2].point)
        local_curve = en.curve === nothing ? nothing : restrict_to_plane(en.curve, origin, e1, e2)
        if local_curve === nothing
            (p1[2] > y0) == (p2[2] > y0) && continue
            x_at_y = p1[1] + (p2[1] - p1[1]) * (y0 - p1[2]) / (p2[2] - p1[2])
            x_at_y > pt_local[1] && (crossings += 1)
        else
            for x in curve_crossings_at_y(local_curve, y0)
                x > pt_local[1] || continue
                on_arc_between(local_curve, p1, p2, SVector(x, y0)) && (crossings += 1)
            end
        end
    end
    return isodd(crossings)
end

"""
`find_outer_loop`'s own 3D-face-local counterpart: bounding-box area
computed in `(origin,e1,e2)`'s own local 2D frame (`point_in_edge_loop_3d`'s
same convention) instead of assuming the stored points are already 2D.
"""
function find_outer_loop_3d(cx::CellComplex{3}, loops::Vector{Vector{Tuple{Int,Int,Int}}}, origin::Pt{3,Float64}, e1::Pt{3,Float64}, e2::Pt{3,Float64})
    isempty(loops) && return nothing
    to_local(p) = SVector(dot(p - origin, e1), dot(p - origin, e2))
    function loop_bbox_area(w)
        pts = [to_local(cx.nodes[v].point) for (_, v, _) in w]
        lo = reduce((a, b) -> min.(a, b), pts)
        hi = reduce((a, b) -> max.(a, b), pts)
        d = hi - lo
        return d[1] * d[2]
    end
    return argmax([loop_bbox_area(w) for w in loops])
end

"""
`loop_interior_point`'s own 3D-face-local counterpart, returning a point
in `(origin,e1,e2)`'s own local 2D coordinates (`point_in_edge_loop_3d`'s
same convention) rather than a 3D one -- otherwise byte-for-byte the same
"topmost vertex, cast a ray just below it" algorithm.
"""
function loop_interior_point_3d(cx::CellComplex{3}, loop::Vector{Tuple{Int,Int,Int}}, origin::Pt{3,Float64}, e1::Pt{3,Float64}, e2::Pt{3,Float64})
    to_local(p) = SVector(dot(p - origin, e1), dot(p - origin, e2))
    verts = unique(v for (_, v, _) in loop)
    pts = [to_local(cx.nodes[v].point) for v in verts]
    topv = pts[argmax([(p[2], p[1]) for p in pts])]
    y0 = topv[2] - 1e-6 * max(1.0, maximum(p -> abs(p[2]), pts))

    xs = Float64[]
    for (e, _, _) in loop
        en = cx.nodes[e]
        v1, v2 = en.subcells
        p1, p2 = to_local(cx.nodes[v1].point), to_local(cx.nodes[v2].point)
        local_curve = en.curve === nothing ? nothing : restrict_to_plane(en.curve, origin, e1, e2)
        if local_curve === nothing
            (p1[2] > y0) == (p2[2] > y0) && continue
            push!(xs, p1[1] + (p2[1] - p1[1]) * (y0 - p1[2]) / (p2[2] - p1[2]))
        else
            for x in curve_crossings_at_y(local_curve, y0)
                on_arc_between(local_curve, p1, p2, SVector(x, y0)) && push!(xs, x)
            end
        end
    end
    sort!(xs)
    length(xs) < 2 && return sum(pts) / length(pts)
    i = clamp(searchsortedlast(xs, topv[1]), 1, length(xs) - 1)
    return SVector((xs[i] + xs[i+1]) / 2, y0)
end

"""
An approximate local 2D frame for a *curved* face `Q1`, for hole-placement
purposes only (`clip_curved_face_3d!`'s own use of
`point_in_edge_loop_3d`/`loop_interior_point_3d`/`find_outer_loop_3d`) --
unlike `flat_face_frame_cached!`'s exact flat frame (a genuine flat face
has one true plane every point lies in), a curved face has no single exact
flat chart covering a whole bounded region in general. Uses `Q1`'s own
tangent plane at `walk`'s own vertex centroid (gradient `M*x+b` as the
normal, Gram-Schmidt for the tangent basis -- the same well-defined,
globally-meaningful-at-that-one-point construction `trace_cap_faces` uses
per-vertex) as a reasonable local flattening, good enough for "which
resulting piece does this hole fall into" the same way this codebase
already accepts an approximate, fallback-guarded answer there (see
`clip_flat_face_3d!`'s own hole loop) -- not used for anything requiring
exact geometry.
"""
function curved_face_local_frame(Q1::Quadric{3,Float64}, cx::CellComplex{3}, walk::Vector{Tuple{Int,Int,Int}})
    verts = unique(v for (_, v, _) in walk)
    pts = [cx.nodes[v].point for v in verts]
    origin = sum(pts) / length(pts)
    g = Q1.M * origin + Q1.b
    ng = norm(g)
    ng < 1e-9 && error("curved_face_local_frame: Q1's own gradient vanishes at this face's own vertex centroid -- can't build a local frame there (a genuine saddle point coinciding exactly with the centroid, not handled)")
    g = g / ng
    d0 = pts[1] - origin
    d0 = d0 - dot(d0, g) * g
    norm(d0) < 1e-9 && (d0 = pts[2] - origin - dot(pts[2] - origin, g) * g)
    e1 = normalize(d0)
    e2 = cross(g, e1)
    return origin, e1, e2
end

"""
An orthonormal `(origin, e1, e2)` frame for the flat plane spanned by
`verts` (at least 3 non-collinear points, as a flat face's own boundary
vertices always are) -- `origin` is `verts[1]`, `e1` the unit direction to
the first other vertex not coincident with it, `e2` the Gram-Schmidt
remainder of the first vertex not collinear with `e1`. Errors if every
vertex is collinear (or coincident) with the first -- a degenerate
"face" with no genuine 2D extent, which shouldn't reach here.
"""
function flat_face_frame(verts::Vector{Pt{N,Float64}}) where {N}
    origin = verts[1]
    e1 = nothing
    for v in verts[2:end]
        d = v - origin
        if norm(d) > 1e-9
            e1 = d / norm(d)
            break
        end
    end
    e1 === nothing && error("flat_face_frame: every vertex coincides with the first -- no genuine face here")
    e2 = nothing
    for v in verts[2:end]
        d = v - origin
        perp = d - dot(d, e1) * e1
        if norm(perp) > 1e-9 * max(1.0, norm(d))
            e2 = perp / norm(perp)
            break
        end
    end
    e2 === nothing && error("flat_face_frame: every vertex is collinear with the first two -- no genuine 2D face here")
    return origin, e1, e2
end

"""
Gets a flat `N=3` face's own local `(origin, e1, e2)` 2D frame, from
`cx.face_frames` if already derived, or from its boundary vertices
(`flat_face_frame`) otherwise, caching either way.

Boundary vertices alone can fail to pin down the plane -- a two-vertex
"bigon" piece has no third non-collinear vertex, even though its plane is
well-defined (same as its parent face). Reusing an inherited frame
handles that. Shared by `clip_flat_face_3d!` and `sweep_topology_check`.

Reads `fnode.subcells`' own raw edge endpoints directly, not via any
loop/tracing decomposition -- a flat plane's extent is planar regardless
of boundary-graph structure, so any vertex works, and this also breaks a
circular dependency (`trace_surface_faces` needs this frame's normal
before it can build a flat face's own rotation system).
"""
function flat_face_frame_cached!(cx::CellComplex{3}, face_id::Int)
    haskey(cx.face_frames, face_id) && return cx.face_frames[face_id]
    fnode = cx.nodes[face_id]
    isempty(fnode.subcells) && error("flat_face_frame_cached!: face $face_id has no boundary edges at all -- a face with no real enclosed area, which shouldn't happen")
    verts = Pt{3,Float64}[]
    for e in fnode.subcells
        v1, v2 = cx.nodes[e].subcells
        push!(verts, cx.nodes[v1].point, cx.nodes[v2].point)
    end
    frame = flat_face_frame(verts)
    cx.face_frames[face_id] = frame
    return frame
end

"""
`trace_surface_faces` specialized to an `N=3` face's own boundary
(`fnode.subcells`) -- the drop-in replacement for
`cyclic_boundary_walks(cx, fnode.subcells)` everywhere a face's boundary
is walked (`clip_flat_face_3d!`, `clip_curved_face_3d!`), since that
boundary can develop a genuine branch point after enough independent
clips, exactly like a cap can (see `trace_surface_faces`'s own docstring).

Picks the right globally-consistent normal for `normal_at` depending on
whether the face is flat or already curved: `fnode.curve`'s own gradient
for a curved face (identical reasoning to `trace_cap_faces`'s own use of
`quad`'s gradient -- every boundary point lies exactly on `fnode.curve`'s
zero set by construction), or the face's own cached plane normal
(`flat_face_frame_cached!`, a single constant vector) for a flat one.
"""
function face_boundary_faces(cx::CellComplex{3}, face_id::Int)
    fnode = cx.nodes[face_id]
    if fnode.curve !== nothing
        curve = fnode.curve
        normal_at = v -> curve.M * cx.nodes[v].point + curve.b
        return trace_surface_faces(cx, fnode.subcells, normal_at)
    else
        origin, e1, e2 = flat_face_frame_cached!(cx, face_id)
        n = cross(e1, e2)
        normal_at = v -> n
        return trace_surface_faces(cx, fnode.subcells, normal_at)
    end
end

"""
Finds a live, flat (`curve === nothing`) `dim=2` face that references
`edge_id` as an immediate subcell -- used to locate a curved edge's own
embedding: a curved `dim=1` edge's own `curve` field alone doesn't pin
down its shape at `N=3` ("curve lives on facets", i.e. dimension `N-1` --
a dim=1 edge only gets that treatment at `N=2`), so it also needs
*some* flat plane it's confined to, and any curved edge in this codebase
(a trace edge left by capping a cut) always has exactly one -- the
original flat face it was cut from. Used by `clip_curved_face_3d!` (a
curved face's own boundary edges, to locate where a new *flat* clipping
plane crosses them) and by the read-only `sweep_topology_check`. Errors
if none exists (an edge bounded only by curved faces on every side --
e.g. two different segments' own caps meeting -- is beyond what either of
those callers can handle yet).
"""
function find_flat_neighbor_face(cx::CellComplex{3}, edge_id::Int)
    for f in get(cx.referenced_by, edge_id, Int[])
        haskey(cx.superseded_by, f) && continue
        fnode = cx.nodes[f]
        fnode.dim == 2 || continue
        fnode.curve === nothing && return f
    end
    error("find_flat_neighbor_face: edge $edge_id has no live flat neighboring face -- can't locate a curved edge without one")
end

"""
Every point (arc order, `p1` -> `p2`, inclusive tolerance) where `quad`
crosses curved edge `edge_id`'s own bounded arc -- tries the ordinary
flat-neighbor route first (`find_flat_neighbor_face`, exact, the same one
every other curved edge in this codebase uses, reusing 2D
`edge_curve_crossings` restricted to that flat neighbor's own frame --
already general to up to 4 crossings via Bezout, not just one), falling
back to `ruled_trace_all_crossings` (`ruled_quadric.jl`, numeric) only
when that fails *and* the edge is confined to two curved surfaces at once
(`enode.curve2 !== nothing` -- see `CellNode`'s own docstring). The two
routes are mutually exclusive by construction: a flat-neighbor edge never
has `curve2` set, and vice versa.

Generalizes the old single-crossing `curved_edge_crossing_point`: any
call site that still wants exactly one crossing asserts
`length(...) == 1` itself now, mirroring how `edge_curve_crossings` vs.
`clip_top_cell_2d!`'s own single-crossing call sites already relate.
"""
function curved_edge_all_crossings(cx::CellComplex{3}, edge_id::Int, quad::Quadric{3,Float64})
    enode = cx.nodes[edge_id]
    p1id, p2id = enode.subcells
    p1, p2 = cx.nodes[p1id].point, cx.nodes[p2id].point
    flat_face = try
        find_flat_neighbor_face(cx, edge_id)
    catch e
        enode.curve2 === nothing && rethrow(e)
        frame = ruled_frame(enode.curve2)
        return ruled_trace_all_crossings(enode.curve, quad, frame, p1, p2)
    end
    origin, e1, e2 = flat_face_frame_cached!(cx, flat_face)
    quad2 = restrict_to_plane(quad, origin, e1, e2)
    edge_curve2 = enode.curve === nothing ? nothing : restrict_to_plane(enode.curve, origin, e1, e2)
    to_local(p) = SVector(dot(p - origin, e1), dot(p - origin, e2))
    local_crossings = edge_curve_crossings(quad2, to_local(p1), to_local(p2), edge_curve2)
    return [origin + u * e1 + v * e2 for (u, v) in local_crossings]
end

"""
Every point (same arc-order, inclusive convention as
`curved_edge_all_crossings`) where `quad` crosses the *strict interior* of
curved edge `edge_id`'s own bounded arc -- used when both endpoints agree
on side, the same "is this agreement genuine, or does the bisector dip in
and back out along the way" question `edge_strict_interior_crossings`
answers for a flat face's own local-plane edges. Flat-neighbor route
reuses `edge_strict_interior_crossings` directly (strict by construction);
the ruled-fallback route's own bisection scan is already effectively
strict (a bracket only registers on a genuine sign change between sampled
interior points, never merely landing on an endpoint), so no separate
tolerance handling is needed there.
"""
function curved_edge_strict_interior_crossings(cx::CellComplex{3}, edge_id::Int, quad::Quadric{3,Float64})
    enode = cx.nodes[edge_id]
    p1id, p2id = enode.subcells
    p1, p2 = cx.nodes[p1id].point, cx.nodes[p2id].point
    flat_face = try
        find_flat_neighbor_face(cx, edge_id)
    catch e
        enode.curve2 === nothing && rethrow(e)
        frame = ruled_frame(enode.curve2)
        return ruled_trace_all_crossings(enode.curve, quad, frame, p1, p2)
    end
    origin, e1, e2 = flat_face_frame_cached!(cx, flat_face)
    quad2 = restrict_to_plane(quad, origin, e1, e2)
    edge_curve2 = enode.curve === nothing ? nothing : restrict_to_plane(enode.curve, origin, e1, e2)
    to_local(p) = SVector(dot(p - origin, e1), dot(p - origin, e2))
    local_crossings = edge_strict_interior_crossings(quad2, to_local(p1), to_local(p2), edge_curve2)
    return [origin + u * e1 + v * e2 for (u, v) in local_crossings]
end

"""
Whether any face (dim=2) of 3D cell `cell_id`, or any of *those* faces' own
boundary edges (dim=1), is already curved -- the two dimensions `curve`
can ever be set at for `N=3` (never dim=0 vertices, never the dim=3 cell
itself). Used to decide whether a clip needs `clip_top_cell_3d!` at all:
a cell whose own faces are all flat *and* whose faces' own edges are all
straight is exactly the "no curvature anywhere nearby" case the original,
already-proven-correct general-purpose dim-by-dim loop already handles
correctly (confirmed via extensive stress testing before any curved
bisector existed in this codebase at all) -- routing it through the newer,
much less battle-tested 3D machinery instead would be pure added risk for
no benefit. A flat face can still have an already-curved *edge* (shared
with a sibling top cell whose own independent clip curved it first, via
the same "a shared boundary is already patched by the time a later cell
is examined" mechanism `insert_features!`'s docstring describes for `N=2`)
-- checked explicitly here rather than assumed away, since that's exactly
the scenario `clip_top_cell_3d!` exists to handle correctly.
"""
function cell_has_any_curve_3d(cx::CellComplex{3}, cell_id::Int)
    for f in cx.nodes[cell_id].subcells
        fnode = cx.nodes[f]
        fnode.curve === nothing || return true
        for e in fnode.subcells
            cx.nodes[e].curve === nothing || return true
        end
    end
    return false
end

"""
N=2 replacement for `clip_by_hyperplane!`'s general-purpose per-subcell
dictionary assembly, handling `start_id`'s own edges and its top-level
split together in one pass, to support multi-crossing (see
`clip_by_hyperplane!`'s own docstring).

Walks `start_id`'s boundary in cyclic order (`cyclic_boundary_walks`);
splits each edge at its interior crossings with `quad` (0-2, the true
ceiling since crossing-finding solves at most a quadratic); flattens into
one cyclic sequence of (side, piece) tuples; cuts into maximal same-side
runs. Each run becomes its own new top-level cell, closed by its own
fresh arc of `quad` (not shared with any other run). No crossings reduces
to one run spanning the whole boundary, leaving `start_id` untouched.

More than two runs (e.g. a curve dipping in and out of one edge, carving
an isolated "bite") is what the label/bbox infrastructure exists to
support: several live cells sharing one label, distinguished by disjoint
bounding boxes -- not a fully closed case if a future shape violates that
disjointness.

A hole (`orig_edges` decomposing into more than one loop) is carried
through unchanged, attached to whichever resulting run contains it
(`point_in_edge_loop`) -- if `quad` actually cuts into the hole, that's
independently discovered by the enclosed cell's own clip in this same
insertion pass, so reprocessing it here would be redundant.

Returns `(results, extra_a, extra_b, extra_cut, pending)`: `results` is
`Vector{Tuple{Bool,Int}}` per resulting top-level cell; `extra_a`/
`extra_b`/`extra_cut` are ids needing the corresponding label (not a
one-id-per-original dict, since one edge can expand into up to three
pieces); `pending` is this call's own supersessions.
"""
function clip_top_cell_2d!(cx::CellComplex{2}, start_id::Int, quad::Quadric{2,Float64}, a_piece::Dict{Int,Int}, b_piece::Dict{Int,Int}, idxA::Set{VertexIdx}, idxB::Set{VertexIdx})
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
                # If either endpoint is on record (`CellNode.exact_ties`,
                # populated by this same clip's own vertex-level loop just
                # before this function was ever called) as a *genuine*
                # exact tie between `idxA`/`idxB` -- not just numerically
                # close, actually `exact_sign == 0` -- its own side is
                # `symbolic_tiebreak`'s canonical, deterministic answer,
                # strictly more trustworthy than a raw magnitude guess.
                # Empirically (see `project_pairwise_vs_multiway_ties.md`),
                # this rarely applies -- most disagreements reaching here
                # are between two genuinely, unambiguously opposite-signed
                # points, not a tie -- but when it does apply it's a real
                # fact, not a heuristic.
                t1, t2 = exact_ties(cx, p1id), exact_ties(cx, p2id)
                if t1 !== nothing && (idxA in t1 || idxB in t1)
                    # s1 already is p1id's own canonical side -- keep it.
                elseif t2 !== nothing && (idxA in t2 || idxB in t2)
                    s1 = s2
                else
                    # The endpoint with the larger |value| is the one whose
                    # side is trustworthy (further from the zero set, so its
                    # sign isn't a coin flip); use that one for the whole
                    # (unsplit) edge instead of erroring over the other's own
                    # arbitrary tie-break disagreeing with it.
                    v1, v2 = evaluate(quad, p1), evaluate(quad, p2)
                    s1 = (abs(v1) >= abs(v2) ? v1 : v2) < 0
                end
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
Clips one *flat* 3D face `face_id` by ambient bisector `quad`, restricted
to this face's own plane -- the 3D analogue of one edge's own processing
inside `clip_top_cell_2d!`, one dimension up.

v1 scope: the face must itself be flat (see `clip_curved_face_3d!` for a
curved one). Multi-crossing within one face is supported (mirrors
`clip_top_cell_2d!`'s own "bite" handling). Holes are supported too: the
largest boundary loop by bbox area is the outer boundary, every other
loop a hole attached to whichever resulting piece contains it, with the
same best-effort fallback if a hole doesn't land cleanly in any one piece.

Returns `(results, trace_edges, extra_a, extra_b, pending)`, mirroring
`clip_top_cell_2d!`'s own contract: `results` one entry per resulting
same-side run; `trace_edges` every new curved edge this cut produced (the
caller collects these into the new capping face); `extra_a`/`extra_b`
every new piece needing that side's label; `pending` supersessions.

`edge_cache` (keyed by original edge id) is shared across every face of
the same enclosing 3D cell: a box edge is generically shared by two
faces, and independently re-splitting it would leave two `cx` vertices a
ULP apart at the same cut location instead of one -- the second face to
reach an edge reuses the first's cut vertices and pieces instead
(confirmed as a real bug before this cache existed: a fresh box's four
trace edges never shared a vertex pairwise, so `cyclic_boundary_walks`
found zero closed loops instead of one).
"""
function clip_flat_face_3d!(cx::CellComplex{3}, face_id::Int, quad::Quadric{3,Float64}, a_piece::Dict{Int,Int}, b_piece::Dict{Int,Int}, edge_cache::Dict{Int,Vector{Tuple{Bool,Int,Int,Int}}}, idxA::Set{VertexIdx}, idxB::Set{VertexIdx})
    fnode = cx.nodes[face_id]
    fnode.curve === nothing || error("clip_flat_face_3d!: face $face_id is already curved -- splitting an already-curved face isn't supported yet (v1 scope)")

    floops = face_boundary_faces(cx, face_id)
    origin, e1, e2 = flat_face_frame_cached!(cx, face_id)
    if length(floops) == 1
        walk = floops[1]
        hole_walks = Vector{Tuple{Int,Int,Int}}[]
    else
        outer_idx = find_outer_loop_3d(cx, floops, origin, e1, e2)
        outer_idx === nothing && error("clip_flat_face_3d!: face $face_id's own boundary edges form no genuine closed loop at all (every one is part of a dangling spike) -- a face with no real enclosed area, which shouldn't happen")
        walk = floops[outer_idx]
        hole_walks = [floops[i] for i in eachindex(floops) if i != outer_idx]
    end

    quad2 = restrict_to_plane(quad, origin, e1, e2)
    to_local(p) = SVector(dot(p - origin, e1), dot(p - origin, e2))

    extra_a = Int[]
    extra_b = Int[]
    pending = Tuple{Int,Vector{Int}}[]
    trace_verts = Int[]
    flat = Tuple{Bool,Int,Int,Int}[]

    # The side of whichever vertex the walk has most recently arrived at
    # (`nothing` only before the very first edge) -- see the long comment
    # at its first use below for why this exists: independently
    # re-deriving every edge's own side from `a_piece` is what the
    # crossing-alternation logic is *supposed* to make unnecessary past
    # the first edge, but the code used to do it anyway, which is exactly
    # what let a single near-tied vertex (an arbitrary, if locally
    # defensible, tie-break label) desync two edges that both touch it.
    cur_side = nothing

    for (e, fv, tv) in walk
        enode = cx.nodes[e]
        p1id, p2id = enode.subcells[1], enode.subcells[2]

        if haskey(edge_cache, e)
            pieces = edge_cache[e]
        else
            # A boundary edge of a *flat* face is not always straight
            # itself: this same face may have been reached, whole, as the
            # *other* side of a shared boundary a sibling top cell's own
            # independent clip already split (`supersede!` patches every
            # parent's own subcells in place, this face's included,
            # whenever that happens -- see `insert_features!`'s own
            # docstring on why "a shared boundary is already patched by
            # the time a later cell is examined" is exactly the existing,
            # relied-upon behavior here too). Its own curve, if any, is
            # `enode.curve` (a `Quadric{3}`, the *other* bisector that
            # created it) -- restricted to *this* face's own local frame
            # (the same reduction `quad` itself already goes through),
            # reusing the already-proven 2D curved-edge machinery
            # (`edge_curve_crossings`) rather than needing a separate
            # 3D-specific path.
            edge_curve2 = enode.curve === nothing ? nothing : restrict_to_plane(enode.curve, origin, e1, e2)
            p1, p2 = cx.nodes[p1id].point, cx.nodes[p2id].point
            lp1, lp2 = to_local(p1), to_local(p2)
            s1, s2 = haskey(a_piece, p1id), haskey(a_piece, p2id)

            # Every interior crossing of `quad2` along this edge flips the
            # side, so `s1` alone (alternating once per crossing)
            # determines every resulting piece's side, for any number of
            # crossings -- exactly `clip_top_cell_2d!`'s own multi-
            # crossing logic (see its docstring for why strict crossings,
            # not `s1==s2`, drive this, and for the tie-broken-endpoint
            # fallback below), ported one dimension up unchanged.
            xs = edge_strict_interior_crossings(quad2, lp1, lp2, edge_curve2)
            if isempty(xs) && s1 != s2
                crossings = edge_curve_crossings(quad2, lp1, lp2, edge_curve2)
                if length(crossings) == 1
                    xs = crossings
                elseif cur_side !== nothing
                    # `cur_side` is the walk's own already-established,
                    # consistent side (ultimately traced back to `a_piece`,
                    # itself computed by exact_sign/symbolic_tiebreak -- see
                    # the vertex-level loop in `clip_by_hyperplane!`), so it
                    # is strictly more trustworthy than re-deriving a fresh
                    # answer from this face's own local, floating-point
                    # restriction of `quad`. Confirmed empirically (see
                    # `project_pairwise_vs_multiway_ties.md`): every case
                    # this branch has actually hit had `exact_sign` return a
                    # clean, unambiguous, *nonzero* sign for both endpoints
                    # -- there was no genuine tie to break, just a crossing
                    # the finder above failed to locate. Trusting the
                    # already-consistent walk state sidesteps that failure
                    # instead of compounding it with a second, independent
                    # floating-point guess.
                    s1 = cur_side
                else
                    # First edge of the walk (nothing propagated yet): the
                    # same exact-ties check as `clip_top_cell_2d!` -- see
                    # its own comment on this identical branch.
                    t1, t2 = exact_ties(cx, p1id), exact_ties(cx, p2id)
                    if t1 !== nothing && (idxA in t1 || idxB in t1)
                        # keep s1 as-is
                    elseif t2 !== nothing && (idxA in t2 || idxB in t2)
                        s1 = s2
                    else
                        v1, v2 = evaluate(quad2, lp1), evaluate(quad2, lp2)
                        s1 = (abs(v1) >= abs(v2) ? v1 : v2) < 0
                    end
                end
            end

            if isempty(xs)
                pieces = [(s1, e, p1id, p2id)]
            else
                length(xs) <= 4 || error("clip_flat_face_3d!: edge $e has $(length(xs)) interior crossings -- two genuine quadrics can cross at most 4 times (Bezout), so this many indicates a bug in the crossing finder, not a valid configuration")
                cuts = [add_cell!(cx, 0, Label(), Int[], origin + u * e1 + v * e2) for (u, v) in xs]
                verts = vcat(p1id, cuts, p2id)
                piece_ids = Int[]
                pieces = Tuple{Bool,Int,Int,Int}[]
                for i in 1:length(verts)-1
                    side = isodd(i) ? s1 : !s1
                    # The new pieces stay on `enode`'s own curve (if any),
                    # unchanged -- splitting an edge doesn't change *which*
                    # bisector it lies on, only where it's now bounded.
                    pid = add_cell!(cx, 1, Label(), [verts[i], verts[i+1]]; curve=enode.curve)
                    push!(pieces, (side, pid, verts[i], verts[i+1]))
                    push!(piece_ids, pid)
                end
                push!(pending, (e, piece_ids))
            end
            edge_cache[e] = pieces
        end

        for (is_a, pid, _, _) in pieces
            push!(is_a ? extra_a : extra_b, pid)
        end
        append!(trace_verts, [p[4] for p in pieces[1:end-1]])
        walk_pieces = fv == p1id ? pieces : [(is_a, pid, to, from) for (is_a, pid, from, to) in reverse(pieces)]
        append!(flat, walk_pieces)
        # A cached edge's own recorded side (from whichever sibling face
        # reached it first) always wins over this walk's own propagation
        # when the two would otherwise disagree -- it's a decision
        # already baked into real `cx` nodes (`extra_a`/`extra_b` already
        # pushed, `pid`s already created), not something this walk can
        # silently reinterpret. Propagation only ever *fills in* what a
        # fresh computation would otherwise have asked `a_piece` for,
        # never overrides an already-committed one.
        cur_side = walk_pieces[end][1]
    end

    if isempty(trace_verts)
        side = flat[1][1]
        all(p -> p[1] == side, flat) ||
            error("clip_flat_face_3d!: face $face_id's own edges disagree on side despite finding no crossing anywhere -- a genuine inconsistency, not something this function knows how to resolve")
        return [(side, face_id)], Int[], extra_a, extra_b, pending
    end

    # Cut the face's own cyclic edge-piece sequence into its same-side
    # runs -- the general (any number of cuts) version of the old
    # "exactly one cut" logic, byte-for-byte `clip_top_cell_2d!`'s own
    # runs-forming step (see its docstring), one dimension down.
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

    # Unlike `clip_top_cell_2d!` (where each run's own closing chord is
    # private to that one resulting *top* cell, with no further structure
    # ever needing the two sides of a cut to reference the very same edge
    # object), a *face*'s own trace edge(s) get collected upward by
    # `clip_top_cell_3d!` into one shared capping surface -- so the single
    # cut (`length(runs) == 2`) case needs its *one* trace edge to be the
    # very same `cx` edge on both sides, not two coincident-but-distinct
    # ones: with only two cut vertices total, both runs' own "direct
    # shortcut between my own first and last vertex" happen to be the
    # identical undirected chord, and creating it twice would leave two
    # parallel, unshared edges between the same two points -- exactly the
    # kind of duplicate `cyclic_boundary_walks` can't stitch into one
    # closed loop later (confirmed: this was a real bug during
    # development, breaking every single-cut case, not just a hypothetical
    # one). For more than two cut vertices (multiple disjoint cuts on this
    # one face), each run's own chord connects a *different* pair of cut
    # vertices, so no such coincidence occurs and each genuinely needs its
    # own edge -- mirroring `clip_top_cell_2d!`'s own per-run chords
    # exactly, with whether the resulting collection of trace edges
    # actually stitches into one connected cap left to
    # `clip_top_cell_3d!`'s own existing check.
    trace_edges = Int[]
    run_trace_edge = Vector{Int}(undef, length(runs))
    if length(runs) == 2
        first_v, last_v = runs[1][1][3], runs[1][end][4]
        shared_edge = add_cell!(cx, 1, Label(), [last_v, first_v]; curve=(is_curved(quad) ? quad : nothing))
        push!(trace_edges, shared_edge)
        run_trace_edge .= shared_edge
    else
        for (i, run) in enumerate(runs)
            first_v, last_v = run[1][3], run[end][4]
            trace_edge = add_cell!(cx, 1, Label(), [last_v, first_v]; curve=(is_curved(quad) ? quad : nothing))
            push!(trace_edges, trace_edge)
            run_trace_edge[i] = trace_edge
        end
    end

    # Each hole belongs entirely to whichever resulting run's own (now
    # closed, trace edge included) boundary contains it -- same question,
    # same "sample an interior point, test membership, fall back to
    # nearest-by-bbox-center if no run claims it cleanly (e.g. a hole
    # independently cut by this same bisector)" answer as
    # `clip_top_cell_2d!`'s own hole-placement, one dimension down.
    run_extra_edges = [Int[] for _ in runs]
    for hw in hole_walks
        sample_pt = loop_interior_point_3d(cx, hw, origin, e1, e2)
        ri = findfirst(i -> point_in_edge_loop_3d(cx, vcat([p[2] for p in runs[i]], run_trace_edge[i]), origin, e1, e2, sample_pt), eachindex(runs))
        if ri === nothing
            hole_pts = [cx.nodes[v].point for (_, v, _) in hw]
            hole_center = sum(hole_pts) / length(hole_pts)
            run_center(run) = sum(vcat([cx.nodes[p[3]].point for p in run], [cx.nodes[p[4]].point for p in run])) / (2 * length(run))
            ri = argmin([norm(run_center(run) - hole_center) for run in runs])
            @warn "clip_flat_face_3d!: a hole in face $face_id's own territory didn't land cleanly inside any of the $(length(runs)) resulting piece(s) -- falling back to the nearest piece by bounding-box center"
        end
        append!(run_extra_edges[ri], [e for (e, _, _) in hw])
    end

    results = Tuple{Bool,Int}[]
    for (i, run) in enumerate(runs)
        cell_id = add_cell!(cx, 2, Label(), vcat([p[2] for p in run], run_trace_edge[i], run_extra_edges[i]))
        cx.face_frames[cell_id] = (origin, e1, e2)
        is_a = run[1][1]
        push!(is_a ? extra_a : extra_b, cell_id)
        push!(results, (is_a, cell_id))
    end
    push!(pending, (face_id, [cid for (_, cid) in results]))

    return results, trace_edges, extra_a, extra_b, pending
end

"""
Clips one *curved* 3D face `face_id` (a patch of `fnode.curve`, e.g. a
segment's own cap) by an ambient bisector `quad` -- flat or curved --
the case `clip_flat_face_3d!` itself refuses. Confirmed as the dominant
real-world blocker for a second 3D segment (57/60 stress trials).

Tractable without a dedicated "surface ∩ surface -> space curve"
primitive because every boundary edge of a curved face is a trace edge
cut from some *flat* neighboring face (`find_flat_neighbor_face`), so the
intersection reduces to an ordinary 2D problem via `restrict_to_plane`.
`quad` can be curved too -- `edge_curve_crossings`/
`quadric_quadric_crossings` already handle two curved conics via the
conic pencil. What has no workaround: a new trace edge confined to both
`Q1` and `quad` is a real space curve with no flat neighbor of its own,
so any *later* clip needing to re-touch it errors loudly there (v1 scope)
rather than mishandling it -- confirmed to matter in practice (a
segment's own second validity-cut plane routinely needs to).

Pieces and new trace edges stay on `Q1` unchanged -- clipping subdivides
extent, doesn't change which surface it's a patch of. Otherwise same as
`clip_flat_face_3d!`: multi-crossing and holes both supported, via the
curved analogues of the flat-face crossing-finding and frame functions.
"""
function clip_curved_face_3d!(cx::CellComplex{3}, face_id::Int, quad::Quadric{3,Float64}, a_piece::Dict{Int,Int}, b_piece::Dict{Int,Int}, edge_cache::Dict{Int,Vector{Tuple{Bool,Int,Int,Int}}}, idxA::Set{VertexIdx}, idxB::Set{VertexIdx})
    fnode = cx.nodes[face_id]
    Q1 = fnode.curve
    Q1 === nothing && error("clip_curved_face_3d!: face $face_id isn't actually curved -- use clip_flat_face_3d! instead")

    floops = face_boundary_faces(cx, face_id)
    if length(floops) == 1
        walk = floops[1]
        hole_walks = Vector{Tuple{Int,Int,Int}}[]
    else
        origin0, e10, e20 = curved_face_local_frame(Q1, cx, vcat(floops...))
        outer_idx = find_outer_loop_3d(cx, floops, origin0, e10, e20)
        outer_idx === nothing && error("clip_curved_face_3d!: face $face_id's own boundary edges form no genuine closed loop at all (every one is part of a dangling spike) -- a face with no real enclosed area, which shouldn't happen")
        walk = floops[outer_idx]
        hole_walks = [floops[i] for i in eachindex(floops) if i != outer_idx]
    end

    extra_a = Int[]
    extra_b = Int[]
    pending = Tuple{Int,Vector{Int}}[]
    trace_verts = Int[]
    flat = Tuple{Bool,Int,Int,Int}[]

    # See `clip_flat_face_3d!`'s own matching declaration for the full
    # rationale: propagates the walk's own established side forward
    # instead of re-deriving every edge's own side from `a_piece`
    # independently, which is what let a single near-tied vertex desync
    # two different edges that both touch it.
    cur_side = nothing

    for (e, fv, tv) in walk
        enode = cx.nodes[e]
        p1id, p2id = enode.subcells[1], enode.subcells[2]

        if haskey(edge_cache, e)
            pieces = edge_cache[e]
        else
            p1, p2 = cx.nodes[p1id].point, cx.nodes[p2id].point
            s1, s2 = haskey(a_piece, p1id), haskey(a_piece, p2id)

            # Same multi-crossing-by-alternation logic as
            # `clip_flat_face_3d!` (see its own docstring), just via the
            # 3D-level crossing finders instead of the local-plane ones
            # directly.
            xs = curved_edge_strict_interior_crossings(cx, e, quad)
            if isempty(xs) && s1 != s2
                crossings = curved_edge_all_crossings(cx, e, quad)
                if length(crossings) == 1
                    xs = crossings
                elseif cur_side !== nothing
                    # See the identical branch in `clip_flat_face_3d!` for
                    # why the walk's own already-established side wins over
                    # a fresh local floating-point guess.
                    s1 = cur_side
                else
                    # First edge of the walk: same exact-ties check as
                    # `clip_top_cell_2d!`/`clip_flat_face_3d!`.
                    t1, t2 = exact_ties(cx, p1id), exact_ties(cx, p2id)
                    if t1 !== nothing && (idxA in t1 || idxB in t1)
                        # keep s1 as-is
                    elseif t2 !== nothing && (idxA in t2 || idxB in t2)
                        s1 = s2
                    else
                        v1, v2 = evaluate(quad, p1), evaluate(quad, p2)
                        s1 = (abs(v1) >= abs(v2) ? v1 : v2) < 0
                    end
                end
            end

            # An edge confined to two curved surfaces at once
            # (`enode.curve2 !== nothing`) only ever gets its crossings
            # via `ruled_trace_all_crossings`'s numerical branch-tracking
            # scan (see its own docstring: "not fully rigorous... a
            # deliberate, documented simplification"), which can lose
            # the true branch and report spurious extra sign flips well
            # past the Bezout bound of 4 -- confirmed reachable (a fresh
            # two-segment repro hit 8). Rather than hard-error over a
            # known numerical limitation of that fallback (as opposed to
            # the *exact* flat-neighbor route, where >4 would mean a
            # genuine bug worth stopping for), fall back to the same
            # "pick the more confident endpoint, don't split" tiebreak
            # used above when a tie can't be resolved cleanly, and warn
            # rather than silently drop the ambiguity.
            if length(xs) > 4 && enode.curve2 !== nothing
                @warn "clip_curved_face_3d!: edge $e's ruled-fallback crossing scan found $(length(xs)) candidate crossings against a curve2-confined edge (likely branch-tracking noise, not a real topology) -- falling back to a single-side tiebreak instead of splitting"
                # Only consult a tiebreak if `a_piece` itself is genuinely
                # ambiguous here (`s1 != s2`) -- when it already agrees,
                # that's the correct, unambiguous answer and must not be
                # overridden by `cur_side`/`exact_ties`/magnitude just
                # because *this* edge's own crossing count happened to be
                # untrustworthy; only the crossing count itself needs
                # discarding. Doing this unconditionally used to be able to
                # manufacture a disagreement on an edge whose own two
                # vertices already agreed perfectly (confirmed as a real,
                # not hypothetical, bug -- see the identical fix above).
                if s1 != s2
                    if cur_side !== nothing
                        s1 = cur_side
                    else
                        t1, t2 = exact_ties(cx, p1id), exact_ties(cx, p2id)
                        if t1 !== nothing && (idxA in t1 || idxB in t1)
                            # keep s1 as-is
                        elseif t2 !== nothing && (idxA in t2 || idxB in t2)
                            s1 = s2
                        else
                            v1, v2 = evaluate(quad, p1), evaluate(quad, p2)
                            s1 = (abs(v1) >= abs(v2) ? v1 : v2) < 0
                        end
                    end
                end
                xs = eltype(xs)[]
            end

            if isempty(xs)
                pieces = [(s1, e, p1id, p2id)]
            else
                length(xs) <= 4 || error("clip_curved_face_3d!: edge $e has $(length(xs)) interior crossings -- two genuine quadrics can cross at most 4 times (Bezout), so this many indicates a bug in the crossing finder, not a valid configuration")
                cuts = [add_cell!(cx, 0, Label(), Int[], x) for x in xs]
                verts = vcat(p1id, cuts, p2id)
                piece_ids = Int[]
                pieces = Tuple{Bool,Int,Int,Int}[]
                for i in 1:length(verts)-1
                    side = isodd(i) ? s1 : !s1
                    # `e` itself can already be a doubly-curved edge
                    # (that's exactly why the crossing finders above might
                    # have needed the `curve2` fallback) -- splitting it
                    # further doesn't change which two surfaces it's
                    # confined to, so every new piece inherits `curve2`
                    # too, the same way `curve` already gets preserved
                    # across a split.
                    pid = add_cell!(cx, 1, Label(), [verts[i], verts[i+1]]; curve=enode.curve, curve2=enode.curve2)
                    push!(pieces, (side, pid, verts[i], verts[i+1]))
                    push!(piece_ids, pid)
                end
                push!(pending, (e, piece_ids))
            end
            edge_cache[e] = pieces
        end

        for (is_a, pid, _, _) in pieces
            push!(is_a ? extra_a : extra_b, pid)
        end
        append!(trace_verts, [p[4] for p in pieces[1:end-1]])
        walk_pieces = fv == p1id ? pieces : [(is_a, pid, to, from) for (is_a, pid, from, to) in reverse(pieces)]
        append!(flat, walk_pieces)
        cur_side = walk_pieces[end][1]
    end

    if isempty(trace_verts)
        side = flat[1][1]
        all(p -> p[1] == side, flat) ||
            error("clip_curved_face_3d!: face $face_id's own edges disagree on side despite finding no crossing anywhere -- a genuine inconsistency, not something this function knows how to resolve")
        return [(side, face_id)], Int[], extra_a, extra_b, pending
    end

    # Cut the face's own cyclic edge-piece sequence into its same-side
    # runs -- the general (any number of cuts) version of the old
    # "exactly one cut" logic; see `clip_flat_face_3d!`'s own matching
    # step for the full rationale (byte-for-byte the same algorithm).
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

    # Each new trace edge lies on Q1 (both endpoints do). If `quad` is
    # flat, the whole arc is `Q1` restricted to `quad`'s own flat plane --
    # the same "curve lives on facets, but an edge can carry it too,
    # combined with *some* flat neighbor's frame" pattern every other
    # curved edge in this codebase already follows; its flat neighbor
    # (once looked up later, e.g. by `sweep_topology_check`) will be
    # `quad`'s own new flat capping face, built by the caller
    # (`clip_top_cell_3d!`). If `quad` is *also* curved, the arc is
    # confined to two curved surfaces at once -- a genuine space curve
    # with no flat neighbor at all -- so `quad` itself is remembered too
    # (`curve2`; see `CellNode`'s own docstring), letting
    # `curved_edge_all_crossings`/`curved_edge_strict_interior_crossings`
    # locate it later via `ruled_trace_all_crossings` instead.
    # Same "single cut needs one *shared* edge, not two coincident ones"
    # subtlety as `clip_flat_face_3d!` -- see its own comment on this same
    # step for the full rationale.
    trace_edges = Int[]
    run_trace_edge = Vector{Int}(undef, length(runs))
    if length(runs) == 2
        first_v, last_v = runs[1][1][3], runs[1][end][4]
        shared_edge = add_cell!(cx, 1, Label(), [last_v, first_v]; curve=Q1, curve2=(is_curved(quad) ? quad : nothing))
        push!(trace_edges, shared_edge)
        run_trace_edge .= shared_edge
    else
        for (i, run) in enumerate(runs)
            first_v, last_v = run[1][3], run[end][4]
            trace_edge = add_cell!(cx, 1, Label(), [last_v, first_v]; curve=Q1, curve2=(is_curved(quad) ? quad : nothing))
            push!(trace_edges, trace_edge)
            run_trace_edge[i] = trace_edge
        end
    end

    # Same hole-placement question as `clip_flat_face_3d!` -- see its own
    # comment for the full rationale -- just using `curved_face_local_frame`'s
    # own approximate (not exact) local frame instead of a flat face's
    # exact one.
    run_extra_edges = [Int[] for _ in runs]
    if !isempty(hole_walks)
        origin, e1, e2 = curved_face_local_frame(Q1, cx, walk)
        for hw in hole_walks
            sample_pt = loop_interior_point_3d(cx, hw, origin, e1, e2)
            ri = findfirst(i -> point_in_edge_loop_3d(cx, vcat([p[2] for p in runs[i]], run_trace_edge[i]), origin, e1, e2, sample_pt), eachindex(runs))
            if ri === nothing
                hole_pts = [cx.nodes[v].point for (_, v, _) in hw]
                hole_center = sum(hole_pts) / length(hole_pts)
                run_center(run) = sum(vcat([cx.nodes[p[3]].point for p in run], [cx.nodes[p[4]].point for p in run])) / (2 * length(run))
                ri = argmin([norm(run_center(run) - hole_center) for run in runs])
                @warn "clip_curved_face_3d!: a hole in face $face_id's own territory didn't land cleanly inside any of the $(length(runs)) resulting piece(s) -- falling back to the nearest piece by bounding-box center"
            end
            append!(run_extra_edges[ri], [e for (e, _, _) in hw])
        end
    end

    results = Tuple{Bool,Int}[]
    for (i, run) in enumerate(runs)
        # Every new piece stays on `Q1` -- clipping subdivides a curved
        # face's own extent, it doesn't change which surface it's a patch
        # of.
        cell_id = add_cell!(cx, 2, Label(), vcat([p[2] for p in run], run_trace_edge[i], run_extra_edges[i]); curve=Q1)
        is_a = run[1][1]
        push!(is_a ? extra_a : extra_b, cell_id)
        push!(results, (is_a, cell_id))
    end
    push!(pending, (face_id, [cid for (_, cid) in results]))

    return results, trace_edges, extra_a, extra_b, pending
end

"""
Merges near-duplicate vertices among `cap_edges`' own endpoints in-place
(union-find-by-distance, like `weld_near_duplicate_vertices!` but scoped
to this cap, without recomputing labels), then drops any edge left with
both endpoints collapsed onto the same vertex.

Needed because multi-crossing makes it reachable for two different faces
(or two edges of the same face) to each independently compute their own
cut point at what's the same point in exact arithmetic but lands a few
ULPs apart in float (confirmed: a two-segment insertion produced four
such near-duplicates, `cyclic_boundary_walks` erroring on the resulting
degree-3 vertex). The codebase's normal weld timing is too late here --
it runs only after every top cell for the current bisector is clipped,
but this function's own check runs immediately, per cell.

Merging vertices alone isn't enough: two different cap edges can end up
connecting the same now-canonical pair once their endpoints collapse --
the one-dimension-up analogue, ported here in miniature from
`weld_duplicate_edges!` since that only runs at the same too-late point.
Confirmed reachable: the same repro still failed a different way with
vertex-welding alone.
"""
function weld_cap_vertices!(cx::CellComplex{3}, cap_edges::Vector{Int}; atol=1e-9)
    verts = unique(v for e in cap_edges for v in cx.nodes[e].subcells)
    length(verts) < 2 && return cap_edges

    parent = Dict(id => id for id in verts)
    function find_root(x)
        while parent[x] != x
            x = parent[x]
        end
        return x
    end
    for i in 1:length(verts), j in (i+1):length(verts)
        a, b = verts[i], verts[j]
        pa, pb = cx.nodes[a].point, cx.nodes[b].point
        d = norm(pa - pb)
        tol = atol * max(1.0, norm(pa), norm(pb))
        d > tol && continue
        ra, rb = find_root(a), find_root(b)
        ra != rb && (parent[ra] = rb)
    end

    groups = Dict{Int,Vector{Int}}()
    for id in verts
        push!(get!(() -> Int[], groups, find_root(id)), id)
    end
    for (_, group) in groups
        length(group) < 2 && continue
        canonical = minimum(group)
        for id in group
            id == canonical && continue
            supersede!(cx, id, [canonical])
        end
    end

    survivors = Int[]
    for e in cap_edges
        a, b = cx.nodes[e].subcells
        if a == b
            supersede!(cx, e, Int[])
        else
            push!(survivors, e)
        end
    end

    # NOTE: tried also requiring same-curve agreement here (mirroring
    # `weld_duplicate_edges!`'s own guard, `multi_points.jl`), on the
    # hypothesis that two genuinely different arcs sharing an endpoint
    # pair were being wrongly collapsed, losing a real boundary piece and
    # causing exactly the odd-degree, unresolvable-by-any-tie-break
    # failures seen in `cyclic_boundary_walks` downstream. Measured net
    # *negative* on a 150-trial 3-segment stress run (19/150 successes,
    # down from 21/150, plus a new `flat_face_frame` failure mode) --
    # reverted rather than kept on a plausible-sounding but unverified
    # story. The actual cause of those odd-degree failures is still open;
    # see `project_3d_segments_milestone.md` (memory) for what was ruled
    # out (this same-curve guard, and `cyclic_boundary_walks`'s own
    # tie-breaking, both couldn't be it) and what wasn't.
    by_endpoints = Dict{Tuple{Int,Int},Vector{Int}}()
    for e in survivors
        a, b = cx.nodes[e].subcells
        push!(get!(() -> Int[], by_endpoints, a <= b ? (a, b) : (b, a)), e)
    end
    deduped = Int[]
    for (_, group) in by_endpoints
        canonical = minimum(group)
        push!(deduped, canonical)
        for e in group
            e == canonical && continue
            supersede!(cx, e, [canonical])
        end
    end
    return deduped
end

"""
Decomposes `cap_edges` (the trace edges collected while clipping every
face of a 3D cell by `quad`, `clip_top_cell_3d!`'s own cap) into the
distinct 2D *faces* of the graph they form on `quad`'s own surface -- a
thin wrapper over `trace_surface_faces` (see its own docstring for the
full argument for why this needs a different algorithm from
`cyclic_boundary_walks`), fixing the normal direction to `quad`'s own
gradient (`2(Mx+b)`, well-defined and globally consistent by
construction, no sign ambiguity -- every cap edge lies exactly on `quad`'s
own zero set) and discarding the from/to direction each returned triple
also carries, since `clip_top_cell_3d!` only ever needs the edge ids
themselves. `clip_top_cell_3d!` expects exactly one face in the common
case; more than one means a genuinely disjoint (not just self-touching)
cut -- still v1-scope unsupported there, and distinguishable from the
self-touching case instead of both erroring alike.
"""
function trace_cap_faces(cx::CellComplex{3}, cap_edges::Vector{Int}, quad::Quadric{3,Float64})
    normal_at(v) = quad.M * cx.nodes[v].point + quad.b
    return [[e for (e, _, _) in f] for f in trace_surface_faces(cx, cap_edges, normal_at)]
end

"""
`clip_by_hyperplane!`'s `N=3` replacement for the general-purpose
per-subcell dictionary assembly, walking `start_id`'s own 2-skeleton
(faces) -- via `clip_flat_face_3d!` for a flat one, `clip_curved_face_3d!`
for a face that's already curved (e.g. an earlier segment's own cap) --
and capping the resulting cut with a new curved face -- the 3D analogue
of `clip_top_cell_2d!`, one dimension up (that one walks a cell's own
1-skeleton and caps with a curved *edge*; this walks a cell's own
2-skeleton and caps with a curved *face*).

Every v1-scope restriction this function used to inherit is gone now:
holes on an individual face, multi-crossing within a single face (any
number of connected cuts), a curved clipping bisector against an
already-curved face, and a multiply-connected cut across the *whole*
cell (several genuinely disjoint cap patches, or patches that touch at a
shared vertex without being simply connected) are all supported --
the last of these via `trace_cap_faces`, which decomposes `cap_edges`
into however many distinct 2D patches the cut's own topology actually
needs, rather than requiring (and erroring when it doesn't get) exactly
one simple loop.

Returns `(results, extra_a, extra_b, extra_cut, pending)`, exactly
`clip_top_cell_2d!`'s own contract, one dimension up.
"""
function clip_top_cell_3d!(cx::CellComplex{3}, start_id::Int, quad::Quadric{3,Float64}, a_piece::Dict{Int,Int}, b_piece::Dict{Int,Int}, idxA::Set{VertexIdx}, idxB::Set{VertexIdx}, global_edge_cache::Union{Nothing,Dict{Int,Vector{Tuple{Bool,Int,Int,Int}}}})
    orig_faces = cx.nodes[start_id].subcells
    extra_a = Int[]
    extra_b = Int[]
    extra_cut = Int[]
    pending = Tuple{Int,Vector{Int}}[]
    cap_edges = Int[]
    a_faces = Int[]
    b_faces = Int[]
    # Normally fresh per call (this top cell's own faces only ever share
    # an edge with *each other*, never with a different top cell) --
    # `global_edge_cache`, when the caller provides one, widens that
    # sharing to every top cell the caller processes against this same
    # `quad` (see `clip_by_hyperplane!`'s own docstring on why).
    edge_cache = global_edge_cache === nothing ? Dict{Int,Vector{Tuple{Bool,Int,Int,Int}}}() : global_edge_cache

    for f in orig_faces
        clip_face! = cx.nodes[f].curve === nothing ? clip_flat_face_3d! : clip_curved_face_3d!
        face_results, trace_edges, fa, fb, fpending = clip_face!(cx, f, quad, a_piece, b_piece, edge_cache, idxA, idxB)
        append!(extra_a, fa)
        append!(extra_b, fb)
        append!(pending, fpending)
        append!(cap_edges, trace_edges)
        for (is_a, fid) in face_results
            push!(is_a ? a_faces : b_faces, fid)
        end
    end

    if isempty(cap_edges)
        (isempty(a_faces) || isempty(b_faces)) ||
            error("clip_top_cell_3d!: no face was split, yet both sides are represented among $start_id's own faces -- a genuine inconsistency")
        is_a = !isempty(a_faces)
        return [(is_a, start_id)], extra_a, extra_b, extra_cut, pending
    end

    cap_edges = weld_cap_vertices!(cx, cap_edges)
    cap_faces = trace_cap_faces(cx, cap_edges, quad)

    # Every cap patch -- whether the cut is one simple loop, several
    # genuinely disjoint ones (separate "windows" between the two
    # territories), or a self-touching one `trace_cap_faces` still traces
    # as a single patch -- is *always* the shared boundary between the
    # same two new 3-cells: the comparison is strictly binary (closer to
    # A or to B), so however many patches the cut's own topology needs,
    # they all separate the same A-side from the same B-side. No per-patch
    # side bookkeeping is needed, unlike `a_faces`/`b_faces` above, which
    # is why this loop is so much simpler than it might look.
    cap_ids = [add_cell!(cx, 2, Label(), face; curve=(is_curved(quad) ? quad : nothing)) for face in cap_faces]
    append!(extra_cut, cap_ids)

    a_id = add_cell!(cx, 3, Label(), vcat(a_faces, cap_ids))
    b_id = add_cell!(cx, 3, Label(), vcat(b_faces, cap_ids))
    push!(extra_a, a_id)
    push!(extra_b, b_id)
    push!(pending, (start_id, [a_id, b_id]))

    return [(true, a_id), (false, b_id)], extra_a, extra_b, extra_cut, pending
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
`A` (`idxA`) and `B` (`idxB`): every descendant cell ends up labeled by
whichever is closer throughout it (or both, on the shared boundary -- a
genuine tie). `A`/`B` can be any `AffineQuadratic`, not just points --
the same dimension-by-dimension assembly works unchanged for a curved
bisector, since everything shape-dependent is factored into
`edge_crossings`/`quadric_crossing_point`, and the new cut cell records
that shape (`curve`). `start_id` need not be the whole complex's only top
cell -- clips just its own descendants (`descendant_nodes_by_dim`), safe
to call repeatedly against different already-partially-built cells.

Processes dimension `0, 1, ..., dim(start_id)` in order, reusing each
dimension's split to determine the next: a cell is left unsplit if all
its subcells ended up on the same side; otherwise replaced by two new
cells sharing a cut cell one dimension down. Every split node is
immediately `supersede!`d, patching any other cell that referenced it.

A genuine exact tie (an existing vertex lying exactly on the bisector) is
handled via `exact_sign`+`symbolic_tiebreak` in the vertex-level loop
below, not a degenerate case needing special handling here. A single edge
crossed more than once by a curved bisector is handled at `N=2` via
`clip_top_cell_2d!`; any other `N`, or a crossing count it can't make
sense of, falls back to this function's own one-crossing-only assembly.
"""
function clip_by_hyperplane!(cx::CellComplex{N}, start_id::Int, A::AffineQuadratic{N,KA,Float64}, B::AffineQuadratic{N,KB,Float64}, idxA::Set{VertexIdx}, idxB::Set{VertexIdx}; preserve_label::Union{Nothing,Label}=nothing, global_edge_cache::Union{Nothing,Dict{Int,Vector{Tuple{Bool,Int,Int,Int}}}}=nothing) where {N,KA,KB}
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
    # An edge whose two endpoints' `a_piece` labels disagree, but whose
    # actual crossing the numeric finder fails to locate, has no answer
    # derivable from the edge and `quad` alone that's first-class to the
    # edge itself rather than to whichever top cell's own walk asked
    # first -- *unless* every top cell sharing that edge, for this same
    # `quad`, shares one `edge_cache` (normally fresh per call, scoped to
    # a single top cell's own faces -- see `clip_top_cell_3d!`). Callers
    # that process more than one top cell against the same comparison
    # (`insert_features!`'s own `to_process` loop, the dominant real case)
    # can pass one explicitly via `global_edge_cache` so the *first* top
    # cell to resolve a shared edge -- by whatever means, ambiguous
    # fallback included -- commits it for every other cell that later
    # touches the same edge id, instead of each independently re-deriving
    # its own, possibly-disagreeing answer. Confirmed (not assumed) to be
    # the dominant real cause of "edges disagree on side" failures: 41/96
    # in a 150-trial 3-segment stress run were this exact message, far
    # ahead of every other category -- each independent call's own
    # ambiguous-fallback resolution depends on *which walk got there
    # first, from which direction* (`cur_side`), not on the edge and
    # `quad` alone, so two top cells sharing a physical edge can (and did)
    # commit two different answers for what must be the same fact. A
    # caller that doesn't pass one (every call site except
    # `insert_features!`'s comparison loop, for now) gets a fresh,
    # unshared cache per call -- identical to previous behavior, since
    # there's only ever one top cell in play there anyway.
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
        # the winning side, not the tie -- except that side of it is no
        # longer thrown away: `record_exact_tie!` keeps it as a first-class
        # fact on the node itself (see `CellNode.exact_ties`'s docstring),
        # queryable by anything that cares later, independent of which
        # side this particular clip ended up choosing.
        if s == 0
            record_exact_tie!(node, idxA)
            record_exact_tie!(node, idxB)
            s = symbolic_tiebreak(sort(collect(idxA)), sort(collect(idxB)))
        end
        if s < 0
            a_piece[id] = id
        else
            b_piece[id] = id
        end
    end

    use_3d_path = N == 3 && cx.nodes[start_id].dim == 3 && (is_curved(quad) || cell_has_any_curve_3d(cx, start_id))
    if (N == 2 && cx.nodes[start_id].dim == 2) || use_3d_path
        results, ea, eb, ec, pending = N == 2 ? clip_top_cell_2d!(cx, start_id, quad, a_piece, b_piece, idxA, idxB) :
                                        clip_top_cell_3d!(cx, start_id, quad, a_piece, b_piece, idxA, idxB, global_edge_cache)
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
