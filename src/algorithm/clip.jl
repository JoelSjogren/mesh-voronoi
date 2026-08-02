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
Real intersection points of two genuinely curved quadrics with different
`M` (`curve.M != edge_curve.M`). By the tie-boundary invariant, `edge_curve`
always ties the current cell's own winner `f` against some neighbor `g`,
and `curve` (the new clip) always compares that same `f` against the
incoming feature `new` -- so both are `±bisector(f,new)`/`±bisector(f,g)`
for some fixed pair `f`,`new` and `f`,`g` respectively, and (since a stored
curved edge is only ever a point-vs-line bisector -- two lines always meet
along a *straight* tie boundary instead, via `clip_by_hyperplane!`'s
dedicated `K=N-1` vs `K=N-1` dispatch, before a curved edge is ever created)
exactly one of `f`/`g` is a point and the other a line.

`curve`'s sign is fixed (`insert_features!` always computes `bisector(cur_feat,
new_feat)` with the cell's own winner first), but `edge_curve`'s is *not* --
it was computed the same way, but potentially in an *earlier* insertion
round where `f` played the *other* role (the newly-inserted feature being
compared against a then-existing winner `g`), storing `bisector(g,f)` =
`-bisector(f,g)` instead. So which of `curve - edge_curve` or `curve +
edge_curve` equals (up to overall sign, irrelevant to a zero set)
`bisector(g,new)` depends on a construction-history detail neither curve's
own coefficients reveal -- both combinations are tried.

Two structural sub-cases, independent of that sign ambiguity:
  - `f` a point: `g`,`new` are the two lines. `curve`/`edge_curve` are each
    an irreducible parabola, but the correct-sign combination of them
    eliminates `f`'s own `|x-f|²` term algebraically, leaving exactly
    `±(distG(x)² - distNew(x)²)` -- the same "two angle-bisector lines"
    factorization `is_line_pair`/`clip_by_line_pair!` already exploit for
    comparing two lines directly (or, if `g`,`new` share a direction, a
    single line instead -- `distG(x)²-distNew(x)²` degenerates to linear
    exactly then, same as the already-handled equal-`M` case one level up).
  - `f` a line: `f`,`new` are the two lines, so `curve` itself is already
    the pair of angle-bisector lines between them -- no combination needed,
    `curve` factors directly (this one doesn't depend on `edge_curve`'s
    sign at all, since `curve`'s own convention is always fixed).
So: try `curve-edge_curve`, `curve+edge_curve`, then `curve` itself (and
`edge_curve` itself, for symmetry/robustness), taking whichever first comes
out a genuine line or line pair. Intersecting each resulting line with
`edge_curve` (already just `line_meets_quadric`) finds every point on both
original curves: a point on one of a line pair's own factor lines
automatically satisfies whichever combination produced it, and combined
with also lying on `edge_curve`, satisfies `curve` too by construction.
Reduces what would otherwise be a general degree-4 (Bezout) quadric-quadric
intersection to two ordinary line-vs-quadric calls.
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
    factor_source = if is_line_pair(diff)
        diff
    elseif is_line_pair(sum_)
        sum_
    elseif is_line_pair(curve)
        curve
    elseif is_line_pair(edge_curve)
        edge_curve
    else
        error("quadric_quadric_crossings: none of curve, edge_curve, or curve±edge_curve is a genuine line/line-pair -- outside the cases this codebase's feature model can structurally produce")
    end
    (x0a, t̂a), (x0b, t̂b) = line_pair_factors(factor_source)
    pts = vcat(line_meets_quadric(x0a, t̂a, edge_curve, p1, p2; strict=strict),
        line_meets_quadric(x0b, t̂b, edge_curve, p1, p2; strict=strict))
    out = Pt{2,Float64}[]
    for p in pts
        any(q -> norm(p - q) < 1e-9 * max(1.0, norm(p)), out) || push!(out, p)
    end
    return out
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
"""
function cyclic_boundary_walks(cx::CellComplex{2}, edges::Vector{Int})
    n = length(edges)
    edge_verts = [(cx.nodes[e].subcells[1], cx.nodes[e].subcells[2]) for e in edges]
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
            push!(out, (edges[idx], from, to))
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
"""
function find_outer_loop(cx::CellComplex{2}, loops::Vector{Vector{Tuple{Int,Int,Int}}})
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

        if s1 == s2
            # The old all_a/all_b shortcut's own case (both endpoints
            # already agree, via `a_piece`/`b_piece`'s possibly
            # tie-broken sides): a strict interior crossing here used to
            # be an error ("outside the generic case"); now it's the
            # genuine bite/island case this whole function exists to
            # handle, provided it's exactly 2 (0 stays the plain
            # unsplit case, matching the old shortcut exactly).
            xs = edge_strict_interior_crossings(quad, p1, p2, enode.curve)
            if isempty(xs)
                pieces = [(s1, e, p1id, p2id)]
            elseif length(xs) == 2
                cuts = (add_cell!(cx, 0, Label(), Int[], xs[1]), add_cell!(cx, 0, Label(), Int[], xs[2]))
                verts = (p1id, cuts[1], cuts[2], p2id)
                sides = (s1, !s1, s1)
                pieces = Tuple{Bool,Int,Int,Int}[]
                for i in 1:3
                    pid = add_cell!(cx, 1, Label(), [verts[i], verts[i+1]]; curve=enode.curve)
                    push!(pieces, (sides[i], pid, verts[i], verts[i+1]))
                end
                push!(pending, (e, [p[2] for p in pieces]))
            else
                error("clip_by_hyperplane!: edge $e has $(length(xs)) interior crossings while both endpoints agree on side -- only 0 or 2 are handled here (more are algebraically possible for two curved quadrics, just not implemented)")
            end
        else
            # Exactly the old dim==1 split branch, unchanged: endpoints
            # disagree, so expect exactly one (inclusive-tolerance)
            # crossing -- a root landing near a tie-broken endpoint here
            # is `weld_near_duplicate_vertices!`'s job to clean up later,
            # not something to special-case away in this function.
            crossings = edge_points_at_crossings(quad, p1, p2, enode.curve)
            length(crossings) == 1 || error("clip_by_hyperplane!: edge $e has $(length(crossings)) crossings (expected exactly 1) -- not handled here (disagreeing endpoints only forces an odd count, and more than 1 is possible in principle for two curved quadrics)")
            x = only(crossings)
            cut = add_cell!(cx, 0, Label(), Int[], x)
            pid1 = add_cell!(cx, 1, Label(), [p1id, cut]; curve=enode.curve)
            pid2 = add_cell!(cx, 1, Label(), [cut, p2id]; curve=enode.curve)
            pieces = [(s1, pid1, p1id, cut), (s2, pid2, cut, p2id)]
            push!(pending, (e, [pid1, pid2]))
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
                error("clip_by_hyperplane!: bisector is a genuine pair of two distinct lines (a line-vs-line comparison), and this run's own two cut points lie on *different* lines of that pair -- connecting them with a single curved edge would be wrong (there is no single line through them that stays on the bisector); handling this properly needs treating the two lines as two separate flat clips, not yet implemented")
        end
        arc_id = add_cell!(cx, 1, Label(), [last_v, first_v]; curve=arc_curve)
        push!(extra_cut, arc_id)
        push!(arc_ids, arc_id)
    end

    run_extra_edges = [Int[] for _ in runs]
    for hw in hole_walks
        sample_pt = cx.nodes[hw[1][2]].point
        ri = findfirst(i -> point_in_edge_loop(cx, vcat([p[2] for p in runs[i]], arc_ids[i]), sample_pt), eachindex(runs))
        ri === nothing && error("clip_top_cell_2d!: a hole in start_id=$start_id's territory doesn't land inside any of the $(length(runs)) resulting piece(s) -- shouldn't happen for a well-formed multiply-connected cell")
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
