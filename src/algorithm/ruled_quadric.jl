"""
Every genuinely curved quadric this codebase produces has `rank(M) <= 2`
(point-vs-segment rank 1, segment-vs-segment rank 2 indefinite -- never
3). That's what makes an explicit two-parameter surface description
possible: decompose `x` along `M`'s eigenbasis into the part `M` acts on
and the null part, then solve for the null coordinate from the linear/
constant terms.

Rank 2 additionally rotates the raw eigenbasis `(P,Q)` (`λ₁P²+λ₂Q²`,
`λ₁>0>λ₂`) into the "light-cone" basis `(F,R) = (√λ₁P+√|λ₂|Q, √λ₁P-√|λ₂|Q)`,
for which `λ₁P²+λ₂Q² = F·R` exactly -- the algebraic content of "a
hyperbolic paraboloid is doubly-ruled": fixing `F` and varying `R` traces
a straight line lying entirely on the surface, since there's no `R²` term
left. Rank 1 needs no rotation: `Q` (the already-null eigenvector) has no
`Q²` term either, so fixing `P` is already a ruling line.

Unified as one convention: `evaluate(quad,x) = a·F² + b·F·R + 2dF·F +
2dR·R + 2dN·N + c`, `(a,b)=(0,1)` for rank 2, `(λ,0)` for rank 1 --
deliberately no `R²` term either way, so "fix `F`, vary `R`" is always a
ruling line (`ruling_line` below).

This avoids needing a general "quadric ∩ quadric -> space curve"
primitive: an edge confined to two curved quadrics (`Q1`/`Q2`, stored as
`CellNode.curve`/`curve2`) has a third quadric's crossing
(`ruled_trace_crossing`) found by parametrizing `Q2` via its ruling, then
solving `Q1`'s intersection with each ruling line (`line_quadric_roots`,
an ordinary quadratic) -- reducing a 3D problem to a 1D root-find in `F`.
"""
function ruled_frame(quad::Quadric{3,Float64})
    Mf = Symmetric(Matrix(quad.M))
    decomp = eigen(Mf)
    vals = decomp.values
    tol = 1e-9 * max(1.0, maximum(abs.(vals)))
    nz = findall(v -> abs(v) > tol, vals)

    if length(nz) == 2
        i, j = nz
        lam_i, lam_j = vals[i], vals[j]
        sign(lam_i) == sign(lam_j) &&
            error("ruled_frame: rank-2 but definite (both nonzero eigenvalues share a sign) -- not a ruled quadric")
        lam_i < lam_j && ((i, j) = (j, i))
        lam1, lam2 = vals[i], vals[j]   # lam1 > 0 > lam2
        e1 = SVector{3,Float64}(decomp.vectors[:, i])
        e2 = SVector{3,Float64}(decomp.vectors[:, j])
        n_idx = only(setdiff(1:3, (i, j)))
        n = SVector{3,Float64}(decomp.vectors[:, n_idx])
        eF = sqrt(lam1) * e1 + sqrt(-lam2) * e2
        eR = sqrt(lam1) * e1 - sqrt(-lam2) * e2
        a, b = 0.0, 1.0
    elseif length(nz) == 1
        i = only(nz)
        a = vals[i]
        eF = SVector{3,Float64}(decomp.vectors[:, i])
        others = setdiff(1:3, (i,))
        eR = SVector{3,Float64}(decomp.vectors[:, others[1]])
        n = SVector{3,Float64}(decomp.vectors[:, others[2]])
        b = 0.0
    else
        error("ruled_frame: quadric has rank $(length(nz)) (expected 1 or 2) -- not a curved quadric this codebase should ever produce")
    end

    L = vcat(eF', eR', n')
    Linv = inv(L)
    dF = dot(quad.b, Linv[:, 1])
    dR = dot(quad.b, Linv[:, 2])
    dN = dot(quad.b, Linv[:, 3])
    abs(dN) < 1e-9 * max(1.0, norm(quad.b)) &&
        error("ruled_frame: degenerate (the linear term has no component along the one solvable direction)")
    return (; a, b, eF, eR, n, Linv, dF, dR, dN, c=quad.c)
end

"""Point at ruled parameters `(F,R)` on `frame`'s quadric -- exact, not approximate."""
function ruled_point(frame, F::Float64, R::Float64)
    N = -(frame.a * F^2 + frame.b * F * R + 2 * frame.dF * F + 2 * frame.dR * R + frame.c) / (2 * frame.dN)
    return frame.Linv * SVector(F, R, N)
end

"""Ruled parameters `(F,R)` of a point already known to lie on `frame`'s quadric."""
ruled_params(frame, x::Pt{3,Float64}) = (dot(x, frame.eF), dot(x, frame.eR))

"""The straight ruling line `{x(F0,R) : R∈ℝ}` as `(origin, direction)`."""
function ruling_line(frame, F0::Float64)
    N0 = -(frame.a * F0^2 + 2 * frame.dF * F0 + frame.c) / (2 * frame.dN)
    slope = -(frame.b * F0 + 2 * frame.dR) / (2 * frame.dN)
    origin = frame.Linv * SVector(F0, 0.0, N0)
    direction = frame.Linv * SVector(0.0, 1.0, slope)
    return origin, direction
end

"""Real roots of `evaluate(quad, origin + t*direction) = 0`, the whole line (not a `[0,1]` chord)."""
function line_quadric_roots(quad::Quadric{3,Float64}, origin::Pt{3,Float64}, direction::Pt{3,Float64})
    Md = quad.M * direction
    A = dot(direction, Md)
    B = 2 * dot(origin, Md) + 2 * dot(quad.b, direction)
    C = evaluate(quad, origin)
    return quadratic_roots(A, B, C)
end

"""
Where quadric `quad3` crosses the arc of `Q1 ∩ Q2` from `p1` to `p2` --
needed for a boundary edge confined to two curved surfaces at once
(`CellNode.curve2`). Traces via `frame` (`Q2`'s ruled structure): at each
fixed `F`, the arc's `R` is the root of `Q1` on that ruling line closest
to the guess, turning "does `quad3` cross" into a 1D bisection in `F`.

A vertex sitting numerically on `quad3`'s own zero set is the routine
case here (often *why* the edge needed splitting), so raw endpoint
evaluation can be noise-dominated -- this scans interior points along the
arc first to find a genuine bracket, then bisects within it, rather than
trusting the two endpoints directly.

Not fully exact (numerical bisection, not closed-form) -- a deliberate,
documented simplification for this narrow case. `ruled_trace_all_crossings`
generalizes to every crossing; this wraps it with an "expect exactly one"
assertion.
"""
function ruled_trace_crossing(Q1::Quadric{3,Float64}, quad3::Quadric{3,Float64}, frame, p1::Pt{3,Float64}, p2::Pt{3,Float64})
    points = ruled_trace_all_crossings(Q1, quad3, frame, p1, p2)
    length(points) == 1 || error("ruled_trace_crossing: found $(length(points)) crossings along this edge's own arc (expected exactly one)")
    return only(points)
end

"""
Every point (not just one) where `quad3` crosses the arc of `Q1 ∩ Q2`
from `p1` to `p2` -- the multi-crossing counterpart of
`ruled_trace_crossing` (up to 4 via Bezout), same tracing machinery but
scanning the whole sampled arc for every bracket instead of just the
first. Returns points in arc order, empty if no sign change is found.

Scans from *both* endpoints toward each other (see `_scan_branch`) instead
of only forward from `p1`, and unions whatever each direction resolves.
`Q1 ∩ Q2` genuinely can have several disjoint branches (confirmed via the
tie-locus gallery: 3 generic lines' pairwise-bisector intersections split
into up to 4 disconnected space curves) -- when `p1`/`p2` sit on branches
that don't connect via a single monotonic sweep in `F`, a one-directional
scan runs off the near end's branch and finds no real intersection at all
partway across (confirmed via a direct repro: both endpoints verify
exactly on `Q1`, degree-4 apart in `F`, yet `Q1` has no real point on the
ruling line roughly a third of the way between them). Scanning from the
far endpoint too recovers crossings on its own reachable stretch even
when the near-end scan can't get there.
"""
function ruled_trace_all_crossings(Q1::Quadric{3,Float64}, quad3::Quadric{3,Float64}, frame, p1::Pt{3,Float64}, p2::Pt{3,Float64})
    F0, R0 = ruled_params(frame, p1)
    F1, R1 = ruled_params(frame, p2)

    fwd = _scan_branch(Q1, quad3, frame, F0, R0, F1)
    # The forward scan already covers the whole range -- original,
    # battle-tested behavior, unchanged.
    fwd.reached_end && return _merge_close_points(fwd.points)

    bwd = _scan_branch(Q1, quad3, frame, F1, R1, F0)
    bwd.reached_end && return reverse(_merge_close_points(bwd.points))

    # Neither direction made it across -- union whatever each resolved
    # (in arc order from its own starting endpoint), rather than
    # discarding a partial, still-useful answer. A genuine gap neither
    # scan can reach (unreachable from *either* known-good endpoint) is
    # logged, not silently guessed at.
    if fwd.lost_at !== nothing && bwd.lost_at !== nothing
        @warn "ruled_trace_all_crossings: lost the arc's own branch scanning from both ends (forward stalled at F=$(fwd.lost_at), backward at F=$(bwd.lost_at)) -- unioning whatever each direction resolved instead of erroring; a crossing strictly inside the unreached gap would be missed" p1 p2
    end
    combined = vcat(fwd.points, reverse(bwd.points))
    return _merge_close_points(combined)
end

"""
Scans `F` from `F_start` (known point `(F_start,R_start)` on `Q1∩Q2`)
toward `F_stop` in up to 65 steps, using the continuity heuristic
documented on `ruled_trace_all_crossings`: at each `F`, pick whichever of
`Q1`'s (up to 2) real intersections with `Q2`'s ruling line at that `F` is
closest to the previous step's resolved `R`, since a straight-line guess
from the two far-apart endpoints can jump between sheets. Stops early
(without erroring) the first time `Q1` has no real intersection with the
ruling line at all -- that step's own `F` is `lost_at`; `reached_end` is
true only if the scan's last successfully-resolved sample is `F_stop`
itself. Returns crossing points against `quad3` found within whatever
prefix of the range was actually resolved, in arc order from `F_start`.
"""
function _scan_branch(Q1::Quadric{3,Float64}, quad3::Quadric{3,Float64}, frame, F_start::Float64, R_start::Float64, F_stop::Float64)
    function branch_point(F::Float64, guess_R::Float64)
        origin, direction = ruling_line(frame, F)
        roots = line_quadric_roots(Q1, origin, direction)
        isempty(roots) && return nothing
        R = roots[argmin(abs.(roots .- guess_R))]
        return origin + R * direction, R
    end

    nsamples = 65
    Fs = [F_start + k * (F_stop - F_start) / (nsamples - 1) for k in 0:nsamples-1]
    branch_pts = Pt{3,Float64}[]
    Rs = Float64[]
    resolved_Fs = Float64[]
    prev_R = R_start
    lost_at = nothing
    for F in Fs
        res = branch_point(F, prev_R)
        if res === nothing
            lost_at = F
            break
        end
        pt, R = res
        push!(branch_pts, pt)
        push!(Rs, R)
        push!(resolved_Fs, F)
        prev_R = R
    end

    n = length(branch_pts)
    gs = evaluate.(Ref(quad3), branch_pts)
    points = Pt{3,Float64}[]
    for k in 2:n
        sign(gs[k]) == sign(gs[k-1]) && continue
        lo, hi = resolved_Fs[k-1], resolved_Fs[k]
        glo = gs[k-1]
        lo_R, hi_R = Rs[k-1], Rs[k]
        for _ in 1:60
            mid = (lo + hi) / 2
            mid_res = branch_point(mid, (lo_R + hi_R) / 2)
            # A bracket's own two endpoints already resolved on the true
            # branch; losing it strictly inside a bisection step this
            # fine (started >= 1/64 of the outer range, halved 60x) would
            # mean the branch itself is discontinuous at sub-numerical-
            # noise scale -- treat as converged at the nearer edge instead
            # of erroring over it.
            mid_res === nothing && break
            mid_pt, mid_R = mid_res
            gmid = evaluate(quad3, mid_pt)
            if sign(gmid) == sign(glo)
                lo, lo_R = mid, mid_R
            else
                hi, hi_R = mid, mid_R
            end
        end
        final = branch_point((lo + hi) / 2, (lo_R + hi_R) / 2)
        final === nothing || push!(points, final[1])
    end

    reached_end = n == nsamples
    return (; points, reached_end, lost_at)
end

# Adjacent sample brackets can each register a sign change around the same
# true crossing during a near-tangency stretch (confirmed: one repro hit 8
# "crossings", double the Bezout bound). Merge consecutive points within
# numerical noise of each other; a genuine widely-separated near-tangency
# isn't resolved here and surfaces downstream as a large crossing count
# instead.
function _merge_close_points(points::Vector{Pt{3,Float64}})
    isempty(points) && return points
    merged = [points[1]]
    for p in points[2:end]
        prev = merged[end]
        tol = 1e-7 * max(1.0, norm(prev), norm(p))
        norm(p - prev) > tol && push!(merged, p)
    end
    return merged
end
