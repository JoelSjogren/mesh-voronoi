"""
A group of one or more input simplices whose currently-active feature is
*literally the same quadratic function* -- detected via exact `face`
equality (e.g. two segments sharing an endpoint both have `face={v}`),
not floating-point tolerance. `labels` is the set of simplex indices in
the group.
"""
struct QuadraticGroup
    quad::Quadratic
    face::Set{VertexIdx}
    labels::Set{Int}
end

function group_active_features(feats::Vector{Feature})
    groups = Dict{Set{VertexIdx},QuadraticGroup}()
    for f in feats
        if haskey(groups, f.face)
            push!(groups[f.face].labels, f.simplex)
        else
            groups[f.face] = QuadraticGroup(f.quad, f.face, Set([f.simplex]))
        end
    end
    return collect(values(groups))
end

"""
Area-weighted (shoelace) centroid of a polygon given as an ordered point
list. Unlike a plain vertex average, this doesn't get pulled off-center by
edges that happen to be tessellated into more points than others (e.g. a
short edge and a long edge both get the same fixed sample count in
`face_outline`), so it's a much better first guess for elongated or
irregular faces.
"""
function area_centroid(pts::Vector{SVector{2,Float64}})
    n = length(pts)
    a6, cx, cy = 0.0, 0.0, 0.0
    for i in 1:n
        x1, y1 = pts[i]
        x2, y2 = pts[mod1(i + 1, n)]
        cross = x1 * y2 - x2 * y1
        a6 += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    end
    abs(a6) < 1e-14 && return sum(pts) / n   # near-zero area: fall back to plain average
    return SVector(cx, cy) / (3 * a6)
end

"""
A point guaranteed (checked, not just assumed) to lie inside `face_id`.
Faces can be thin slivers, so a naive vertex-average centroid is not
reliable and this tries progressively more robust (and more expensive)
fallbacks.
"""
function interior_sample_point(dcel::DCEL, face_id::Int)
    pts = face_outline(dcel, face_id; n=24)

    centroid = area_centroid(pts)
    point_in_face(dcel, face_id, centroid) && return centroid

    for p in pts
        candidate = (centroid + p) / 2
        point_in_face(dcel, face_id, candidate) && return candidate
    end

    xs, ys = [p[1] for p in pts], [p[2] for p in pts]
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    for nx in (16, 64, 256)
        for x in range(xmin, xmax, length=nx), y in range(ymin, ymax, length=nx)
            candidate = SVector(x, y)
            point_in_face(dcel, face_id, candidate) && return candidate
        end
    end
    error("interior_sample_point: no interior point found for face $face_id (bbox=($xmin,$xmax,$ymin,$ymax))")
end

"""
Canonical (n,d) representation of a line so that two algebraically-opposite
descriptions of the same geometric line (`n` vs `-n`, `d` vs `-d`) compare
equal.
"""
function canonical_line(n::SVector{2,Float64}, d::Float64)
    (n[1] < 0 || (n[1] == 0 && n[2] < 0)) && return -n, -d
    return n, d
end

"""
Round to a fixed precision and normalize signed zero: `Set`/`Dict` use
`isequal`/`hash`, under which `-0.0` and `0.0` are *distinct* keys (unlike
`==`), so without this a duplicate curve can silently evade `curve_key`-based
dedup whenever a computation happens to land on negative zero.
"""
canon0(x::Real) = (r = round(x; digits=6); iszero(r) ? zero(r) : r)
canon0(v::SVector) = canon0.(v)

"""
Identity key for deduplicating curves: two different pairs of quadratics can
legitimately produce the exact same bisector curve (e.g. two segments
sharing an endpoint each generate a "point vs. the other's line" bisector
that degenerates to the same perpendicular line).
"""
function curve_key(c::Line)
    n, d = canonical_line(c.n, c.d)
    return (:line, canon0(n), canon0(d))
end
curve_key(c::Parabola) = (:parabola, canon0(c.p), canon0(c.n), canon0(c.d))

"""
The classification atom for a winning group at `sample`: which sub-simplex
(`face`) is closest, which `side` of it (only meaningful for a
codimension-1 feature -- a segment interior in 2D, the one case with a
single well-defined normal direction; a vertex's orthogonal complement is a
whole subspace, so it has no discrete side), and which input `simplices`
realize it.
"""
function atom_of(g::QuadraticGroup, sample::SVector{2,Float64})
    side = g.quad isa LineQuadratic ? (dot(g.quad.n, sample) - g.quad.d >= 0) : nothing
    return (face=g.face, side=side, simplices=g.labels)
end

"""
Project a classification-atom label down to the plain set of winning input
simplices -- what `brute_force_label` (the oracle) produces. Kept as a
separate, trivially-correct function rather than folding face/side
granularity into the oracle itself.
"""
simplices_of(label::Set) = reduce(union, (a.simplices for a in label); init=Set{Int}())
simplices_of(::Nothing) = nothing

"""
The classification-atom label (a `Set` of tied atoms) that wins at `x`
among `feats` -- the point-level core shared by `label_all_faces!` (called
once per live face, at that face's own interior sample point) and the
raster preview (`label_grid`, called once per grid point, independent of
any DCEL).
"""
function atom_label_at(feats::Vector{Feature}, x::SVector{2,Float64})
    active = [f for f in feats if is_valid(f.validity, x)]
    groups = group_active_features(active)
    vals = [sqdist(g.quad, x) for g in groups]
    m = minimum(vals)
    winners = findall(v -> v <= m + 1e-9 * max(1.0, m), vals)
    return Set(atom_of(groups[w], x) for w in winners)
end

"""
Label every live face from its own interior sample point, using only
`feats` (the incremental construction calls this once per simplex
inserted so far, so `feats` grows over the course of the pipeline).
"""
function label_all_faces!(dcel::DCEL, feats::Vector{Feature})
    for face_id in eachindex(dcel.faces)
        face_id == dcel.outer_face && continue
        dcel.faces[face_id].halfedge == -1 && continue
        sample = interior_sample_point(dcel, face_id)
        dcel.faces[face_id].label = atom_label_at(feats, sample)
    end
end

"""
Coalesce adjacent faces that ended up with the same label.
"""
function merge_equal_label_faces!(dcel::DCEL)
    changed = true
    while changed
        changed = false
        for h in eachindex(dcel.halfedges)
            he = dcel.halfedges[h]
            he.face == 0 && continue
            t = he.twin
            fa, fb = he.face, dcel.halfedges[t].face
            fa == fb && continue
            (fa == dcel.outer_face || fb == dcel.outer_face) && continue
            la, lb = dcel.faces[fa].label, dcel.faces[fb].label
            (la === nothing || lb === nothing) && continue
            la != lb && continue
            merge_edge!(dcel, h)
            changed = true
            break
        end
    end
end

function merge_edge!(dcel::DCEL, h::Int)
    he = dcel.halfedges[h]
    t = he.twin
    ht = dcel.halfedges[t]
    fa, fb = he.face, ht.face

    h_prev, h_next = he.prev, he.next
    t_prev, t_next = ht.prev, ht.next

    dcel.halfedges[h_prev].next = t_next
    dcel.halfedges[t_next].prev = h_prev
    dcel.halfedges[t_prev].next = h_next
    dcel.halfedges[h_next].prev = t_prev

    cur = t_next
    while true
        dcel.halfedges[cur].face = fa
        cur = dcel.halfedges[cur].next
        cur == t_next && break
    end

    dcel.faces[fa].halfedge = h_prev
    dcel.faces[fb].halfedge = -1
    he.face = 0
    ht.face = 0
end

live_faces(dcel::DCEL) = [f for f in eachindex(dcel.faces) if f != dcel.outer_face && dcel.faces[f].halfedge != -1]

"""
Polish pass (not required for correctness): remove redundant degree-2
"pass-through" vertices, where a vertex's only two incident edges lie on
the *same* curve. Splices each one out, merging its two incident edges
back into one on both sides.
"""
function dissolve_degree2_vertices!(dcel::DCEL)
    for v in eachindex(dcel.vertices)
        outgoing = [h for h in eachindex(dcel.halfedges) if dcel.halfedges[h].face != 0 && dcel.halfedges[h].origin == v]
        length(outgoing) == 2 || continue
        h_a, h_b = outgoing
        curve_key(dcel.halfedges[h_a].curve) == curve_key(dcel.halfedges[h_b].curve) || continue

        t_a, t_b = dcel.halfedges[h_a].twin, dcel.halfedges[h_b].twin
        (dcel.halfedges[t_a].next == h_b && dcel.halfedges[t_b].next == h_a) || continue

        h_b_next, h_a_next = dcel.halfedges[h_b].next, dcel.halfedges[h_a].next
        dcel.halfedges[t_a].next = h_b_next
        dcel.halfedges[h_b_next].prev = t_a
        dcel.halfedges[t_b].next = h_a_next
        dcel.halfedges[h_a_next].prev = t_b
        dcel.halfedges[t_a].twin = t_b
        dcel.halfedges[t_b].twin = t_a

        fa, fb = dcel.halfedges[t_a].face, dcel.halfedges[t_b].face
        dcel.faces[fa].halfedge in (h_a, h_b) && (dcel.faces[fa].halfedge = t_a)
        dcel.faces[fb].halfedge in (h_a, h_b) && (dcel.faces[fb].halfedge = t_b)

        dcel.halfedges[h_a].face = 0
        dcel.halfedges[h_b].face = 0
    end
end

"""
Insert `curve` unless a geometrically-identical curve has already been
inserted (tracked via `seen`, shared across *every* own-line inserted
anywhere in the incremental construction). Own-lines are genuinely global
facts -- once inserted, permanently present everywhere they could be scoped
to -- so it's correct and sufficient to remember "have we ever inserted
this" once, globally, forever.
"""
function insert_curve_deduped!(dcel::DCEL, curve::Curve, seen::Set{Any}; candidate_faces=nothing)
    key = curve_key(curve)
    key in seen && return candidate_faces
    push!(seen, key)
    return insert_curve!(dcel, curve; candidate_faces=candidate_faces)
end

"""
Whether `curve` already runs along some edge currently bounding one of
`scope`'s faces -- checked structurally (against the live boundary itself)
rather than via a global "ever inserted" flag, since a bisector insertion is
local: the exact same algebraic curve can be the right thing to insert
independently into a different, disconnected face later (two separate
regions both currently won by the same quadratic, both now being compared
against the same newly-incorporated feature), so a global flag would
wrongly block that second, unrelated insertion. Checking the live boundary
instead is correct in *every* case: it catches a bisector degenerating into
a line some simplex's own cut line already put right here (e.g. comparing a
vertex against a segment whose supporting line that vertex happens to lie
on) and a bisector recurring in a region descended from a face that already
got this exact curve in an earlier round -- both of which are genuine
double-insertions that would otherwise produce degenerate zero-width
faces -- while never flagging a face this curve hasn't actually touched.
"""
function curve_on_scope_boundary(dcel::DCEL, curve::Curve, scope::Set{Int})
    key = curve_key(curve)
    for f in scope
        for h in face_halfedges(dcel, f)
            curve_key(dcel.halfedges[h].curve) == key && return true
        end
    end
    return false
end

"""
Insert simplex `i`'s own splitting lines into `dcel`. Two lines per segment
for the usual feature-validity cut (perpendicular to it, through each
endpoint) *plus* a third: the segment's own supporting line. Without that
third line, a cell where "interior of segment i" wins would straddle both
sides of the segment (the interior validity strip spans both sides by
construction), making `side` ill-defined there.
"""
function insert_own_lines!(dcel::DCEL, complex::InputComplex, i::Int, seen::Set{Any})
    s = complex.simplices[i]
    s isa SegmentSimplex || return
    a, b = complex.coords[s.a], complex.coords[s.b]
    t̂ = (b - a) / norm(b - a)
    n̂ = SVector(-t̂[2], t̂[1])
    for (n, d) in (canonical_line(t̂, dot(t̂, a)), canonical_line(t̂, dot(t̂, b)), canonical_line(n̂, dot(n̂, a)))
        insert_curve_deduped!(dcel, Line(n, d), seen)
    end
end

"""
Incorporate simplex `i` into the diagram built so far (`feats_so_far`,
*not* including `i`): for every currently live face, find whichever of
`i`'s own features is locally active there, and compare it against that
face's current winner (recomputed the same way `label_all_faces!` does --
any one tied representative works, since tied quadratics are equal by
definition). Faces are then grouped by the exact `(winner, new feature)`
pair they need compared -- *not* handled one face at a time -- and one
combined `insert_curve!` call is made per group, scoped to the union of
every face in it.

That grouping step matters, not just as a batching optimization: two
distinct live faces can independently need the identical bisector (e.g. two
separate regions currently won by the same quadratic, both now bordering
simplex `i`), and if those two faces also happen to be *adjacent* to each
other, inserting the same curve into each one's scope separately would
create two independent parallel chords between the same two vertices
instead of one shared edge -- `insert_curve!` has no way to notice, from
inside one call, that a different call already drew the identical line one
face over. Routing every face that needs a given comparison through a
single call sidesteps this entirely: `insert_curve!` already accepts a
`candidate_faces` scope spanning multiple (possibly non-adjacent) faces and
finds all the real crossings against their combined boundary in one pass,
so adjacent members of the same group end up correctly sharing the one
resulting edge.
"""
function incorporate_simplex!(dcel::DCEL, complex::InputComplex, i::Int, feats_so_far::Vector{Feature})
    isempty(feats_so_far) && return   # nothing yet to compare against; label_all_faces! will pick this up
    feats_i = features(complex, i)

    groups = Dict{Tuple{Set{VertexIdx},Set{VertexIdx}},Tuple{QuadraticGroup,Feature,Set{Int}}}()
    for face_id in live_faces(dcel)
        sample = interior_sample_point(dcel, face_id)
        feat_i = only(f for f in feats_i if is_valid(f.validity, sample))

        active_old = [f for f in feats_so_far if is_valid(f.validity, sample)]
        groups_old = group_active_features(active_old)
        vals_old = [sqdist(g.quad, sample) for g in groups_old]
        m = minimum(vals_old)
        cur = groups_old[findfirst(v -> v <= m + 1e-9 * max(1.0, m), vals_old)]

        # If the new feature is literally the same classification atom as
        # the current winner (e.g. simplex i shares a vertex with an
        # already-incorporated simplex -- routine, not rare), they're tied
        # everywhere already and `label_all_faces!`'s face-based grouping
        # will merge them on its own; no curve to insert, and the
        # point-vs-point bisector of two identical points is undefined
        # (zero-length normal) besides.
        cur.face == feat_i.face && continue

        key = (cur.face, feat_i.face)
        if haskey(groups, key)
            push!(groups[key][3], face_id)
        else
            groups[key] = (cur, feat_i, Set([face_id]))
        end
    end

    for (cur, feat_i, faces) in values(groups)
        scope = faces
        for curve in bisector(cur.quad, feat_i.quad)
            curve_on_scope_boundary(dcel, curve, scope) && continue
            scope = insert_curve!(dcel, curve; candidate_faces=scope)
        end
    end
end

"""
Incorporate one already-appended input simplex (`complex.simplices[i]`) into
a `dcel`/`feats_so_far`/`seen_curves` that already correctly reflect every
earlier simplex -- the single incremental step `compute_output_complex`
repeats once per simplex. Factored out so a live/interactive caller (see
`examples/interactive_demo.jl`) can run just this one new step against its
own persistent state instead of rebuilding the whole diagram from scratch
every time a point or segment is added.
"""
function incorporate_and_label!(dcel::DCEL, complex::InputComplex, i::Int, feats_so_far::Vector{Feature}, seen_curves::Set{Any})
    insert_own_lines!(dcel, complex, i, seen_curves)
    incorporate_simplex!(dcel, complex, i, feats_so_far)
    append!(feats_so_far, features(complex, i))
    label_all_faces!(dcel, feats_so_far)
    merge_equal_label_faces!(dcel)
end

"""
Full pipeline: input simplicial complex -> exact output complex (a DCEL
whose live faces are labeled by the set of classification atoms attaining
the minimum distance there). Builds incrementally, one input simplex at a
time, keeping the diagram fully consolidated (merged) after every step --
unlike inserting every pairwise bisector among all active groups at once,
this only ever creates curves that are locally relevant when they're
inserted, so there's no separate over-refine-then-merge phase and no need
to track curve->face provenance across the whole complex.
"""
function compute_output_complex(complex::InputComplex)
    xmin, xmax, ymin, ymax = padded_bbox(complex.coords; pad=0.3)
    dcel = init_bbox_dcel(xmin, xmax, ymin, ymax)
    seen_curves = Set{Any}()
    feats_so_far = Feature[]

    for i in eachindex(complex.simplices)
        incorporate_and_label!(dcel, complex, i, feats_so_far, seen_curves)
    end
    dissolve_degree2_vertices!(dcel)
    return dcel
end

"""
Point location by label: walk the DCEL to find which live face contains
`x`, and return its label. Used both for cross-validation against
`brute_force_label` (via `simplices_of`) and for the hover query.
"""
function label_at(dcel::DCEL, x::SVector{2,Float64})
    f = locate_face(dcel, x, Set(live_faces(dcel)))
    f === nothing && return nothing
    return dcel.faces[f].label
end
