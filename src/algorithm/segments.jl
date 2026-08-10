"""Every currently-live top-dimensional (`dim == N`) cell id."""
top_cell_ids(cx::CellComplex{N}) where {N} = [id for id in eachindex(cx.nodes) if !haskey(cx.superseded_by, id) && cx.nodes[id].dim == N]

"""
Merges every group of currently-live, mutually-adjacent top-dimensional
cells sharing the same label into one cell each, in place. "Adjacent"
means sharing an immediate `(N-1)`-dimensional subcell -- union-find over
shared subcells, so a whole chain merges in one pass. Cells sharing a
label without being connected this way (legitimately at disconnected
locations) are correctly left separate.

Fixes a real construction gap: nothing prevents two neighboring cells
from ending up with the same label after enough insertions, leaving a
purely historical "scar" edge. Left unmerged, `find_containing_cell` only
finds *one* piece, so a query in "the other half" reports the wrong cell.

Each merged group's new cell gets the subcells referenced by only *one*
member (the outer boundary; a subcell referenced by two is a scar edge
being dissolved). Originals are `supersede!`d to the new cell.
"""
function merge_adjacent_same_label_cells!(cx::CellComplex{N}) where {N}
    by_label = Dict{Label,Vector{Int}}()
    for id in top_cell_ids(cx)
        push!(get!(() -> Int[], by_label, cx.nodes[id].label), id)
    end

    for (label, ids) in by_label
        length(ids) < 2 && continue

        parent = Dict(id => id for id in ids)
        function find_root(x)
            while parent[x] != x
                x = parent[x]
            end
            return x
        end
        subcell_owners = Dict{Int,Vector{Int}}()
        for id in ids, s in cx.nodes[id].subcells
            push!(get!(() -> Int[], subcell_owners, s), id)
        end
        for (_, owners) in subcell_owners
            length(owners) < 2 && continue
            # A subcell shared by 2+ same-label cells at once (e.g. a
            # genuine multi-way tie at N>=3) makes all of them mutually
            # adjacent -- union them all in, not just the first pair, or
            # the merged cell ends up with a hole in its boundary.
            for k in 2:length(owners)
                ra, rb = find_root(owners[1]), find_root(owners[k])
                ra != rb && (parent[ra] = rb)
            end
        end

        components = Dict{Int,Vector{Int}}()
        for id in ids
            push!(get!(() -> Int[], components, find_root(id)), id)
        end

        for comp in values(components)
            length(comp) < 2 && continue

            # Two group members can, rarely, turn out to be exact
            # duplicates (identical subcell *set*) -- a construction
            # defect, not a legitimate adjacency. Left as-is, every edge
            # gets counted >= 2 times, which can leave zero `external`
            # edges for the whole component (fatal: `add_cell!` with no
            # subcells). Collapsed by keeping one representative per
            # distinct subcell-set before the external/internal count.
            seen = Dict{Set{Int},Int}()
            deduped = Int[]
            for id in comp
                sig = Set(cx.nodes[id].subcells)
                if haskey(seen, sig)
                    supersede!(cx, id, [seen[sig]])
                else
                    seen[sig] = id
                    push!(deduped, id)
                end
            end
            comp = deduped
            length(comp) < 2 && continue

            counts = Dict{Int,Int}()
            for id in comp, s in cx.nodes[id].subcells
                counts[s] = get(counts, s, 0) + 1
            end
            external = [s for (s, c) in counts if c == 1]
            internal = [s for (s, c) in counts if c > 1]
            merged_id = add_cell!(cx, N, label, external)
            for id in comp
                supersede!(cx, id, [merged_id])
            end
            # Dissolved (internal) subcells are still technically "live"
            # unless explicitly retired here. Relabeling them (rather than
            # retiring) would be wrong, not approximate: they'd become
            # permanently unreachable from the live tree, keeping a stale
            # label forever once anything here changes.
            for s in internal
                supersede_if_orphaned!(cx, s)
            end
        end
    end
    assert_label_bbox_invariant(cx)
    return nothing
end

"""
If `id` is no longer referenced by anything live, marks it `supersede!`d
with no replacement and recurses into its subcells; otherwise leaves it
untouched. An "internal" subcell from `merge_adjacent_same_label_cells!`
can also be shared with a live cell entirely outside the merge (e.g. a
multi-way tie), in which case it must stay live, not be retired just
because it looked internal from the merge's own narrow view.
"""
function supersede_if_orphaned!(cx::CellComplex{N}, id::Int) where {N}
    refs = get(cx.referenced_by, id, Int[])
    any(r -> !haskey(cx.superseded_by, r), refs) && return nothing
    node = cx.nodes[id]
    children = node.dim == 0 ? Int[] : copy(node.subcells)
    supersede!(cx, id, Int[])
    for s in children
        supersede_if_orphaned!(cx, s)
    end
    return nothing
end

"""
Axis-aligned bounding box of `id` -- the geometric extent a label alone
(a combinatorial `Set{Set{VertexIdx}}`) doesn't carry, needed to tell
"genuinely the same place" apart from "coincidentally the same label".
Thin wrapper around the node's own cached `bbox_lo`/`bbox_hi`.
"""
function cell_bbox(cx::CellComplex{N}, id::Int) where {N}
    node = cx.nodes[id]
    return node.bbox_lo, node.bbox_hi
end

"""Whether two boxes overlap or merely touch -- deliberately inclusive, since a shared boundary is exactly the adjacent-but-unmerged case this catches."""
boxes_touch_or_overlap(lo1, hi1, lo2, hi2; atol=1e-9) = all(lo1 .<= hi2 .+ atol) && all(lo2 .<= hi1 .+ atol)

"""
Whether 2D cells `id1`,`id2` share a genuine boundary *edge*, not just a
vertex -- distinguishes a real missed merge (positive-length shared
boundary) from several regions meeting at a single point (geometrically
ordinary, not a bug).
"""
function cells_share_edge(cx::CellComplex{2}, id1::Int, id2::Int; atol=1e-9)
    function edge_endpoints(id)
        out = Tuple{Pt{2,Float64},Pt{2,Float64}}[]
        for e in cx.nodes[id].subcells
            v1, v2 = cx.nodes[e].subcells
            push!(out, (cx.nodes[v1].point, cx.nodes[v2].point))
        end
        return out
    end
    close(a, b) = norm(a - b) < atol * max(1.0, norm(a), norm(b))
    for (p1, p2) in edge_endpoints(id1), (q1, q2) in edge_endpoints(id2)
        ((close(p1, q1) && close(p2, q2)) || (close(p1, q2) && close(p2, q1))) && return true
    end
    return false
end

"""
Whether `pt` lies within `atol` of any edge in `edges` -- straight via
perpendicular distance, curved via the curve's own equation value near
zero plus `on_arc_between` to confirm `pt` is within that specific arc's
bounds. Lets `cells_area_overlap` tell "genuinely inside" apart from
"exactly on the boundary", which the standard even-odd ray-cast can't
(a point on an edge/vertex is a classic degenerate case for it).
"""
function point_near_edge_loop(cx::CellComplex{2}, edges::Vector{Int}, pt::Pt{2,Float64}; atol=1e-6)
    for e in edges
        en = cx.nodes[e]
        v1, v2 = en.subcells
        p1, p2 = cx.nodes[v1].point, cx.nodes[v2].point
        if en.curve === nothing
            d = p2 - p1
            len2 = sum(abs2, d)
            if len2 < 1e-18
                norm(pt - p1) < atol * max(1.0, norm(pt)) && return true
                continue
            end
            t = clamp(dot(pt - p1, d) / len2, 0.0, 1.0)
            norm(pt - (p1 + t * d)) < atol * max(1.0, norm(pt)) && return true
        else
            abs(evaluate(en.curve, pt)) < atol * max(1.0, norm(pt))^2 || continue
            on_arc_between(en.curve, p1, p2, pt) && return true
        end
    end
    return false
end

"""
Whether 2D cells `id1`,`id2` actually overlap in area, checked exactly (not
by sampling their bounding boxes' shared region): a genuine overlap between
two cells built by a sequence of clips essentially always traps at least
one vertex of one cell's own outer boundary strictly inside the other's
(their boundaries would otherwise have to cross an even number of times
without ever separating the two interiors, an exotic degenerate
configuration this doesn't attempt to catch) -- far cheaper than a grid
scan, and this runs inside a check (`assert_label_bbox_invariant`) called
after every single insertion.

A vertex that lies on (not just near, but within `point_near_edge_loop`'s
own tolerance of) the *other* cell's own boundary is skipped rather than
tested: it isn't strictly inside or outside by definition, and the
even-odd ray-cast `point_in_edge_loop` uses below has no reliable answer
for it either way (see `point_near_edge_loop`'s own docstring) -- without
this, two cells merely touching at a shared vertex (an exact id match,
routine and correct wherever three or more territories meet at a point)
or along another cell's curved arc (confirmed reachable: a vertex created
by one clip can sit, geometrically, exactly on an *unrelated* edge's own
curve without being one of its stored endpoints) could get spuriously
reported as overlapping, which is exactly what was happening before this
check existed -- confirmed via a minimal reproduction where the reported
"overlap" vertex was the *one point* two cells genuinely share.
"""
function cells_area_overlap(cx::CellComplex{2}, id1::Int, id2::Int)
    # `find_outer_loop` returns `nothing` for a cell whose own edges are
    # entirely a dangling spike (no genuine closed cycle, hence no real
    # area) -- such a cell can't overlap anything.
    outer_edges(id) = let loops = cyclic_boundary_walks(cx, cx.nodes[id].subcells)
        outer = find_outer_loop(cx, loops)
        outer === nothing ? Tuple{Int,Int,Int}[] : loops[outer]
    end
    ew1, ew2 = outer_edges(id1), outer_edges(id2)
    (isempty(ew1) || isempty(ew2)) && return false
    e1, e2 = [e for (e, _, _) in ew1], [e for (e, _, _) in ew2]
    for (_, v, _) in ew1
        pt = cx.nodes[v].point
        point_near_edge_loop(cx, e2, pt) && continue
        point_in_edge_loop(cx, e2, pt) && return true
    end
    for (_, v, _) in ew2
        pt = cx.nodes[v].point
        point_near_edge_loop(cx, e1, pt) && continue
        point_in_edge_loop(cx, e1, pt) && return true
    end
    return false
end

"""
The invariant `merge_adjacent_same_label_cells!` is supposed to maintain,
checked directly: no two distinct, live top cells sharing a label
genuinely overlap or share a boundary. Warns rather than errors (task
#42/#49, a real, still-open bug, is outstanding, so construction can keep
running for experimentation). Called at the end of every merge pass.

`boxes_touch_or_overlap` alone isn't sound at `N=2` (a cell's bbox can be
much looser than its true footprint for a thin or multiply-connected
shape), so it's kept as a cheap pre-filter only -- the actual verdict is
`cells_share_edge`/`cells_area_overlap`'s exact geometric check.
"""
function assert_label_bbox_invariant(cx::CellComplex{N}) where {N}
    by_label = Dict{Label,Vector{Int}}()
    for id in top_cell_ids(cx)
        push!(get!(() -> Int[], by_label, cx.nodes[id].label), id)
    end
    for (label, ids) in by_label
        length(ids) < 2 && continue
        boxes = [cell_bbox(cx, id) for id in ids]
        for i in 1:length(ids), j in (i+1):length(ids)
            lo1, hi1 = boxes[i]
            lo2, hi2 = boxes[j]
            boxes_touch_or_overlap(lo1, hi1, lo2, hi2) || continue
            if N == 2
                cells_share_edge(cx, ids[i], ids[j]) || cells_area_overlap(cx, ids[i], ids[j]) || continue
            end
            @warn "assert_label_bbox_invariant: cells $(ids[i]) and $(ids[j]) share label $label and have touching/overlapping bounding boxes ($((lo1,hi1)) vs $((lo2,hi2))) -- should have been merged (task #42/#49, downgraded to a warning for now)"
        end
    end
    return nothing
end

"""
A point guaranteed inside `id`: plain vertex average. `N`-generic
fallback, used at any `N != 2`; superseded at `N=2` by the dedicated
method below, exact for non-convex cells too.
"""
interior_sample(cx::CellComplex{N}, id::Int) where {N} = (pts = descendant_points(cx, id); sum(pts) / length(pts))

"""
A point provably inside the simple polygon a single cyclic `loop` bounds,
even non-convex -- unlike a plain vertex-centroid average, which lands
outside for a sufficiently concave shape. Also gives `clip_top_cell_2d!`'s
hole-placement code a genuinely interior point, rather than an arbitrary
boundary vertex (which the even-odd ray cast can then answer either way
for, including "outside every piece").

Exact for any simple polygon: picks the topmost vertex (a locally convex
corner, since nothing can be above it), casts a ray an epsilon below it,
and returns the midpoint of whichever crossing span straddles that
vertex's `x` -- guaranteed interior by local convexity there. Curved
edges solved directly against their stored quadric, not approximated.
"""
function loop_interior_point(cx::CellComplex{2}, loop::Vector{Tuple{Int,Int,Int}})
    verts = unique(v for (_, v, _) in loop)
    pts = [cx.nodes[v].point for v in verts]
    topv = pts[argmax([(p[2], p[1]) for p in pts])]
    y0 = topv[2] - 1e-6 * max(1.0, maximum(p -> abs(p[2]), pts))

    xs = Float64[]
    for (e, _, _) in loop
        en = cx.nodes[e]
        v1, v2 = en.subcells
        p1, p2 = cx.nodes[v1].point, cx.nodes[v2].point
        if en.curve === nothing
            (p1[2] > y0) == (p2[2] > y0) && continue
            push!(xs, p1[1] + (p2[1] - p1[1]) * (y0 - p1[2]) / (p2[2] - p1[2]))
        else
            for x in curve_crossings_at_y(en.curve, y0)
                on_arc_between(en.curve, p1, p2, SVector(x, y0)) && push!(xs, x)
            end
        end
    end
    sort!(xs)
    length(xs) < 2 && return sum(pts) / length(pts)
    i = clamp(searchsortedlast(xs, topv[1]), 1, length(xs) - 1)
    return SVector((xs[i] + xs[i+1]) / 2, y0)
end

"""
`loop_interior_point` applied to 2D cell `id`'s own outer loop --
restricted to it first so a hole's own vertices can't pull the sample
toward or past them.
"""
function interior_sample(cx::CellComplex{2}, id::Int)
    loops = cyclic_boundary_walks(cx, cx.nodes[id].subcells)
    outer = find_outer_loop(cx, loops)
    outer === nothing && error("interior_sample: cell $id has no genuine closed boundary loop at all (every one of its own edges is part of a dangling spike) -- a live top-level cell with no real enclosed area, which shouldn't happen; likely the same open construction issue as a hole independently cut by the same bisector as its container (see the hole-placement investigation notes on `clip_top_cell_2d!`)")
    return loop_interior_point(cx, loops[outer])
end

"""
Label every live boundary cell, at every dimension below `N`, with the
union of whichever next-dimension-up cells reference it -- needed after a
first sub-simplex's own regions are established purely by sampling
(`is_valid`): the top cells get labeled directly, but a boundary between
two differently-labeled neighbors (down to a single *vertex* where several
regions meet) is just as genuine a tie and would otherwise never get
recorded, silently defeating any later code that checks an existing label
instead of re-deriving the tie numerically.

Propagates one dimension at a time, `N-1` down to `0`: each pass only
reads dimension-`d+1` labels, which the *previous* pass already made
correct, so by the time dimension `0` (vertices) is reached, every live
vertex's label is the union of every live edge incident to it, which is
itself already the union of every live cell touching that edge -- the
same "true tied-winner set" a genuinely multi-entry construction would
otherwise only ever get from `clip_by_hyperplane!`'s own per-vertex
`exact_sign`/`symbolic_tiebreak` logic.

Originally only handled `dim == N-1` (edges in 2D): correct whenever a
first entry's own `insert_own_lines!` pass makes at least one real cut,
since a newly-created vertex from that cut gets its own label from
`weld_near_duplicate_vertices!`'s `recompute_feature_label` callback
instead. But a first entry whose every feature has *empty* validity (a
lone point, or a lone unbounded `:line` -- nothing to compare against, so
`insert_own_lines!` is a documented no-op, splitting nothing) leaves the
domain as a single top cell, and the domain's own pre-existing boundary
vertices (from `init_bbox_complex`, predating this entry) never get
touched by *anything* -- confirmed as a real, previously-unknown bug via
a bare single-point/-line stress check, not a hypothetical: every one of
a padded bbox's own 4 corners stayed at their pristine empty `Label()`
forever, since nothing else in the construction ever revisits them for a
domain with only one entry, ever.
"""
function label_boundaries!(cx::CellComplex{N}) where {N}
    for dim in (N-1):-1:0
        for id in eachindex(cx.nodes)
            haskey(cx.superseded_by, id) && continue
            node = cx.nodes[id]
            node.dim == dim || continue
            parents = [p for p in get(cx.referenced_by, id, Int[])
                       if !haskey(cx.superseded_by, p) && cx.nodes[p].dim == dim + 1 && !isempty(cx.nodes[p].label)]
            isempty(parents) && continue
            set_label!(cx, id, reduce(∪, (cx.nodes[p].label for p in parents); init=node.label))
        end
    end
    return nothing
end

"""
Split `start_id` by the plane `{x : n·x = d}` *without* introducing a new
winner -- both resulting pieces keep `start_id`'s own original label.
Pure topological refinement (carving up a cell for a later comparison,
not a real feature competition).

Reuses `clip_by_hyperplane!` itself: the perpendicular bisector of two
points placed symmetrically around a point on the target plane, along its
normal, *is* that plane -- constructs such a pair and passes `start_id`'s
own label through as `preserve_label` so every piece is correctly labeled
from the start.
"""
function clip_by_plane_preserving_label!(cx::CellComplex{N}, start_id::Int, n::Pt{N,Float64}, d::Float64) where {N}
    original_label = cx.nodes[start_id].label
    n̂ = n / norm(n)
    p0 = (d / norm(n)) * n̂
    pA, pB = p0 - n̂, p0 + n̂
    dummy = Set{VertexIdx}()
    clip_by_hyperplane!(cx, start_id, AffineQuadratic(pA), AffineQuadratic(pB), dummy, dummy; preserve_label=original_label)
    return nothing
end

"""Unit normal `n` and offset `d` of the line `q` represents -- `q.basis`'s column is the direction, rotated 90° for the normal."""
function line_normal_form(q::AffineQuadratic{2,1,Float64})
    t = SVector(q.basis[1, 1], q.basis[2, 1])
    n = SVector(-t[2], t[1])
    return n, dot(n, q.p)
end

"""
`clip_by_hyperplane!`'s dedicated path for comparing two line-like
features against each other: unlike every other pairing, this bisector
isn't one connected curve. Squared-distance-to-a-line is `(n·x-d)²`, so
with `u = n_A·x-d_A`, `v = n_B·x-d_B`, the tie condition `u²=v²` factors
as `(u-v)(u+v)=0` -- two straight lines `l1=u-v`, `l2=u+v` crossing at one
point. `A` wins where `l1`,`l2` have opposite signs; `B` where they agree.

Applies `clip_by_hyperplane!` twice: split by `l1=0`, then split each
resulting piece by `l2=0`. Each of the up to four leaf pieces sits
entirely within one quadrant, so its label is overwritten directly here
based on which two clips it came from, rather than trusted from either
clip's own provisional labeling. Sentinel faces (negative, never collide
with a real `VertexIdx`) are internal bookkeeping only -- never visible
outside this function, since every leaf is relabeled before returning and
`fix_boundary_labels!` re-derives boundary labels afterward.
"""
function clip_by_line_pair!(cx::CellComplex{2}, start_id::Int, A::AffineQuadratic{2,1,Float64}, B::AffineQuadratic{2,1,Float64}, idxA::Set{VertexIdx}, idxB::Set{VertexIdx})
    nA, dA = line_normal_form(A)
    nB, dB = line_normal_form(B)
    n1, d1 = nA - nB, dA - dB
    n2, d2 = nA + nB, dA + dB

    # `clip_by_hyperplane!`'s own return value is a legacy scalar (one id
    # per side) that silently keeps only the *last* same-side piece when a
    # clip produces more than one. Every piece is still correctly
    # sentinel-labeled internally though, so `resolve` + a label check
    # recovers the complete set instead of trusting the lossy scalar.
    function flat_clip!(id, n, d, sentA, sentB)
        n̂ = n / norm(n)
        p0 = (d / norm(n)) * n̂
        pA, pB = p0 - n̂, p0 + n̂
        clip_by_hyperplane!(cx, id, AffineQuadratic(pA), AffineQuadratic(pB), sentA, sentB)
        label_a, label_b = Label([sentA]), Label([sentB])
        results = resolve(cx, id)
        return [r for r in results if cx.nodes[r].label == label_a],
        [r for r in results if cx.nodes[r].label == label_b]
    end

    a1s, b1s = flat_clip!(start_id, n1, d1, Set([-1]), Set([-2]))
    for (pieces, l1_sign) in ((a1s, -1), (b1s, 1)), piece in pieces
        a2s, b2s = flat_clip!(piece, n2, d2, Set([-3]), Set([-4]))
        for (leaves, l2_sign) in ((a2s, -1), (b2s, 1)), leaf in leaves
            set_label!(cx, leaf, Label([l1_sign == l2_sign ? idxB : idxA]))
        end
    end
    fix_boundary_labels!(cx, 1)
    return nothing, nothing, nothing
end

"""
Insert a segment's own perpendicular endpoint cuts into every currently-live
top-dimensional cell -- pure refinement (see `clip_by_plane_preserving_label!`),
done *before* comparing the segment's features against existing winners so
that every remaining top-dimensional cell afterward sits entirely within
exactly one of the segment's three features' validity regions (mirrors the
2D prototype's `insert_own_lines!`, minus its third line -- the segment's
own supporting line, needed there only to keep the two sides of the
interior feature in separate cells for its side-atom coloring scheme; this
package's renderer computes side directly from the geometry instead, so
skipping it doesn't affect correctness, only leaves the interior's cell as
one strip spanning both sides).
"""
function insert_segment_own_lines!(cx::CellComplex{N}, pa::Pt{N,Float64}, pb::Pt{N,Float64}) where {N}
    t̂ = (pb - pa) / norm(pb - pa)
    for d in (dot(t̂, pa), dot(t̂, pb))
        for cell_id in top_cell_ids(cx)
            clip_by_plane_preserving_label!(cx, cell_id, t̂, d)
        end
    end
    return nothing
end

"""
Insert `new_feats`' own validity-region boundaries into every currently-live
top-dimensional cell -- pure refinement (see `clip_by_plane_preserving_label!`),
done *before* comparing against existing winners so that every remaining
top-dimensional cell afterward sits entirely within exactly one of the new
features' validity regions. A point's single feature has no validity
boundaries (`HalfSpace{N,Float64}[]`), so this is a no-op for a point --
exactly matching G1's behavior, where this step never existed at all.
Mirrors the 2D prototype's `insert_own_lines!`, minus its third line (the
segment's own supporting line, needed there only for its side-atom
coloring scheme; this package's renderer computes side directly from the
geometry instead, so skipping it doesn't affect correctness, only leaves
a segment's interior cell as one strip spanning both sides).

Different features of the *same* sub-simplex routinely share a boundary
plane (e.g. a segment's "beyond A" cut and its interior's own lower cut are
literally the same plane, just seen with an opposite-signed normal from
each feature's own perspective) -- inserting it twice would re-clip an
edge that's already exactly on it, an avoidable degenerate case, so each
geometrically distinct plane is only clipped in once (`seen`, keyed
canonically so `n` and `-n` versions of the same plane collide).
"""
function insert_own_lines!(cx::CellComplex{N}, new_feats::Vector{GFeature{N}}) where {N}
    seen = Set{Tuple{Pt{N,Float64},Float64}}()
    # `clip_by_plane_preserving_label!` forces `preserve_label` onto every
    # piece a refining cut touches, including a *pre-existing* vertex or
    # edge that happened to lie in the cut's path -- even when it already
    # correctly carried a genuine tie against some other, completely
    # unrelated cell (this refinement has no way to know about that cell; it
    # only knows the one label it's preserving). Confirmed as a real,
    # prevalent (not rare) gap via random-stress testing of mixed
    # point+segment input -- a large fraction of a construction's own
    # vertices, not just an occasional one. Every dim=0 descendant of every
    # cell touched here is collected (before touching it, so the ids are
    # valid regardless of what the clip itself does) and returned for the
    # caller to fold into its own broader recompute (`insert_features!`'s
    # own `touched_verts` -- this function doesn't have the `feats_so_far`
    # a proper recompute needs, only its caller does). Edges (dim=N-1) get
    # the analogous fix via `fix_boundary_labels!`'s own full rescan below.
    first_new_id = length(cx.nodes) + 1
    touched_verts = Int[]
    for f in new_feats, hs in f.validity
        n̂ = hs.n / norm(hs.n)
        dd = hs.d / norm(hs.n)
        i = argmax(abs.(n̂))
        key = n̂[i] < 0 ? (-n̂, -dd) : (n̂, dd)
        key in seen && continue
        push!(seen, key)
        for cell_id in top_cell_ids(cx)
            append!(touched_verts, descendant_nodes_by_dim(cx, cell_id)[1])
            clip_by_plane_preserving_label!(cx, cell_id, hs.n, hs.d)
        end
        # A segment's own two endpoint planes each independently sweep
        # every top cell -- the same "independently clip related
        # structure" pattern `weld_near_duplicate_vertices!` exists for,
        # just within this loop's own multi-plane sequence, before the
        # caller ever gets a chance to weld. Confirmed reachable: two of a
        # face's own boundary vertices can become near-duplicates from the
        # two different endpoint-plane cuts, and the *next* plane's own
        # face-frame lookup has no third distinct vertex once that pair is
        # dropped -- a crash before this function even returns. The label
        # here is only provisional; the real one is recomputed later by
        # `insert_features!`'s own broader weld and `fix_boundary_labels!`.
        weld_near_duplicate_vertices!(cx, first_new_id, pt -> recompute_feature_label(pt, new_feats); extra_verts=touched_verts)
        weld_duplicate_edges!(cx, first_new_id)
    end
    # Full rescan (not scoped to new nodes only) since the corrupted node
    # is often an *old* one, not one created by this call.
    fix_boundary_labels!(cx, 1)
    return touched_verts
end

"""
The true tied-winner set at `pt` against feature list `feats`, by direct
distance comparison -- segments counterpart to `recompute_point_label`
(multi_points.jl), used the same way by `weld_near_duplicate_vertices!`.
"""
function recompute_feature_label(pt::Pt{N,Float64}, feats::Vector{GFeature{N}}; atol=1e-9) where {N}
    valid = [f for f in feats if is_valid(f.validity, pt)]
    isempty(valid) && return Label()
    ds = [sqdist(f.quad, pt) for f in valid]
    m = minimum(ds)
    return Label([f.face for (f, d) in zip(valid, ds) if d <= m + atol * max(1.0, m)])
end

"""
Incorporate a new sub-simplex's features (`new_feats`) into a complex
that already correctly reflects every previous one (`feats_so_far`,
appended in place): first the new features' own validity boundaries, then
for each live top cell, find which new feature is locally active there,
compare against the existing winner, and clip if it wins. Mirrors
`insert_point!`'s snapshot-then-process structure. Works identically
whether `new_feats`/`feats_so_far` are points, segments, or a mix --
nothing here is type-specific.
"""
function insert_features!(cx::CellComplex{N}, feats_so_far::Vector{GFeature{N}}, new_feats::Vector{GFeature{N}}) where {N}
    # Captured before `insert_own_lines!` so the later weld/fix pass's
    # scope covers every vertex this whole call creates -- including
    # `insert_own_lines!`'s own endpoint-plane cuts, which can land at
    # essentially the same point as a vertex the main loop creates
    # independently later.
    first_new_id = length(cx.nodes) + 1
    touched_verts = insert_own_lines!(cx, new_feats)

    # `insert_own_lines!`'s own internal validity-cut planes are each
    # applied across every top cell independently -- the same
    # "independently clip related structure" pattern the final weld below
    # exists for, just within this function's own two-plane sequence.
    # Confirmed reachable: two boundary vertices can become near-
    # duplicates from the two endpoint-plane cuts, and a later face lookup
    # (before the final weld runs) has no third distinct vertex once that
    # pair is dropped. An early weld here closes that gap.
    weld_near_duplicate_vertices!(cx, first_new_id, pt -> recompute_feature_label(pt, feats_so_far); extra_verts=touched_verts)
    weld_duplicate_edges!(cx, first_new_id)

    to_process = Tuple{Int,GFeature{N},GFeature{N}}[]
    for cell_id in top_cell_ids(cx)
        node = cx.nodes[cell_id]
        isempty(node.label) && continue
        winner_face = only(node.label)
        sample = interior_sample(cx, cell_id)
        new_feat = first(f for f in new_feats if is_valid(f.validity, sample))
        cur_feat = first(f for f in feats_so_far if f.face == winner_face)
        push!(to_process, (cell_id, cur_feat, new_feat))
    end

    # Every vertex touched below can end up stale unless recomputed
    # afterward -- collected before processing each cell so the ids are
    # captured regardless of what that cell's clip does.
    #
    # Different top cells sharing a boundary edge, both against the same
    # `cur_feat`, get compared against the exact same `quad` --
    # `global_edge_cache` lets every cell sharing that comparison share one
    # edge-resolution cache, so the first cell to resolve a shared edge
    # commits it for every other cell touching the same edge id (the
    # dominant real cause of "edges disagree on side" failures -- see
    # `clip_by_hyperplane!`'s own docstring). Keyed by the `(cur_feat,
    # new_feat)` pairing so cells with different current winners share
    # nothing.
    edge_caches = Dict{Tuple{Set{VertexIdx},Set{VertexIdx}},Dict{Int,Vector{Tuple{Bool,Int,Int,Int}}}}()
    for (cell_id, cur_feat, new_feat) in to_process
        append!(touched_verts, descendant_nodes_by_dim(cx, cell_id)[1])
        quad = bisector(cur_feat.quad, new_feat.quad)
        # Not just a vertex-sign check: a curved bisector can dip into a
        # single edge's interior while every vertex still agrees --
        # `cell_uniformly_signed` catches that.
        side = cell_uniformly_signed(cx, cell_id, quad, p -> exact_sign((q, x) -> evaluate(q, x), quad, p))
        if side == :a
            continue   # the new sub-simplex never wins here
        elseif side == :b
            set_label!(cx, cell_id, Label([new_feat.face]))
        else
            key = (cur_feat.face, new_feat.face)
            shared_cache = get!(() -> Dict{Int,Vector{Tuple{Bool,Int,Int,Int}}}(), edge_caches, key)
            clip_by_hyperplane!(cx, cell_id, cur_feat.quad, new_feat.quad, cur_feat.face, new_feat.face; global_edge_cache=shared_cache)
        end
    end
    append!(feats_so_far, new_feats)
    # Independently clipping each cell above can leave two near-duplicate
    # vertices at a genuine multi-way tie instead of one. Weld those,
    # recompute touched pre-existing vertices' true labels too, then fix
    # any boundary whose label is now stale.
    weld_near_duplicate_vertices!(cx, first_new_id, pt -> recompute_feature_label(pt, feats_so_far); extra_verts=touched_verts)
    weld_duplicate_edges!(cx, first_new_id)
    fix_boundary_labels!(cx, first_new_id)
    return nothing
end

insert_segment!(cx::CellComplex{N}, feats_so_far::Vector{GFeature{N}}, pa::Pt{N,Float64}, pb::Pt{N,Float64}, idxa::VertexIdx, idxb::VertexIdx) where {N} =
    insert_features!(cx, feats_so_far, segment_features(pa, pb, idxa, idxb))

"""
The simplest G2 construction: one point and one segment. Builds a padded
bounding box, labels it entirely by the point (as in G1's base case), then
inserts the segment via `insert_segment!`. Returns `(cx, feats)`.
"""
function point_segment_complex(p::Pt{N,Float64}, pa::Pt{N,Float64}, pb::Pt{N,Float64}) where {N}
    lo, hi = padded_bbox([p, pa, pb])
    cx, top = init_bbox_complex(Val(N), lo, hi)
    pf = point_feature(p, 1)
    label1 = Label([pf.face])
    for id in eachindex(cx.nodes)
        set_label!(cx, id, label1)
    end
    feats = GFeature{N}[pf]
    insert_segment!(cx, feats, pa, pb, 2, 3)
    return cx, feats
end

"""
Incorporate one sub-simplex's features into `cx`, whether it's the very
first (`feats` empty) or one more addition to an already-correct complex
(appends to `feats` in place).

The first entry is special: no existing winner to compare against, so its
region boundaries are established purely by sampling (`insert_own_lines!`
+ `interior_sample`/`is_valid`) rather than a real
`clip_by_hyperplane!` comparison; `label_boundaries!` then records the
genuine tie where two of its own regions meet, which would otherwise be
left unlabeled. Every subsequent entry goes through `insert_features!`
directly. Shared by `multi_complex` and any interactive, one-at-a-time
caller.
"""
# N-generic no-op: the dangling-spike/degenerate-cell situation this
# guards against is 2D-specific (see the docstring on the N=2 method
# below), and G1 (points-only, any N) never produces a curved or
# multiply-connected boundary in the first place.
retire_degenerate_cells!(cx::CellComplex{N}) where {N} = nothing

"""
Retires any live top cell with no genuine closed boundary loop at all --
a cell with no real enclosed area, which can be left behind by
`clip_top_cell_2d!`'s own hole-placement fallback (an open, not-yet-fully-
solved gap). Run at the end of every `insert_entry!` call so a degenerate
leftover never reaches `interior_sample` in a later round.
"""
function retire_degenerate_cells!(cx::CellComplex{2})
    for id in top_cell_ids(cx)
        loops = cyclic_boundary_walks(cx, cx.nodes[id].subcells)
        if find_outer_loop(cx, loops) === nothing
            @warn "retire_degenerate_cells!: cell $id has no genuine closed boundary loop (every edge is part of a dangling spike) -- retiring it with no replacement"
            supersede!(cx, id, Int[])
        end
    end
    return nothing
end

function insert_entry!(cx::CellComplex{N}, feats::Vector{GFeature{N}}, new_feats::Vector{GFeature{N}}) where {N}
    if isempty(feats)
        # Scoped from before `insert_own_lines!` for the same reason
        # `insert_features!` does: a first entry's own validity-boundary
        # cuts can leave near-duplicate vertices too.
        first_new_id = length(cx.nodes) + 1
        touched_verts = insert_own_lines!(cx, new_feats)
        for cell_id in top_cell_ids(cx)
            sample = interior_sample(cx, cell_id)
            # `first`, not `only`: a sample exactly on the shared boundary
            # between two of this entry's own regions is valid for both
            # (their formulas agree there by construction), so either is fine.
            f = first(ff for ff in new_feats if is_valid(ff.validity, sample))
            set_label!(cx, cell_id, Label([f.face]))
        end
        weld_near_duplicate_vertices!(cx, first_new_id, pt -> recompute_feature_label(pt, new_feats); extra_verts=touched_verts)
        weld_duplicate_edges!(cx, first_new_id)
        label_boundaries!(cx)
        append!(feats, new_feats)
    else
        insert_features!(cx, feats, new_feats)
    end
    # Invariant: no two live, adjacent top cells share a label past one call.
    merge_adjacent_same_label_cells!(cx)
    retire_degenerate_cells!(cx)
    return nothing
end

"""
The features of one `multi_complex`/interactive-demo entry -- `(:point, p,
idx)`, `(:segment, pa, pb, idxa, idxb)`, or `(:line, pa, pb, idxa, idxb)`
(the same shape as `:segment`, but the unbounded affine line through the
two points instead of the bounded segment between them -- see
`line_features`) -- as a `Vector{GFeature{N}}`, ready for `insert_entry!`.
"""
function entry_feats(e, ::Val{N}) where {N}
    e[1] === :point && return GFeature{N}[point_feature(e[2], e[3])]
    e[1] === :line && return line_features(e[2], e[3], e[4], e[5])
    return segment_features(e[2], e[3], e[4], e[5])
end

"""
General incremental multi-sub-simplex construction, mixing points,
segments, and unbounded lines freely: `entries` is a list of `(:point, p,
idx)`, `(:segment, pa, pb, idxa, idxb)`, or `(:line, pa, pb, idxa, idxb)`
tuples, processed in order via `insert_entry!`. Returns `(cx, feats)`.
"""
function multi_complex(entries::Vector, ::Val{N}; pad=0.3) where {N}
    allpts = Pt{N,Float64}[]
    for e in entries
        e[1] === :point ? push!(allpts, e[2]) : push!(allpts, e[2], e[3])
    end
    lo, hi = padded_bbox(allpts; pad=pad)
    cx, top = init_bbox_complex(Val(N), lo, hi)

    feats = GFeature{N}[]
    for e in entries
        insert_entry!(cx, feats, entry_feats(e, Val(N)))
    end
    return cx, feats
end

"""Brute-force oracle matching `multi_complex`'s `entries` format, generalizing `brute_force_label_segments` to a mix of points, segments, and unbounded lines."""
function brute_force_label_multi(entries::Vector, x::Pt{N,Float64}; atol=1e-9) where {N}
    ds = Dict{Set{Int},Float64}()
    for e in entries
        if e[1] === :point
            _, p, idx = e
            ds[Set([idx])] = sum(abs2, x - p)
        elseif e[1] === :line
            _, pa, pb, idxa, idxb = e
            t̂ = (pb - pa) / norm(pb - pa)
            # Squared distance from `x` to the *unbounded* line through
            # `pa`,`pb` -- |x-pa|^2 minus the squared component along t̂,
            # i.e. the squared perpendicular residual. Deliberately not
            # `n̂ = SVector(-t̂[2],t̂[1])` (the `:segment` interior branch's
            # own shortcut below): that construction only spans the
            # orthogonal complement of a line in 2D, whereas this formula
            # is `N`-generic (matters once a 3D line oracle is needed).
            ds[Set([idxa, idxb])] = sum(abs2, x - pa) - dot(t̂, x - pa)^2
        else
            _, pa, pb, idxa, idxb = e
            t̂ = (pb - pa) / norm(pb - pa)
            ta, tb = dot(t̂, pa), dot(t̂, pb)
            tx = dot(t̂, x)
            if tx <= ta
                ds[Set([idxa])] = sum(abs2, x - pa)
            elseif tx >= tb
                ds[Set([idxb])] = sum(abs2, x - pb)
            else
                n̂ = SVector(-t̂[2], t̂[1])
                ds[Set([idxa, idxb])] = dot(n̂, x - pa)^2
            end
        end
    end
    m = minimum(values(ds))
    return Set(face for (face, dd) in ds if dd <= m + atol * max(1.0, m))
end

"""Brute-force oracle for the point-vs-segment case, generalizing `brute_force_label_points`."""
function brute_force_label_segments(p::Pt{N,Float64}, pa::Pt{N,Float64}, pb::Pt{N,Float64}, x::Pt{N,Float64}; atol=1e-9) where {N}
    t̂ = (pb - pa) / norm(pb - pa)
    ta, tb = dot(t̂, pa), dot(t̂, pb)
    tx = dot(t̂, x)
    ds = Dict{Set{Int},Float64}()
    ds[Set([1])] = sum(abs2, x - p)
    if tx <= ta
        ds[Set([2])] = sum(abs2, x - pa)
    elseif tx >= tb
        ds[Set([3])] = sum(abs2, x - pb)
    else
        n̂ = SVector(-t̂[2], t̂[1])
        ds[Set([2, 3])] = dot(n̂, x - pa)^2
    end
    m = minimum(values(ds))
    return Set(face for (face, dd) in ds if dd <= m + atol * max(1.0, m))
end
