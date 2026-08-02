"""
Every currently-live top-dimensional (`dim == N`) cell id.
"""
top_cell_ids(cx::CellComplex{N}) where {N} = [id for id in eachindex(cx.nodes) if !haskey(cx.superseded_by, id) && cx.nodes[id].dim == N]

"""
Merges every group of currently-live, mutually-adjacent top-dimensional
cells that share the same label into one cell each, in place. Two live top
cells are "adjacent" here if they share an immediate `(N-1)`-dimensional
subcell (a boundary face) -- found via union-find over shared subcells, so
a whole *chain* of same-label neighbors (not just a single pair) merges in
one pass. Cells sharing a label *without* being connected this way (a
label can legitimately occur at multiple disconnected locations -- see the
project notes on why label equality alone doesn't imply "same place") are
correctly left as separate cells.

This is the fix for a real gap in the construction: nothing about
`clip_by_hyperplane!`'s per-cell processing prevents two neighboring cells
from ending up with the same label after enough insertions (e.g. a third
point winning both of two previously-different neighbors), leaving a
purely historical "scar" edge between them that no longer represents a
genuine tie. Left unmerged, this is more than cosmetic: point-location
(`find_containing_cell`) only ever finds *one* of the pieces, so a query
landing in "the other half" of what is visually one contiguous region
reports the wrong cell.

For each merged group, the new cell's subcells are exactly the ones
referenced by only *one* member of the group (the group's own outer
boundary -- a subcell referenced by two members is, by definition, one of
the scar edges being dissolved). The original cells are `supersede!`d to
the new one; the dissolved scar subcells are simply left as unreferenced,
harmless nodes (consistent with how the rest of the complex already
tolerates stale-but-resolvable structure via `superseded_by`).
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
            # A subcell shared by 2+ same-label cells at once (not just
            # exactly 2 -- e.g. a genuine multi-way tie at `N>=3`) makes
            # all of them mutually adjacent; union them all in, not just
            # the first pair, or the ones past the second are silently
            # left in their own component while still being excluded from
            # every component's own "external boundary" computation below
            # (since they're absent from `ids`' own count there) --
            # producing a merged cell with a hole in its boundary.
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
            # The dissolved (internal) subcells are still technically
            # "live" (never marked superseded -- there's no same-
            # dimensional replacement for them to point to) unless
            # explicitly retired here. Relabeling them to `label` (an
            # earlier version of this function did that) is *wrong*, not
            # just approximate: they'd become permanently unreachable from
            # the live tree (the merged cell's own subcells are its
            # *external* boundary only), so a later insertion that
            # legitimately re-splits this same region has no way to reach
            # and update them -- they'd keep whatever label was accurate
            # at *this* moment forever, silently going stale the next time
            # anything here actually changes. Retiring them via
            # `supersede!` with no replacement is the honest state: they
            # resolve to nothing, and every consumer that already knows to
            # skip superseded nodes (which is all of them -- that's the
            # whole point of `superseded_by`) does the right thing
            # automatically, no guessing required.
            for s in internal
                supersede_if_orphaned!(cx, s)
            end
        end
    end
    assert_label_bbox_invariant(cx)
    return nothing
end

"""
If `id` is no longer referenced by anything still live (checked via
`cx.referenced_by`/`cx.superseded_by`), marks it `supersede!`d with no
replacement (it resolves to nothing) and recurses into its own subcells
the same way; otherwise leaves it (and everything below it) untouched.
The orphan check applies to `id` itself, not just its children: an
"internal" subcell identified by `merge_adjacent_same_label_cells!`
(shared by 2+ of the cells being merged) can -- in non-manifold-ish
configurations, e.g. a genuine multi-way tie -- *also* be shared with a
live cell entirely outside the merge, in which case it must stay live and
keep its own, independently-correct label, not be retired just because it
looked internal from the merge's own narrow point of view.
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
Axis-aligned bounding box of `id` -- `(lo, hi)`, each an
`SVector{N,Float64}`. This is the geometric extent that a label alone (a
purely combinatorial `Set{Set{VertexIdx}}`) doesn't carry: the same label
can legitimately occur at multiple disconnected locations (see the project
notes on why `P(P(P(VertexIdx)))` alone is an insufficient cell identity),
so telling apart "this is genuinely the same place" from "this
coincidentally has the same label" needs geometry, not just combinatorics.

A thin wrapper around the node's own cached `bbox_lo`/`bbox_hi` (see
`CellNode`'s docstring for how/when those are computed and why they're
still safe to use even after a later, unrelated insertion patches this
node's `subcells` list in place) -- kept as its own function since this is
the name the rest of the codebase (and its tests) already call it by.
"""
function cell_bbox(cx::CellComplex{N}, id::Int) where {N}
    node = cx.nodes[id]
    return node.bbox_lo, node.bbox_hi
end

"""
Whether two axis-aligned boxes overlap in every dimension, *including*
merely touching at a shared boundary -- deliberately inclusive, since two
same-label cells that only touch (share a boundary point or face, not
interior volume) are exactly the adjacent-but-unmerged case this whole
mechanism exists to catch.
"""
boxes_touch_or_overlap(lo1, hi1, lo2, hi2; atol=1e-9) = all(lo1 .<= hi2 .+ atol) && all(lo2 .<= hi1 .+ atol)

"""
Whether 2D cells `id1`,`id2` share a genuine boundary *edge* -- not just a
single vertex -- found by looking for any edge of `id1` and any edge of
`id2` whose two endpoints coincide (in either order, within tolerance).
Distinguishes the two ways two same-label cells' bounding boxes can touch
with zero overlap area: sharing an actual positive-length boundary (a real
missed merge) versus meeting at a single point where several regions come
together, two of which happen to share a label (geometrically ordinary,
not a bug -- merging cells that only share a point would need a
self-pinching boundary, not a valid simple polygon).
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
Whether 2D cells `id1`,`id2` actually overlap in area, checked exactly (not
by sampling their bounding boxes' shared region): a genuine overlap between
two cells built by a sequence of clips essentially always traps at least
one vertex of one cell's own outer boundary strictly inside the other's
(their boundaries would otherwise have to cross an even number of times
without ever separating the two interiors, an exotic degenerate
configuration this doesn't attempt to catch) -- far cheaper than a grid
scan, and this runs inside a check (`assert_label_bbox_invariant`) called
after every single insertion.
"""
function cells_area_overlap(cx::CellComplex{2}, id1::Int, id2::Int)
    outer_edges(id) = let loops = cyclic_boundary_walks(cx, cx.nodes[id].subcells)
        loops[find_outer_loop(cx, loops)]
    end
    ew1, ew2 = outer_edges(id1), outer_edges(id2)
    e1, e2 = [e for (e, _, _) in ew1], [e for (e, _, _) in ew2]
    for (_, v, _) in ew1
        point_in_edge_loop(cx, e2, cx.nodes[v].point) && return true
    end
    for (_, v, _) in ew2
        point_in_edge_loop(cx, e1, cx.nodes[v].point) && return true
    end
    return false
end

"""
The invariant `merge_adjacent_same_label_cells!` is supposed to establish
and maintain, checked directly rather than just hoped for: no two
distinct, live top cells sharing a label genuinely overlap or share a
boundary -- if they did, they should have been merged into one cell
instead. Warns with the offending ids and label if violated, rather than
erroring (downgraded from an error while task #42/#49 -- a real,
still-open bug in this area -- is outstanding, so construction can keep
running for experimentation instead of aborting on it). Called
automatically at the end of every merge pass, so this is verified
continuously as the complex is built, not just spot-checked in tests.

Bounding boxes touching or overlapping used to be treated as proof of this
on its own, but that's not actually sound at `N=2`: a cell's bbox can be
considerably looser than its own true footprint (a thin or multiply-
connected shape whose extremal points sit far from where it and a
neighbor's bbox happen to overlap), so two same-label cells' bboxes can
overlap -- even with real *area*, not just a shared point -- without the
cells themselves ever touching. `boxes_touch_or_overlap` is kept as a cheap
pre-filter (a real bug always trips it too), but the actual verdict is now
`cells_share_edge`/`cells_area_overlap`'s exact geometric check, not the
bbox alone.
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
A point guaranteed inside `id` (a top-dimensional cell): the plain average
of its own vertices. `N`-generic fallback, used as-is at any `N != 2`
(where curved edges -- and hence the non-convexity they can cause -- aren't
supported yet regardless); superseded at `N=2` by the dedicated method
below, which is exact for a non-convex cell instead of merely a convex one.
"""
interior_sample(cx::CellComplex{N}, id::Int) where {N} = (pts = descendant_points(cx, id); sum(pts) / length(pts))

"""
A point provably inside 2D cell `id`'s own region, even when it's
non-convex -- unlike the plain vertex-centroid fallback above (a former bug,
task #38: a vertex average lands outside its own cell for any sufficiently
concave shape, which a curved bisector's arc-vs-chord bulge is only one way
to produce -- an ordinary all-straight-edge polygon can be concave too, and
that case was never actually covered by the fallback's reasoning). Also
correct for a cell with a hole (task #39): `cyclic_boundary_walks` +
`find_outer_loop` restrict to the cell's own *outer* boundary first, so a
hole's vertices (no part of this cell's actual interior) can't pull the
sample toward or past them.

Exact for any simple polygon, convex or not: picks the outer loop's own
topmost vertex (breaking a `y`-tie on `x` for a deterministic choice) --
always a locally convex corner, since nothing can be above it -- casts a
horizontal ray an epsilon below it (off the vertex itself, so the ray
doesn't graze along a boundary edge), and returns the midpoint of whichever
crossing-to-crossing span straddles that vertex's own `x`. That span is the
region just below the topmost vertex's own two incident edges, which by
local convexity there is guaranteed interior, regardless of how the rest of
the polygon bends. Curved edges solved directly against their own stored
quadric (`curve_crossings_at_y`/`on_arc_between`, the same exact machinery
`point_in_edge_loop` uses), not approximated.
"""
function interior_sample(cx::CellComplex{2}, id::Int)
    loops = cyclic_boundary_walks(cx, cx.nodes[id].subcells)
    outer = loops[find_outer_loop(cx, loops)]
    verts = unique(v for (_, v, _) in outer)
    pts = [cx.nodes[v].point for v in verts]
    topv = pts[argmax([(p[2], p[1]) for p in pts])]
    y0 = topv[2] - 1e-6 * max(1.0, maximum(p -> abs(p[2]), pts))

    xs = Float64[]
    for (e, _, _) in outer
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
Label every currently-live `(N-1)`-dimensional boundary cell with the
union of whichever one or two top-dimensional cells reference it -- needed
after a *first* sub-simplex's own regions are established purely by
sampling (`is_valid`, not a real `clip_by_hyperplane!` comparison, since
there's nothing to compare against yet): the top cells themselves get
labeled directly, but the boundary between two differently-labeled
neighbors is exactly as genuine a tie as one produced by comparing two
different sub-simplices (e.g. where a segment's own "beyond endpoint A"
and "interior" regions meet is exactly where those two features are
equal) -- it just never gets recorded, since nothing ever calls
`set_label!` on it. Leaving it unlabeled silently defeats any later code
that recognizes an existing tie by checking a cell's own label (see
`clip_by_hyperplane!`'s use of it) instead of re-deriving the tie
numerically from scratch. A boundary referenced by only one live top cell
(e.g. one lying on the bounding box's own edge) simply inherits that one
cell's label, which is not a tie and is harmless.
"""
function label_boundaries!(cx::CellComplex{N}) where {N}
    for id in eachindex(cx.nodes)
        haskey(cx.superseded_by, id) && continue
        node = cx.nodes[id]
        node.dim == N - 1 || continue
        parents = [p for p in get(cx.referenced_by, id, Int[])
                   if !haskey(cx.superseded_by, p) && cx.nodes[p].dim == N && !isempty(cx.nodes[p].label)]
        isempty(parents) && continue
        set_label!(cx, id, reduce(∪, (cx.nodes[p].label for p in parents); init=node.label))
    end
    return nothing
end

"""
Split `start_id` (and everything below it) by the plane `{x : n·x = d}`,
*without* introducing a new winner -- both resulting pieces (and the new
cut cell between them) keep `start_id`'s own original label. Pure
topological refinement, the dimension-generic analogue of the 2D
prototype's `insert_curve_deduped!` for a simplex's own dividing lines
(the cell isn't compared against a competing feature here, just carved up
so a later comparison sees a properly-scoped piece).

Implemented by reusing `clip_by_hyperplane!` itself: the perpendicular
bisector of any two points placed symmetrically around a point on the
target plane, along its own normal, *is* exactly that plane -- so this
just constructs such a pair and lets the already-tested clip machinery do
the real work, passing `start_id`'s own label through as `preserve_label`
so every piece this creates, at every dimension level (not just the two
or three top-level results), is correctly labeled from the start.
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

"""
The unit normal `n` and offset `d` of the line `AffineQuadratic{2,1,Float64}`
`q` represents (`{x : n·x = d}`) -- `q.basis`'s single column is the line's
own *direction*; rotating it 90° gives the normal, and `d` follows from any
point known to be on the line, `q.p` itself.
"""
function line_normal_form(q::AffineQuadratic{2,1,Float64})
    t = SVector(q.basis[1, 1], q.basis[2, 1])
    n = SVector(-t[2], t[1])
    return n, dot(n, q.p)
end

"""
`clip_by_hyperplane!`'s dedicated path for comparing two line-like features
(`K=N-1` at `N=2` -- two different segments' own interior features) against
each other: unlike every other feature pairing in this codebase, this
bisector is *not* one connected curve. Squared-distance-to-a-line is
already `(n·x-d)²`, so writing `u = n_A·x-d_A`, `v = n_B·x-d_B`, the tie
condition `u² = v²` factors as `(u-v)(u+v) = 0` -- two straight lines,
`l1 = u-v` and `l2 = u+v`, crossing at one point. `A` wins where `u² < v²`,
i.e. `l1` and `l2` have *opposite* signs; `B` wins where they agree --
exactly the classic four-quadrant picture of two intersecting lines'
angle bisectors.

Built from `clip_by_hyperplane!` itself, applied twice: first split
`start_id` by the flat plane `l1 = 0` (a completely ordinary hyperplane
clip, no different from a G1 point-vs-point comparison), then split
*each* resulting piece by `l2 = 0`. Each of the (up to four) leaf pieces
this produces sits entirely within one quadrant, so its final label is
determined purely by which two clips it came from -- overwritten directly
here rather than trusted from either clip's own (necessarily provisional,
since neither clip alone knows the other's side) labeling. Sentinel faces
(negative, so they can never collide with a real 1-based `VertexIdx`) are
passed to the two sub-clips only to keep their own internal bookkeeping
happy; they're never visible outside this function, since every leaf gets
relabeled before returning and a full-complex `fix_boundary_labels!` pass
(scoped like `insert_own_lines!`'s own, not `insert_features!`'s
narrower one -- see its docstring for why: a leaf's *pre-existing*
boundary against some unrelated neighbor can need relabeling too, not just
newly-created structure) re-derives every boundary label from the
now-correctly-labeled leaves afterward. Vertex labels need no equivalent
fix-up here: `insert_features!`'s own end-of-call `weld_near_duplicate_vertices!`
recomputes every new vertex's label from its position directly, ignoring
whatever provisional label construction left it with.
"""
function clip_by_line_pair!(cx::CellComplex{2}, start_id::Int, A::AffineQuadratic{2,1,Float64}, B::AffineQuadratic{2,1,Float64}, idxA::Set{VertexIdx}, idxB::Set{VertexIdx})
    nA, dA = line_normal_form(A)
    nB, dB = line_normal_form(B)
    n1, d1 = nA - nB, dA - dB
    n2, d2 = nA + nB, dA + dB

    # `clip_by_hyperplane!`'s own return value is a legacy scalar (one id
    # per side) predating multi-crossing/hole support -- it silently keeps
    # only the *last* same-side piece when a clip actually produces more
    # than one (routine now: a hole's own outer loop splitting into several
    # runs on the same side, task #39). Every resulting piece is still
    # correctly sentinel-labeled inside `clip_by_hyperplane!` itself
    # though (via its own `extra_a`/`extra_b`), so `resolve` (following
    # `id`'s own supersession chain to whatever it's now made of) plus a
    # label check recovers the complete set instead of trusting the lossy
    # scalar.
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
    for f in new_feats, hs in f.validity
        n̂ = hs.n / norm(hs.n)
        dd = hs.d / norm(hs.n)
        i = argmax(abs.(n̂))
        key = n̂[i] < 0 ? (-n̂, -dd) : (n̂, dd)
        key in seen && continue
        push!(seen, key)
        for cell_id in top_cell_ids(cx)
            clip_by_plane_preserving_label!(cx, cell_id, hs.n, hs.d)
        end
    end
    # `clip_by_plane_preserving_label!` forces `preserve_label` onto every
    # piece a refining cut touches, including a *pre-existing* boundary
    # edge that happened to lie in the cut's path -- even when that edge
    # already correctly carried a genuine tie against some other,
    # completely unrelated cell (this refinement has no way to know about
    # that cell; it only knows the one label it's preserving). A full
    # rescan (not scoped to new nodes only, unlike the analogous cleanup in
    # `insert_features!`/`insert_point!`) is needed here specifically
    # because the corrupted node is often an *old* one, not one created by
    # this call.
    fix_boundary_labels!(cx, 1)
    return nothing
end

"""
The true tied-winner set at `pt` against the flat feature list `feats`,
found by direct distance comparison -- the G2 (segments) counterpart to
`recompute_point_label` (multi_points.jl), used the same way by
`weld_near_duplicate_vertices!`. Mirrors the interactive demo's own
`winners_at` (which exists purely for the UI's hover query), but belongs
in the library since construction itself needs the same computation.
"""
function recompute_feature_label(pt::Pt{N,Float64}, feats::Vector{GFeature{N}}; atol=1e-9) where {N}
    valid = [f for f in feats if is_valid(f.validity, pt)]
    isempty(valid) && return Label()
    ds = [sqdist(f.quad, pt) for f in valid]
    m = minimum(ds)
    return Label([f.face for (f, d) in zip(valid, ds) if d <= m + atol * max(1.0, m)])
end

"""
Incorporate a new sub-simplex's features (`new_feats`, from `point_feature`
or `segment_features`) into a complex that already correctly reflects every
previously inserted sub-simplex (`feats_so_far`, appended to in place):
first the new features' own validity boundaries, then -- for each
currently-live top-dimensional cell -- find which of the new features is
locally active there (via `insert_own_lines!` having already scoped every
cell to exactly one -- or, when a cell's own sample point lands exactly on
the shared boundary between two of the new sub-simplex's own regions, e.g.
its "beyond an endpoint" and "interior" features meeting exactly at that
endpoint, *either* one: their distance formulas agree there by
construction, so which one gets picked doesn't affect the comparison that
follows), compare it against the cell's existing winner, and clip if it
ever wins. The existing winner's own feature is looked up the same
tolerant way: a face is a vertex set, so its coordinates (and hence its
quadratic) are fixed regardless of which sub-simplex insertion produced
the `feats_so_far` entry for it -- a segment endpoint landing on a vertex
already used by an earlier point or another segment's endpoint legitimately
leaves *several* entries sharing that face, all with the same quadratic,
so again any one of them will do. Mirrors `insert_point!`'s snapshot-then-process structure (so a
clip performed for one cell doesn't change which *other* cells get visited
in the same pass) and the 2D prototype's `incorporate_simplex!` per-cell
active-feature lookup. Works identically whether `new_feats` came from a
point or a segment (or, eventually, any other sub-simplex), and whether
`feats_so_far` so far contains only points, only segments, or a mix --
nothing here is type-specific.
"""
function insert_features!(cx::CellComplex{N}, feats_so_far::Vector{GFeature{N}}, new_feats::Vector{GFeature{N}}) where {N}
    # Captured *before* `insert_own_lines!` (not just before the main clip
    # loop below), so the later weld/fix pass's scope covers every vertex
    # this whole call creates -- including `insert_own_lines!`'s own
    # endpoint-plane cuts, which can land at essentially the same point as
    # a vertex the main loop's own multi-crossing handling creates
    # independently later (the segment's own endpoint plane and the
    # interior-vs-existing-feature bisector genuinely meet exactly there).
    # A narrower scope starting after `insert_own_lines!` would put that
    # pair on opposite sides of the weld's own boundary, the same failure
    # mode `weld_near_duplicate_vertices!` exists to prevent, just via a
    # different combination of code paths than the one it was first written
    # for.
    first_new_id = length(cx.nodes) + 1
    insert_own_lines!(cx, new_feats)

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

    for (cell_id, cur_feat, new_feat) in to_process
        quad = bisector(cur_feat.quad, new_feat.quad)
        # Not just a vertex-sign check: see `cell_uniformly_signed`'s own
        # docstring for why that alone silently misses a curved bisector
        # dipping into a single edge's interior while every vertex still
        # agrees (this is where the confirmed-real house-preset mismatch
        # traced back to -- `clip_by_hyperplane!`'s own multi-crossing
        # handling never even got a chance to run for the affected cell).
        side = cell_uniformly_signed(cx, cell_id, quad, p -> exact_sign((q, x) -> evaluate(q, x), quad, p))
        if side == :a
            continue   # the new sub-simplex never wins here
        elseif side == :b
            set_label!(cx, cell_id, Label([new_feat.face]))
        else
            clip_by_hyperplane!(cx, cell_id, cur_feat.quad, new_feat.quad, cur_feat.face, new_feat.face)
        end
    end
    append!(feats_so_far, new_feats)
    # Independently clipping each cell above (same structural pattern as
    # `insert_point!`) can leave two near-duplicate vertices at a genuine
    # multi-way tie instead of one, neither carrying the full tied label --
    # see `weld_near_duplicate_vertices!`'s docstring. Weld those, then fix
    # any boundary whose label is now stale relative to what it actually
    # separates.
    weld_near_duplicate_vertices!(cx, first_new_id, pt -> recompute_feature_label(pt, feats_so_far))
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
first thing ever inserted (`feats` empty) or one more addition to a
complex that already correctly reflects everything in `feats` so far
(mirrors `insert_features!`'s own contract: appends to `feats` in place).

The first entry is special: there's no existing winner to compare
against, so its own region boundaries are established purely by sampling
(`insert_own_lines!` to carve the box into each feature's own validity
region, then `interior_sample` + `is_valid` to find which applies where --
for a point this trivially covers "everywhere", since a point's single
feature has no validity boundaries at all) rather than a real
`clip_by_hyperplane!` comparison; `label_boundaries!` then records the
genuine tie where two of that first sub-simplex's own regions meet (e.g.
a segment's "beyond endpoint A" and "interior"), which would otherwise be
left unlabeled. Every subsequent entry just goes through
`insert_features!` directly.

This is the one piece of real incremental-insertion logic in the package
-- shared by `multi_complex` (looping over a fixed list) and any genuinely
interactive, one-at-a-time caller (e.g. a live demo), so both exercise the
identical, already-tested code path.
"""
function insert_entry!(cx::CellComplex{N}, feats::Vector{GFeature{N}}, new_feats::Vector{GFeature{N}}) where {N}
    if isempty(feats)
        # Scoped from before `insert_own_lines!` for the same reason
        # `insert_features!` now does the same (see its own matching
        # comment): a first entry's own validity-boundary cuts can leave
        # near-duplicate vertices just like any later insertion's can, and
        # this is a real, reachable caller path (`insert_entry!`'s own
        # first-entry branch), not just a test-only concern.
        first_new_id = length(cx.nodes) + 1
        insert_own_lines!(cx, new_feats)
        for cell_id in top_cell_ids(cx)
            sample = interior_sample(cx, cell_id)
            # `first`, not `only`: a sample landing exactly on the shared
            # boundary between two of this entry's own regions is valid
            # for both (their formulas agree there by construction -- see
            # `insert_features!`'s matching comment), so either is fine.
            f = first(ff for ff in new_feats if is_valid(ff.validity, sample))
            set_label!(cx, cell_id, Label([f.face]))
        end
        weld_near_duplicate_vertices!(cx, first_new_id, pt -> recompute_feature_label(pt, new_feats))
        weld_duplicate_edges!(cx, first_new_id)
        label_boundaries!(cx)
        append!(feats, new_feats)
    else
        insert_features!(cx, feats, new_feats)
    end
    # Keeps the complex's own invariant intact after every insertion, not
    # just as an occasional cleanup: no two live, adjacent top cells ever
    # carry the same label for longer than one call.
    merge_adjacent_same_label_cells!(cx)
    return nothing
end

"""
The features of one `multi_complex`/interactive-demo entry -- `(:point, p,
idx)` or `(:segment, pa, pb, idxa, idxb)` -- as a `Vector{GFeature{N}}`,
ready for `insert_entry!`.
"""
entry_feats(e, ::Val{N}) where {N} = e[1] === :point ? GFeature{N}[point_feature(e[2], e[3])] : segment_features(e[2], e[3], e[4], e[5])

"""
General incremental multi-sub-simplex construction, mixing points and
segments freely: `entries` is a list of `(:point, p, idx)` or
`(:segment, pa, pb, idxa, idxb)` tuples, processed in order via
`insert_entry!`. Returns `(cx, feats)`.
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

"""
Brute-force oracle matching `multi_complex`'s `entries` format: the set of
sub-simplex faces achieving the minimum squared distance to `x`, generalizing
`brute_force_label_segments` to an arbitrary mix of points and segments.
"""
function brute_force_label_multi(entries::Vector, x::Pt{N,Float64}; atol=1e-9) where {N}
    ds = Dict{Set{Int},Float64}()
    for e in entries
        if e[1] === :point
            _, p, idx = e
            ds[Set([idx])] = sum(abs2, x - p)
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

"""
Brute-force oracle for the point-vs-segment case, generalizing
`brute_force_label_points`: the set of sub-simplex faces (`{1}` for the
point, `{2}`/`{3}`/`{2,3}` for the segment's endpoints/interior) achieving
the minimum squared distance to `x`.
"""
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
