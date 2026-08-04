"""
The true tied-winner set at `pt`, found by direct distance comparison
against every known input point -- necessary when welding duplicate
vertices (see `weld_near_duplicate_vertices!`) because simply *union*-ing
the duplicates' own labels does not always recover the full tie: each
duplicate's label only reflects whichever single pairwise comparison its
own originating clip happened to make, not the true multi-way tie a
genuine higher-order Voronoi vertex represents. The G1 (points-only)
counterpart to `recompute_feature_label` (segments.jl), which does the
same thing for G2's flat feature list.
"""
function recompute_point_label(pt::Pt{N,Float64}, points::Dict{VertexIdx,Pt{N,Float64}}; atol=1e-9) where {N}
    ds = Dict(i => sum(abs2, pt - p) for (i, p) in points)
    m = minimum(values(ds))
    return Label([Set([i]) for (i, d) in ds if d <= m + atol * max(1.0, m)])
end

"""
Welds together any live dim=0 vertices created *during this insertion*
(id `>= first_new_id`) that turn out to be numerically the same point --
this can genuinely happen even though the construction is logically
correct: both `insert_point!` (G1) and `insert_features!` (G2) clip each
existing winner's top cell independently, so when two adjacent cells share
a boundary edge that the new winner's bisector needs to split, *each*
cell's clip recomputes where its own copy of that split happens using a
*different* quadric (its own current winner vs. the new one -- not some
single quadric tied to the shared edge itself). These two computations
coincide only in exact arithmetic, precisely at a genuine multi-way tie
point -- so floating-point rounding gives two distinct-but-nearly-identical
points instead of the one true vertex.

Welding matters, not just cosmetically: neither duplicate's own label (an
artifact of whichever single clip created it) reflects the full tie a
genuine multi-way tie point represents, so the welded vertex's true label
is recomputed from scratch via the caller-supplied `label_fn` (either
`recompute_point_label` or `recompute_feature_label`, generalizing over
G1/G2 -- this function itself doesn't need to know which) rather than
merely unioning the duplicates' own incomplete ones. Any edge left
connecting a welded vertex to itself (a zero-length sliver, a direct side
effect of the weld) is retired via `supersede!` with an empty replacement,
the same way orphaned structure elsewhere in the complex is retired -- not
a real geometric feature, and `polygon_vertices_2d`'s boundary walk works
fine without it once both of its former neighbors resolve to the same
shared vertex.

Scoped to only the vertices created this round (not the whole complex)
for efficiency -- a vertex that was already live before this insertion is
already correctly deduplicated, by this same invariant maintained after
every prior insertion.
"""
function weld_near_duplicate_vertices!(cx::CellComplex{N}, first_new_id::Int, label_fn; atol=1e-9) where {N}
    verts = [id for id in first_new_id:length(cx.nodes) if !haskey(cx.superseded_by, id) && cx.nodes[id].dim == 0]
    isempty(verts) && return nothing

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
        d = sqrt(sum(abs2, pa - pb))
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
        true_label = label_fn(cx.nodes[canonical].point)
        set_label!(cx, canonical, true_label)
        for id in group
            id == canonical && continue
            supersede!(cx, id, [canonical])
        end
    end

    for id in first_new_id:length(cx.nodes)
        haskey(cx.superseded_by, id) && continue
        node = cx.nodes[id]
        (node.dim == 1 && length(node.subcells) == 2) || continue
        a, b = node.subcells
        a == b && supersede!(cx, id, Int[])
    end
    return nothing
end

"""
After vertices are welded above, two live edges created *during this
insertion* can end up connecting the exact same pair of (now-canonical)
endpoints without themselves being the same stored node -- the same root
cause as duplicate vertices, one dimension up: independently clipping two
adjacent cells against the same new winner can each create their own copy
of what's geometrically the same shared cut boundary, rather than the two
cells properly referencing one shared edge (the normal, intended
architecture -- see `CellComplex`'s docstring). Left alone, each duplicate
stays exclusively owned by a different cell, so `polygon_vertices_2d`
walks the same curve twice -- once per cell -- and the resulting polygons
overlap almost entirely along what should have been their shared,
disjoint-interior boundary instead (this is exactly what surfaced the
bug: a regression test that already existed for a *different* curved-edge
overlap issue caught this one too).

Two edges are only welded if they connect the same endpoint pair *and* are
the same curve (both straight, or both curved with matching quadric
coefficients) -- sharing endpoints alone isn't sufficient, since a
straight edge and a curved one (or two genuinely different curves) can
share both endpoints without being the same boundary.
"""
function weld_duplicate_edges!(cx::CellComplex{N}, first_new_id::Int) where {N}
    edges = [id for id in first_new_id:length(cx.nodes) if !haskey(cx.superseded_by, id) && cx.nodes[id].dim == 1]
    isempty(edges) && return nothing

    by_endpoints = Dict{Tuple{Int,Int},Vector{Int}}()
    for id in edges
        a, b = cx.nodes[id].subcells
        key = a <= b ? (a, b) : (b, a)
        push!(get!(() -> Int[], by_endpoints, key), id)
    end

    same_curve(c1, c2) = c1 === nothing || c2 === nothing ? c1 === c2 :
                          isapprox(c1.M, c2.M) && isapprox(c1.b, c2.b) && isapprox(c1.c, c2.c)

    for (_, group) in by_endpoints
        length(group) < 2 && continue
        remaining = copy(group)
        while length(remaining) > 1
            id1 = popfirst!(remaining)
            matches = [id1]
            keep = Int[]
            for id2 in remaining
                same_curve(cx.nodes[id1].curve, cx.nodes[id2].curve) ? push!(matches, id2) : push!(keep, id2)
            end
            remaining = keep
            length(matches) < 2 && continue
            canonical = minimum(matches)
            for id in matches
                id == canonical && continue
                supersede!(cx, id, [canonical])
            end
        end
    end
    return nothing
end

"""
Every live `(N-1)`-dimensional boundary's label should be the union of the
(currently live) top cells it separates -- the sub/super-cell duality this
whole complex is built on (a boundary's label is a *superset* of what it
bounds, see `Label`'s docstring). Re-derives that directly from the
current live parents rather than trusting whatever label a boundary
happened to be given at its own moment of creation, which can be stale or
incomplete right after a freshly welded vertex changes what a boundary
actually separates (see `weld_near_duplicate_vertices!`). Scoped to this
insertion's own new nodes for the same efficiency reason as the weld pass.
"""
function fix_boundary_labels!(cx::CellComplex{N}, first_new_id::Int) where {N}
    for id in first_new_id:length(cx.nodes)
        haskey(cx.superseded_by, id) && continue
        node = cx.nodes[id]
        node.dim == N - 1 || continue
        parents = [p for p in get(cx.referenced_by, id, Int[]) if !haskey(cx.superseded_by, p) && cx.nodes[p].dim == N && !isempty(cx.nodes[p].label)]
        isempty(parents) && continue
        union_label = reduce(union, (cx.nodes[p].label for p in parents))
        union_label != node.label && set_label!(cx, id, union_label)
    end
    return nothing
end

"""
Incorporate one new point (`new_idx`, coordinate `points[new_idx]`) into a
complex that already correctly reflects every *other* point in `points`:
for each currently-live cell (one per existing winner -- each winner has
exactly one, since a point's territory stays a single connected convex
region throughout G1's flat-bisector-only construction), check whether the
new point beats the existing winner anywhere in it, and if so clip that
cell by their bisector.

Cells for different existing winners are independent regions, but they can
share boundary structure (an edge between two adjacent cells is the same
stored node, referenced by both) -- clipping one cell's shared boundary
therefore needs to correctly update its neighbor too, which is exactly
what `clip_by_hyperplane!`'s `supersede!` calls handle. Processing winners
in the order captured by `to_process` (snapshotted once at the start, so a
clip performed for one winner doesn't change which *other* winners get
visited) means that by the time a later winner's cell is examined, any
shared boundary it has with an earlier-processed winner is already
patched to reflect the new split.
"""
function insert_point!(cx::CellComplex{N}, points::Dict{VertexIdx,Pt{N,Float64}}, new_idx::VertexIdx) where {N}
    new_point = points[new_idx]
    B = AffineQuadratic(new_point)

    # Only *top-dimensional* (dim N) cells are "territories" to compare the
    # new point against -- a winner's label also sits on that territory's
    # own lower-dimensional boundary pieces (an edge or vertex entirely
    # within it gets the same single-point label), and those must NOT be
    # independently reprocessed here: `clip_by_hyperplane!` already walks a
    # whole territory's full descendant tree in one pass, so touching a
    # boundary piece separately just duplicates that work against a
    # subset of the same points, corrupting the complex.
    to_process = Tuple{VertexIdx,Int}[]
    for winner_idx in keys(points)
        winner_idx == new_idx && continue
        for cell_id in get(cx.label_index, Label([Set([winner_idx])]), Int[])
            cx.nodes[cell_id].dim == N || continue
            push!(to_process, (winner_idx, cell_id))
        end
    end

    first_new_id = length(cx.nodes) + 1
    for (winner_idx, cell_id) in to_process
        A = AffineQuadratic(points[winner_idx])
        quad = bisector(A, B)
        pts = descendant_points(cx, cell_id)
        signs = [exact_sign((q, x) -> evaluate(q, x), quad, p) for p in pts]
        if all(<=(0), signs)
            continue   # the new point never wins here -- no change
        elseif all(>=(0), signs)
            set_label!(cx, cell_id, Label([Set([new_idx])]))
        else
            clip_by_hyperplane!(cx, cell_id, A, B, Set([winner_idx]), Set([new_idx]))
        end
    end
    # Two different winners' cells sharing a boundary that this new point's
    # bisector needs to split are clipped independently above, each against
    # its *own* winner-vs-new-point quadric -- these coincide only in exact
    # arithmetic (precisely at a genuine multi-way tie point), so
    # floating-point rounding can leave two near-duplicate vertices instead
    # of the one true one, neither carrying the full tied label. Weld
    # those, then fix any boundary whose label is now stale relative to
    # what it actually separates -- both scoped to this round's own new
    # nodes only.
    weld_near_duplicate_vertices!(cx, first_new_id, pt -> recompute_point_label(pt, points))
    weld_duplicate_edges!(cx, first_new_id)
    fix_boundary_labels!(cx, first_new_id)
    # Keeps the complex's own invariant intact after every insertion: no
    # two live, adjacent top cells ever carry the same label for longer
    # than one call (see `merge_adjacent_same_label_cells!`'s docstring).
    merge_adjacent_same_label_cells!(cx)
    return nothing
end

"""
The full points-only (G1) construction for an arbitrary number of input
points, general `N`: a padded bounding box, the first point labeled
uncontested everywhere, then every subsequent point incorporated one at a
time via `insert_point!`.
"""
function points_complex(points::Vector{Pt{N,Float64}}) where {N}
    length(points) >= 1 || error("points_complex: need at least one point")
    lo, hi = padded_bbox(points)
    cx, top = init_bbox_complex(Val(N), lo, hi)
    label1 = Label([Set([1])])
    for id in eachindex(cx.nodes)
        set_label!(cx, id, label1)
    end
    points_dict = Dict{VertexIdx,Pt{N,Float64}}(i => p for (i, p) in enumerate(points))
    for i in 2:length(points)
        insert_point!(cx, points_dict, i)
    end
    return cx
end

"""
`points_complex`'s "compactified" counterpart (see the layer-at-infinity
planning report): the starting domain is the input's own convex hull,
offset outward by a fixed distance `D` (default: `mult` times the input's
own bounding-box diagonal, generous enough that the boundary sits well
outside any finite structure the construction itself produces), instead of
`padded_bbox`'s arbitrary axis-aligned rectangle. Otherwise identical to
`points_complex` -- same per-point incremental insertion, same labeling --
because nothing about the construction algorithm needs to know its own
domain is a hull offset rather than a box; see the report for why that's
exactly the point. Needs at least 3 affinely independent points (the
generic case this targets) for the hull itself to be well-defined; returns
`(cx, hull, offset)` so a caller can label the boundary-at-infinity's own
edges/vertices afterward without recomputing the hull.
"""
function compactified_points_complex(points::Vector{Pt{2,Float64}}; mult::Float64=20.0)
    length(points) >= 3 || error("compactified_points_complex: need at least 3 (affinely independent) points")
    hull = convex_hull_2d(points)
    lo = SVector(minimum(p[1] for p in points), minimum(p[2] for p in points))
    hi = SVector(maximum(p[1] for p in points), maximum(p[2] for p in points))
    D = mult * max(norm(hi - lo), 1.0)
    offset = offset_polygon(hull, D)
    cx, top = init_hull_offset_complex(offset)
    label1 = Label([Set([1])])
    for id in eachindex(cx.nodes)
        set_label!(cx, id, label1)
    end
    points_dict = Dict{VertexIdx,Pt{2,Float64}}(i => p for (i, p) in enumerate(points))
    for i in 2:length(points)
        insert_point!(cx, points_dict, i)
    end
    return cx, hull, offset
end

"""
Brute-force oracle for an arbitrary number of points, generalizing
`brute_force_label_two_points`.
"""
function brute_force_label_points(points::Vector{Pt{N,Float64}}, x::Pt{N,Float64}; atol=1e-9) where {N}
    ds = [sum(abs2, x - p) for p in points]
    m = minimum(ds)
    return Set(i for (i, d) in enumerate(ds) if d <= m + atol * max(1.0, m))
end
