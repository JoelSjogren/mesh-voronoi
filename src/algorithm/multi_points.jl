"""
The true tied-winner set at `pt`, by direct distance comparison against
every input point -- needed because union-ing duplicate vertices' own
labels doesn't always recover the full tie (each only reflects whichever
single pairwise comparison created it). Points-only counterpart to
`recompute_feature_label` (segments.jl).
"""
function recompute_point_label(pt::Pt{N,Float64}, points::Dict{VertexIdx,Pt{N,Float64}}; atol=1e-9) where {N}
    ds = Dict(i => sum(abs2, pt - p) for (i, p) in points)
    m = minimum(values(ds))
    return Label([Set([i]) for (i, d) in ds if d <= m + atol * max(1.0, m)])
end

"""
Recomputes the true label of every live dim=0 vertex created during this
insertion (id `>= first_new_id`), welding together any that turn out to
be numerically the same point.

Welding is needed because two adjacent cells' independent clips against
the same new winner compute a shared split point using different
quadrics (each vs. its own current winner) -- these coincide only in
exact arithmetic, so float rounding leaves two near-duplicate points
instead of one true vertex.

Every new vertex's label is recomputed from scratch, not just welded
ones: any newly-created vertex's own label only reflects whichever single
pairwise clip created it, never the full tie a genuine boundary point
represents. A vertex approached from only one side (e.g. the outer
boundary of a compactified complex, with no "other side" cell to produce
a matching duplicate) never becomes a "duplicate" and would otherwise
keep a one-sided label forever.

A zero-length sliver edge left by a weld is retired via `supersede!`.
Scoped to vertices created this round plus whatever the caller passes in
`extra_verts` -- e.g. descendants of a cell wholesale-relabeled (not
clipped) by the caller's own "no split needed" shortcut, which only sets
the cell's own label, never cascading to its subcells.
"""
function weld_near_duplicate_vertices!(cx::CellComplex{N}, first_new_id::Int, label_fn; atol=1e-9, extra_verts=Int[]) where {N}
    verts = [id for id in first_new_id:length(cx.nodes) if !haskey(cx.superseded_by, id) && cx.nodes[id].dim == 0]
    for id in extra_verts
        (haskey(cx.superseded_by, id) || cx.nodes[id].dim != 0 || id in verts) && continue
        push!(verts, id)
    end
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
After vertices are welded above, two live edges created during this
insertion can end up connecting the same pair of (now-canonical)
endpoints without being the same stored node -- same root cause as
duplicate vertices, one dimension up (two adjacent cells' independent
clips against the same new winner each creating their own copy of what's
geometrically one shared cut boundary). Left alone, `polygon_vertices_2d`
walks the same curve twice and the resulting polygons overlap.

Two edges are only welded if they share endpoints *and* are the same
curve (both straight, or matching quadric coefficients) -- shared
endpoints alone isn't enough, since a straight and a curved edge (or two
different curves) can share both endpoints without being the same
boundary.
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
    remove_degenerate_faces!(cx, first_new_id)
    return nothing
end

"""
Removes a face left with no valid closed boundary by the welding above
(e.g. a zero-area sliver whose own third corner turns out to be one of
the other two once welded) -- `supersede!` with no replacement; its real
neighboring faces close up fine without it. A multiply-connected face
(legitimate annular case) is left alone; only the unrecoverable zero-loop
case is degenerate.

At `N=3`, `node.dim==2` is a genuine face (not a 1D boundary), so uses
`face_boundary_faces`, which can develop a real branch point
`cyclic_boundary_walks` can't represent at all. A flat face's own route
through `flat_face_frame_cached!` throws (rather than returning empty)
when it can't find 3 non-collinear points -- exactly the degenerate case
this function exists to catch, so caught here and treated the same as
empty loops.
"""
function remove_degenerate_faces!(cx::CellComplex{N}, first_new_id::Int) where {N}
    for id in first_new_id:length(cx.nodes)
        haskey(cx.superseded_by, id) && continue
        node = cx.nodes[id]
        node.dim == 2 || continue
        degenerate = if N == 3
            try
                isempty(face_boundary_faces(cx, id))
            catch e
                # Only a flat-plane-derivation failure means "degenerate"
                # -- any other error is a real bug and must propagate.
                e isa ErrorException && startswith(e.msg, "flat_face_frame") || rethrow()
                true
            end
        else
            isempty(cyclic_boundary_walks(cx, node.subcells))
        end
        degenerate && supersede!(cx, id, Int[])
    end
    return nothing
end

"""
Every live `(N-1)`-dimensional boundary's label should be the union of
the live top cells it separates -- re-derived directly from current live
parents rather than trusted from creation time, which can go stale right
after a weld changes what a boundary actually separates.
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
Incorporate one new point (`new_idx`) into a complex already correct for
every other point: for each currently-live cell (one per existing
winner), check whether the new point beats the winner anywhere in it, and
if so clip by their bisector.

Cells can share boundary structure (an edge between two adjacent cells is
the same stored node) -- clipping one updates its neighbor too via
`clip_by_hyperplane!`'s own `supersede!` calls. `to_process` is
snapshotted once at the start, so a clip for one winner doesn't change
which others get visited.
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
    # Every vertex touched by *any* branch below -- clipped (where
    # `clip_by_hyperplane!`'s own dim=0 labels are only ever provisional,
    # see its docstring) or wholesale relabeled (that branch only sets
    # `cell_id`'s own label, never cascading to its subcells at all) -- can
    # end up with a stale or incomplete label unless something explicitly
    # recomputes it afterward. Collected *before* processing each cell (so
    # the ids are captured regardless of what that cell's own clip does to
    # it; vertex ids themselves are never superseded by a clip, only
    # cells/edges are) and handed to `weld_near_duplicate_vertices!` below
    # alongside its usual newly-created-vertex scope.
    touched_verts = Int[]
    for (winner_idx, cell_id) in to_process
        append!(touched_verts, descendant_nodes_by_dim(cx, cell_id)[1])
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
    # of the one true one, neither carrying the full tied label. Weld those,
    # recompute every touched pre-existing vertex's own true label too (see
    # `touched_verts` above), then fix any boundary whose label is now stale
    # relative to what it actually separates.
    weld_near_duplicate_vertices!(cx, first_new_id, pt -> recompute_point_label(pt, points); extra_verts=touched_verts)
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
`points_complex`'s "compactified" counterpart: the starting domain is the
input's own convex hull, offset outward by `D` (default: `mult` times the
bounding-box diagonal), instead of `padded_bbox`'s axis-aligned rectangle
-- otherwise identical, since nothing about construction needs to know
its domain is a hull offset rather than a box. Needs at least 3 affinely
independent points. Returns `(cx, hull, offset)`.
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
