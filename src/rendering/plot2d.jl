"""
`n+1` points sampled along `curve` between `p1`,`p2` (already on it) --
parametrizes by `curve`'s own natural single-valued axis
(`curve_natural_axis`), not by whichever raw coordinate happens to vary
more, which needed a branch-choice heuristic and could lose track when
the arc's own turning point fell between `p1`,`p2` (confirmed to produce
a self-intersecting result). Projecting onto the eigenbasis makes the
equation `λu² + 2βᵤu + 2βᵥv + c = 0` (no `v²`/`uv` term), solving for `v`
uniquely given `u` -- linear, no branch to pick.
"""
function tessellate_curve(curve::Quadric{2,Float64}, p1::Pt{2,Float64}, p2::Pt{2,Float64}; n=16)
    axis = curve_natural_axis(curve)
    perp = SVector(-axis[2], axis[1])
    u1, u2 = dot(axis, p1), dot(axis, p2)
    λ = curve.M[1, 1] + curve.M[2, 2]   # trace(M) = the nonzero eigenvalue, since the other is 0
    βu, βv = dot(curve.b, axis), dot(curve.b, perp)
    pts = Pt{2,Float64}[]
    for i in 0:n
        s = i / n
        u = u1 + s * (u2 - u1)
        v = -(λ * u^2 + 2 * βu * u + curve.c) / (2 * βv)
        push!(pts, u * axis + v * perp)
    end
    return pts
end

"""
The vertices of 2D cell `id`, walked around its actual boundary in order
(angle-sorting only works for a straight-edged convex polygon, not once a
boundary edge is curved). Straight edges contribute just their start
point; a curved edge is tessellated via `tessellate_curve` so it renders
as an actual arc, not a chord.
"""
function polygon_vertices_2d(cx::CellComplex{2}, id::Int)
    edges = cx.nodes[id].subcells
    n = length(edges)
    edge_verts = [(cx.nodes[e].subcells[1], cx.nodes[e].subcells[2]) for e in edges]

    used = falses(n)
    order = Int[1]
    used[1] = true
    entry_vertex = edge_verts[1][1]
    cur_vertex = edge_verts[1][2]
    for _ in 2:n
        found_idx = 0
        for i in 1:n
            used[i] && continue
            a, b = edge_verts[i]
            if a == cur_vertex || b == cur_vertex
                found_idx = i
                break
            end
        end
        found_idx == 0 && error("polygon_vertices_2d: cell $id's boundary edges don't form a simple cycle")
        used[found_idx] = true
        push!(order, found_idx)
        a, b = edge_verts[found_idx]
        cur_vertex = (a == cur_vertex) ? b : a
    end

    pts = Pt{2,Float64}[]
    walk_vertex = entry_vertex
    for idx in order
        e = edges[idx]
        enode = cx.nodes[e]
        a, b = edge_verts[idx]
        from, to = (a == walk_vertex) ? (a, b) : (b, a)
        p_from, p_to = cx.nodes[from].point, cx.nodes[to].point
        if enode.curve === nothing
            push!(pts, p_from)
        else
            seg = tessellate_curve(enode.curve, p_from, p_to)
            append!(pts, seg[1:end-1])
        end
        walk_vertex = to
    end
    return pts
end

"""
Whether `pt` lies inside simple polygon `poly` -- standard crossing-
number/ray-casting test (PNPOLY). Correct for non-convex polygons, unlike
a same-side-of-every-edge check: a generalized-Voronoi cell isn't always
convex once curved bisectors are in play (e.g. a segment's own territory
can wrap around a point).
"""
function point_in_polygon_2d(poly::Vector{Pt{2,Float64}}, pt::Pt{2,Float64})
    n = length(poly)
    inside = false
    j = n
    for i in 1:n
        xi, yi = poly[i][1], poly[i][2]
        xj, yj = poly[j][1], poly[j][2]
        if (yi > pt[2]) != (yj > pt[2])
            x_at_y = xi + (xj - xi) * (pt[2] - yi) / (yj - yi)
            pt[1] < x_at_y && (inside = !inside)
        end
        j = i
    end
    return inside
end

"""
Whether `pt` lies within node `id`'s cached bounding box -- a cheap O(1)
reject before an exact-but-expensive test. Safe as a reject because the
box is a guaranteed conservative enclosure; passing doesn't prove `pt` is
actually inside.
"""
function in_bbox(node::CellNode{2}, pt::Pt{2,Float64}; pad::Float64=0.0)
    return all(node.bbox_lo .- pad .<= pt) && all(pt .<= node.bbox_hi .+ pad)
end

"""
Unit direction along which `curve`'s own arc is single-valued (injective)
-- projecting onto this axis never has a turning point, so any two points
on the arc bracket everything genuinely between them (used by
`on_arc_between`/`expand_bbox_for_curve`/`line_meets_quadric`'s own
in-range check to tell "between p1,p2 along the curve" from a raw
coordinate range, which breaks whenever the arc bends back on an axis).

For rank-1 `M` (a parabola: point-vs-line bisector) the nonzero
eigenvalue's own eigenvector is already that axis -- `M`'s first row
falls out of the algebra directly, no eigendecomposition needed.

For rank-2 indefinite `M` (a hyperbola: line-vs-line bisector -- confirmed
reachable, not just theoretical, since `restrict_to_plane`d segment-vs-
segment edges are genuinely rank 2) there are *two* principal axes, and
only one is injective: the *transverse* axis (through the two branches'
own vertices) doubles back on itself at the vertex, exactly the turning
point this function exists to avoid, while the *conjugate* axis is
injective along the whole branch. Which eigenvector is which flips with
the hyperbola's own orientation, so it's resolved algebraically rather
than guessed: completing the square at the curve's own center
`x0 = -M⁻¹b` leaves `evaluate(curve,x0)` as the reduced constant `K`; the
injective axis is whichever eigenvector's eigenvalue shares `K`'s sign
(confirmed against a genuine repro where using the *other* axis silently
rejected a real, in-range crossing point -- the root cause of curve2
edges connecting two disconnected pieces of a tie locus, session 2026-08-09).
"""
function curve_natural_axis(curve::Quadric{2,Float64})
    M = curve.M
    decomp = eigen(Symmetric(Matrix(M)))
    vals = decomp.values
    tol = 1e-9 * max(1.0, maximum(abs.(vals)))
    if abs(vals[1]) < tol || abs(vals[2]) < tol
        i = abs(vals[1]) < tol ? 2 : 1
        return SVector{2,Float64}(decomp.vectors[:, i])
    end
    center = -(M \ curve.b)
    K = evaluate(curve, center)
    i = sign(vals[1]) == sign(K) ? 1 : 2
    return SVector{2,Float64}(decomp.vectors[:, i])
end

"""
Whether `x` (on `curve`) lies strictly between `p1`,`p2` (also on
`curve`) -- exact, not a coordinate-range heuristic: projects all three
onto `curve`'s own natural axis (`curve_natural_axis`), which stays
single-valued through the arc's turning point, unlike a raw x/y range
check.
"""
function on_arc_between(curve::Quadric{2,Float64}, p1::Pt{2,Float64}, p2::Pt{2,Float64}, x::Pt{2,Float64}; tol=1e-9)
    axis = curve_natural_axis(curve)
    t1, t2, tx = dot(axis, p1), dot(axis, p2), dot(axis, x)
    lo, hi = minmax(t1, t2)
    margin = tol * max(1.0, hi - lo)
    return lo - margin <= tx <= hi + margin
end

"""Real `x` values where `curve` crosses `y=y0` -- substitute and solve the resulting quadratic."""
function curve_crossings_at_y(curve::Quadric{2,Float64}, y0::Float64)
    A = curve.M[1, 1]
    B = 2 * (curve.M[1, 2] * y0 + curve.b[1])
    C = curve.M[2, 2] * y0^2 + 2 * curve.b[2] * y0 + curve.c
    return quadratic_roots(A, B, C)
end

"""
Whether `pt` lies inside 2D cell `id`'s region -- exact, not tessellation-
based: even-odd ray-crossing generalized to curved boundary edges. A
curved edge can cross the ray 0-2 times (`curve_crossings_at_y`), kept
only if genuinely on the edge's bounded arc (`on_arc_between`). Immune to
the self-intersection a tessellated approximation could introduce, since
no approximate polygon is ever built.
"""
point_in_cell_2d(cx::CellComplex{2}, id::Int, pt::Pt{2,Float64}) = point_in_edge_loop(cx, cx.nodes[id].subcells, pt)

"""
The id of the live top-dimensional cell of `cx` containing `pt`, or
`nothing`. Exact for straight and curved boundaries; bbox-rejected first.
Linear scan -- `build_bvh` is the upgrade for larger complexes.
"""
function find_containing_cell(cx::CellComplex{2}, pt::Pt{2,Float64})
    for id in eachindex(cx.nodes)
        haskey(cx.superseded_by, id) && continue
        node = cx.nodes[id]
        (node.dim == 2 && !isempty(node.label)) || continue
        in_bbox(node, pt) || continue
        point_in_cell_2d(cx, id, pt) && return id
    end
    return nothing
end

# BVH: turns find_containing_cell's O(n) bbox scan into an O(log n)
# descent. Built fresh from cx's current live top cells each time rather
# than maintained incrementally -- cheap to rebuild outright vs. the
# bookkeeping of tree surgery across supersede/split/merge.

struct BVHNode
    lo::Pt{2,Float64}
    hi::Pt{2,Float64}
    left::Int      # index into the owning BVH's `nodes`, or 0 for a leaf
    right::Int     # 0 for a leaf
    cell_id::Int   # meaningful only when `left == 0` (a leaf)
end

"""
A static BVH over a fixed snapshot of `cx`'s live top cells' cached boxes.
`nodes` is laid out bottom-up (post-order): children always at smaller
indices than their parent, so `root` is simply `length(nodes)`.
"""
struct BVH
    nodes::Vector{BVHNode}
    root::Int
end

function build_bvh_range!(nodes::Vector{BVHNode}, leaves::AbstractVector{<:Tuple})
    if length(leaves) == 1
        id, lo, hi = leaves[1]
        push!(nodes, BVHNode(lo, hi, 0, 0, id))
        return length(nodes)
    end
    lo = reduce((a, b) -> min.(a, b), (l[2] for l in leaves))
    hi = reduce((a, b) -> max.(a, b), (l[3] for l in leaves))
    extent = hi .- lo
    axis = extent[1] >= extent[2] ? 1 : 2
    sorted = sort(collect(leaves), by=l -> l[2][axis] + l[3][axis])
    mid = length(sorted) ÷ 2
    left = build_bvh_range!(nodes, view(sorted, 1:mid))
    right = build_bvh_range!(nodes, view(sorted, mid+1:length(sorted)))
    push!(nodes, BVHNode(lo, hi, left, right, 0))
    return length(nodes)
end

"""
Builds a fresh `BVH` over every live, labeled top-dimensional cell of
`cx`, splitting at the median along the widest axis -- plain median-split,
no SAH tuning, enough to make point-location logarithmic.
"""
function build_bvh(cx::CellComplex{2})
    leaves = Tuple{Int,Pt{2,Float64},Pt{2,Float64}}[]
    for id in eachindex(cx.nodes)
        haskey(cx.superseded_by, id) && continue
        node = cx.nodes[id]
        (node.dim == 2 && !isempty(node.label)) || continue
        push!(leaves, (id, node.bbox_lo, node.bbox_hi))
    end
    nodes = BVHNode[]
    isempty(leaves) && return BVH(nodes, 0)
    root = build_bvh_range!(nodes, leaves)
    return BVH(nodes, root)
end

"""
Point-location against `bvh`: descends only into subtrees whose box
contains `pt`, exact-testing just the surviving leaves. Same answer as
`find_containing_cell`, faster for large cell counts.
"""
function find_containing_cell_bvh(cx::CellComplex{2}, bvh::BVH, pt::Pt{2,Float64})
    bvh.root == 0 && return nothing
    stack = [bvh.root]
    while !isempty(stack)
        i = pop!(stack)
        n = bvh.nodes[i]
        (all(n.lo .<= pt) && all(pt .<= n.hi)) || continue
        if n.left == 0
            point_in_cell_2d(cx, n.cell_id, pt) && return n.cell_id
        else
            push!(stack, n.left)
            push!(stack, n.right)
        end
    end
    return nothing
end

"""
Squared distance from `pt` to the closest point on the segment `a`-`b`.
"""
function point_to_segment_sqdist(pt::Pt{2,Float64}, a::Pt{2,Float64}, b::Pt{2,Float64})
    ab = b - a
    len2 = dot(ab, ab)
    len2 < 1e-18 && return sum(abs2, pt - a)
    t = clamp(dot(pt - a, ab) / len2, 0.0, 1.0)
    return sum(abs2, pt - (a + t * ab))
end

"""True-shape polyline of boundary edge `id`: its two endpoints, or a `tessellate_curve` approximation if curved."""
function edge_polyline(cx::CellComplex{2}, id::Int)
    node = cx.nodes[id]
    p1, p2 = cx.nodes[node.subcells[1]].point, cx.nodes[node.subcells[2]].point
    return node.curve === nothing ? [p1, p2] : tessellate_curve(node.curve, p1, p2)
end

"""Distance from `pt` to boundary edge `id`, via `edge_polyline` -- exact for straight edges, close approximation for curved."""
function edge_distance(cx::CellComplex{2}, id::Int, pt::Pt{2,Float64})
    pts = edge_polyline(cx, id)
    d2 = Inf
    for i in 1:(length(pts)-1)
        d2 = min(d2, point_to_segment_sqdist(pt, pts[i], pts[i+1]))
    end
    return sqrt(d2)
end

"""
Point-location with a thickness tolerance for boundaries, returning
`(:vertex, id)`, `(:edge, id)`, `(:cell, id)`, or `(nothing, nothing)`.

A plain point-in-polygon test always resolves to *some* cell, with no way
to ask "am I hovering the boundary" -- exactly where the interesting
label information is (a genuine tie neither adjacent cell's own label
shows). Vertices are checked first and win outright within the same
`thickness`: the most specific location, and every incident edge passes
arbitrarily close, so without this priority a vertex could never be the
reported target.

`thickness` is in world-space units -- callers wanting a constant
*screen* thickness should scale it by their own current view extent.
"""
function find_hover_target(cx::CellComplex{2}, pt::Pt{2,Float64}, thickness::Float64)
    best_vertex, best_vertex_dist = nothing, thickness
    for id in eachindex(cx.nodes)
        haskey(cx.superseded_by, id) && continue
        node = cx.nodes[id]
        (node.dim == 0 && !isempty(node.label)) || continue
        in_bbox(node, pt; pad=thickness) || continue
        d = sqrt(sum(abs2, pt - node.point))
        if d < best_vertex_dist
            best_vertex_dist = d
            best_vertex = id
        end
    end
    best_vertex !== nothing && return (:vertex, best_vertex)

    best_id, best_dist = nothing, thickness
    for id in eachindex(cx.nodes)
        haskey(cx.superseded_by, id) && continue
        node = cx.nodes[id]
        (node.dim == 1 && !isempty(node.label)) || continue
        in_bbox(node, pt; pad=thickness) || continue
        d = edge_distance(cx, id, pt)
        if d < best_dist
            best_dist = d
            best_id = id
        end
    end
    best_id !== nothing && return (:edge, best_id)
    cell_id = find_containing_cell(cx, pt)
    cell_id !== nothing && return (:cell, cell_id)
    return (nothing, nothing)
end

"""
Render given top-dimensional cells of `cx` as filled polygons, input
points marked in black. Implemented in the `MeshVoronoiGLMakieExt`
package extension so the core package doesn't carry GLMakie as a hard
dependency.
"""
function plot_cells_2d end
