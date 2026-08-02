"""
`n+1` points (including both endpoints) sampled along `curve` between two
points `p1`, `p2` already known to lie on it -- parametrizes by `curve`'s
own natural single-valued axis (`curve_natural_axis`, the eigenvector of
its nonzero eigenvalue), not by whichever raw coordinate (x or y) happens
to vary more between `p1` and `p2`. That older convention needed a
"which of the two roots is nearer the previous sample" heuristic to stay
on the right branch, and could lose track of it entirely when the arc's
own turning point (where *neither* raw coordinate stays monotonic) fell
between `p1` and `p2` -- confirmed to actually happen and produce a
visibly self-intersecting result (see the self-intersection investigation
report). Parametrizing by the curve's own natural axis needs no branch
choice at all: projecting `curve` onto its own eigenbasis turns its
equation into `λu² + 2βᵤu + 2βᵥv + c = 0` (no `v²` or `uv` term, since
`axis`/`perp` are exactly `M`'s eigenvectors), which solves for `v`
*uniquely* given `u` -- linear, not quadratic, so there is no second root
to ever mistakenly pick.
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
The vertices of the 2D cell `id`, walked around its actual boundary in
order (not just collected and angle-sorted -- that trick only recovers the
right order for a straight-edged convex polygon, which every G1 cell is,
but no longer holds once a boundary edge is curved). Straight edges
contribute just their own start point; a curved edge (`curve !== nothing`,
since G2) is tessellated into several points along its true shape via
`tessellate_curve`, so a parabolic bisector renders as an actual arc
rather than the straight chord between its endpoints.
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
Whether `pt` lies inside the simple polygon `poly` (vertices in order,
either winding) -- the standard crossing-number/ray-casting test (PNPOLY).
Correct for non-convex polygons, unlike a same-side-of-every-edge check
(which implicitly assumes convexity): a generalized-Voronoi cell is *not*
always convex once curved bisectors are in play -- the region closer to a
line/segment-interior than to a nearby point is the non-convex complement
of the point's own (convex) region, so e.g. a segment's own territory can
genuinely wrap around a point.
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
Whether `pt` lies within node `id`'s own cached bounding box -- a cheap
`O(1)` reject usable before any exact-but-expensive test (walking a
tessellated curved boundary, a full point-in-polygon scan). Safe as a
*reject* precisely because the box is a guaranteed conservative enclosure
(see `CellNode`'s docstring): failing this check means `pt` cannot
possibly be inside `id`, but passing it is not itself proof of the
opposite -- the exact test downstream still has to run.
"""
function in_bbox(node::CellNode{2}, pt::Pt{2,Float64}; pad::Float64=0.0)
    return all(node.bbox_lo .- pad .<= pt) && all(pt .<= node.bbox_hi .+ pad)
end

"""
The unit direction along which a genuine curved edge's own `curve` is
single-valued: projecting any point known to lie on `curve` onto this axis
gives a value that uniquely identifies that point (the *other* coordinate
is then pinned down by `curve`'s own equation directly, not one of two
`±` choices) -- the eigenvector of `curve.M`'s nonzero eigenvalue. Every
curved edge this codebase actually stores is a genuine parabola (`M` rank
1, one zero eigenvalue) -- a line-vs-line bisector is always a pair of two
straight lines instead, handled by `clip_by_line_pair!` as two separate
flat edges, never stored as a single curved one -- so this needs no
general eigendecomposition: for a rank-1 `M` (`det(M) = 0`), `M`'s own
first row is *already* the eigenvector for the nonzero eigenvalue (falls
straight out of the algebra: `M*(a,b) = (a²+b², b(a+d))`, which is exactly
`(a,b)*(a+d)` whenever `ad-b²=0`, i.e. `det(M)=0` -- so `(a,b)` -- `M`'s
first row -- is already an eigenvector, with eigenvalue `a+d = tr(M)`).
Falls back to the second row if the first happens to be exactly zero
(only the second can be, since `M` isn't the all-zero matrix -- that would
mean `curve` isn't actually curved at all).
"""
function curve_natural_axis(curve::Quadric{2,Float64})
    r1 = SVector(curve.M[1, 1], curve.M[1, 2])
    r = norm(r1) > 1e-12 ? r1 : SVector(curve.M[1, 2], curve.M[2, 2])
    return r / norm(r)
end

"""
Whether `x` (assumed already on `curve`) lies on the arc strictly between
`p1` and `p2` (also on `curve`) -- exact, not a coordinate-range heuristic:
projects all three onto `curve`'s own natural single-valued axis
(`curve_natural_axis`) and checks whether `x`'s projection falls between
`p1`'s and `p2`'s. This is *not* the same as checking whether `x`'s literal
x or y coordinate falls between `p1`'s and `p2`'s -- that only works when
the arc's own turning point (a parabola's vertex) doesn't lie between `p1`
and `p2`, since only then does a raw coordinate stay monotonic along the
whole arc; `tessellate_curve`'s own "whichever coordinate varies more"
convention makes exactly that assumption, and produces a real,
self-intersecting discontinuity when it's violated (see the
self-intersection investigation report). Projecting onto the curve's own
natural axis has no such blind spot: it's single-valued along the *entire*
curve, turning point included, by construction.
"""
function on_arc_between(curve::Quadric{2,Float64}, p1::Pt{2,Float64}, p2::Pt{2,Float64}, x::Pt{2,Float64}; tol=1e-9)
    axis = curve_natural_axis(curve)
    t1, t2, tx = dot(axis, p1), dot(axis, p2), dot(axis, x)
    lo, hi = minmax(t1, t2)
    margin = tol * max(1.0, hi - lo)
    return lo - margin <= tx <= hi + margin
end

"""
The real `x` values where `curve` crosses the horizontal line `y = y0` --
exact, substituting `y=y0` into `curve`'s own equation
(`x²·M₁₁ + 2x·(M₁₂y₀+b₁) + (M₂₂y₀²+2b₂y₀+c) = 0`) and solving the
resulting quadratic in `x` directly. 0, 1, or 2 real roots.
"""
function curve_crossings_at_y(curve::Quadric{2,Float64}, y0::Float64)
    A = curve.M[1, 1]
    B = 2 * (curve.M[1, 2] * y0 + curve.b[1])
    C = curve.M[2, 2] * y0^2 + 2 * curve.b[2] * y0 + curve.c
    return quadratic_roots(A, B, C)
end

"""
Whether `pt` lies inside 2D cell `id`'s own region -- exact, not
tessellation-based: the standard even-odd ray-crossing rule (a horizontal
ray from `pt` toward `+∞` in `x`), generalized to boundary edges that may
themselves be curved. A straight edge contributes its ordinary single
crossing test (same algebra as `point_in_polygon_2d`). A curved edge can
cross the ray 0, 1, or 2 times -- found by solving its own equation
directly at `y = pt[2]` (`curve_crossings_at_y`), each candidate kept only
if it's genuinely on that edge's own bounded arc (`on_arc_between`, not a
coordinate-range guess) and to the right of `pt`. Correct for non-convex
cells the same way `point_in_polygon_2d` is (a generalized-Voronoi cell
isn't always convex once curved bisectors are in play), *and* immune to
the self-intersection a tessellated approximation of a curved edge can
introduce (see the self-intersection investigation report) -- this never
builds an approximate polygon at all, so there's nothing to self-intersect.
"""
point_in_cell_2d(cx::CellComplex{2}, id::Int, pt::Pt{2,Float64}) = point_in_edge_loop(cx, cx.nodes[id].subcells, pt)

"""
The id of the currently-live, labeled top-dimensional cell of `cx`
containing `pt` (2D point-location), or `nothing` if `pt` is outside every
cell (e.g. outside the bounding box, or exactly on a shared boundary where
rounding could go either way). Exact for both straight and curved
boundaries (`point_in_cell_2d`) -- each candidate cell is cheaply
bbox-rejected (`in_bbox`) first, so the exact test only runs for cells
that could actually contain `pt`. Still a linear scan over every live top
cell otherwise -- fine for the interactive demo's modest cell counts; a
BVH built over the same cached boxes (see `build_bvh`) is the natural
upgrade for much larger complexes.
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

# ---------------------------------------------------------------------------
# BVH: `find_containing_cell`'s bbox pre-check still has to visit every
# live top cell once per query (an O(1) reject is still an O(n) loop over
# all of them) -- a bounding-volume hierarchy built over the same cached
# boxes turns that into an O(log n) descent by ruling out whole *groups* of
# cells at once wherever a query point misses an interior node's own
# (already-conservative) box. Built fresh from `cx`'s current live top
# cells rather than maintained incrementally across insertions: a single
# `insert_entry!`/`insert_point!` call can supersede, split, and merge
# enough cells that incremental tree surgery would be its own significant
# bookkeeping burden, for a data structure that's cheap to rebuild outright
# (a handful of comparisons and sorts per cell) relative to the insertion
# step that just ran.

struct BVHNode
    lo::Pt{2,Float64}
    hi::Pt{2,Float64}
    left::Int      # index into the owning BVH's `nodes`, or 0 for a leaf
    right::Int     # 0 for a leaf
    cell_id::Int   # meaningful only when `left == 0` (a leaf)
end

"""
A static BVH over a fixed snapshot of `cx`'s live top cells' own cached
boxes (see `CellNode`'s docstring for why those are always a safe, if not
always perfectly tight, enclosure -- exactly the property a BVH needs from
its leaves to never wrongly skip a cell). `nodes` is laid out bottom-up
(post-order): every node's children always appear at *smaller* indices
than the node itself, so `root` is unambiguous even without a dedicated
pointer field -- it's simply `length(nodes)`.
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
Builds a fresh `BVH` over every currently-live, labeled top-dimensional
cell of `cx`, splitting each internal node's leaves at the median along
whichever axis its own box is widest on -- a plain median-split tree
(no surface-area-heuristic tuning), which is enough to turn point-location
from linear into logarithmic without adding real construction cost.
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
Point-location against `bvh` instead of a plain linear scan: descends only
into subtrees whose own box actually contains `pt`, exact-testing
(`point_in_cell_2d`) just the leaf cells that survive -- everywhere else,
whole groups of cells are ruled out by a single box check on their shared
ancestor. Returns the same answer `find_containing_cell(cx, pt)` would
(assuming `bvh` was built from this same, unmodified `cx`), just faster
for large cell counts.
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

"""
The true-shape polyline of boundary edge `id` (a live `dim == 1` cell): its
two endpoints directly if `curve === nothing`, or a `tessellate_curve`
approximation of its actual arc otherwise -- the same shape
`polygon_vertices_2d` walks each edge as, factored out so anything wanting
to draw or measure against an edge's real geometry (not the straight chord
between its endpoints) doesn't have to re-derive this.
"""
function edge_polyline(cx::CellComplex{2}, id::Int)
    node = cx.nodes[id]
    p1, p2 = cx.nodes[node.subcells[1]].point, cx.nodes[node.subcells[2]].point
    return node.curve === nothing ? [p1, p2] : tessellate_curve(node.curve, p1, p2)
end

"""
Distance from `pt` to the boundary edge `id` (a live, labeled `dim == 1`
cell), following its true shape via `edge_polyline` -- exact for straight
edges, a very close approximation for curved ones.
"""
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
`(:vertex, id)`, `(:edge, id)`, `(:cell, id)`, or `(nothing, nothing)`
(outside everything).

A plain point-in-polygon test (`find_containing_cell`) always resolves to
*some* cell, even for a cursor sitting exactly on (or a pixel away from)
the interface between two of them -- there's no way to ask "am I hovering
the *boundary*", which is exactly where the interesting information is
(the boundary's own label is a genuine tie, e.g. between two segments,
that neither adjacent cell's own single-element label shows). Boundaries
of boundaries need the same treatment one dimension down: a live *vertex*
(dim 0) can itself be a genuine multi-way tie -- e.g. where three cells'
territories meet at once -- that no single edge touching it can show (an
edge only ever carries the 2-way tie between the two cells *it*
separates). Vertices are checked first and win outright over an edge or
cell match within the same `thickness`: a vertex is the most specific
possible location, and every edge incident to it passes arbitrarily close
by construction, so without this priority a vertex could otherwise never
be the reported target at all.

Each candidate (whichever dimension) is cheaply bbox-rejected first
(`in_bbox`, padded by `thickness` so a genuinely within-tolerance
candidate can never be skipped just because `pt` sits outside its own
tight box), so the exact (and, for a curved edge, tessellation-walking)
distance only runs for candidates that could actually be within range.
`thickness` is in the same world-space units as `pt` -- callers that want
a constant *screen* thickness regardless of zoom (e.g. an interactive
demo) should scale it by their own current view extent.
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
Render every given top-dimensional (2D) cell of `cx` as a filled polygon
with its own color, plus the original input points marked in black --
matches the visual style of the 2D prototype's `plot_output_complex`, at
much smaller scope (G1's points-only case, one figure worth of cells
passed explicitly rather than queried from a fully general complex).

Implemented in the `MeshVoronoiGLMakieExt` package extension (only
loaded once the caller also has GLMakie loaded) so the core package
doesn't carry GLMakie as a hard dependency.
"""
function plot_cells_2d end
