const VertexIdx = Int

"""
A cell's label: the set of input sub-simplices tied there (see the project
plan's "sub/super-cell duality" discussion). For the points-only milestone
(G1), every input sub-simplex is just a single vertex, so every label
entry is a singleton `Set{VertexIdx}`.
"""
const Label = Set{Set{VertexIdx}}

"""
One stored cell of a `CellComplex{N}`: its dimension, its label, and the
ids of its immediate *subcells* (its boundary pieces -- see the plan for
why this is the direction that needs explicit indexing, unlike a mesh's
sub-simplices). Per the chosen "recursive, bottom-up" geometry convention,
a cell's shape *is* the union of its subcells' shapes; the only cells that
carry their own independent geometric data beyond that are 0-cells (a
single point) and, since G2, 1-cells that lie on a genuinely curved
bisector (`curve`, `nothing` for a straight edge) -- a CW-complex's 1-cells
are attached via an arbitrary continuous map, not necessarily a straight
line, and G1's all-hyperplane construction was simply the special case
where that map always happened to be linear (so `curve === nothing`
always, and every downstream consumer could safely treat an edge as the
straight segment between its two endpoints). G2 introduces bisectors
(point-vs-segment-interior, etc.) whose zero set is a genuine curve
(a parabola in 2D), so a 1-cell now needs to remember *which* quadric it
lies on for anything that needs its true shape (rendering, sampling a
curved boundary) rather than a chord approximation.

A cell's own axis-aligned bounding box is part of its identity, not a
derived afterthought: two live cells sharing a label are only genuinely
"the same place" if their bounding boxes are disjoint from every *other*
same-label cell (see `assert_label_bbox_invariant`) -- so `bbox_lo`/
`bbox_hi` are computed once, bottom-up, when a node is created (`add_cell!`
below), the same way its geometry itself is defined recursively from its
subcells. This makes the box a cheap O(1) field lookup everywhere it's
needed (an invariant check, a point-location fast-reject, a BVH leaf)
instead of an O(subtree size) walk repeated on every call.

The box is *conservative*, not always perfectly tight: if a neighboring
insertion later subdivides one of this cell's subcells via `supersede!`
(patching this node's own `subcells` list in place without touching this
node otherwise -- see `supersede!`'s docstring), the cached box is never
recomputed. That's safe rather than stale-and-wrong, because subdividing a
piece can only shrink (or preserve) its own contribution to the union --
never grow it beyond what it started as -- so every consumer here only
ever needs "never an underestimate," which the box keeps for its whole
lifetime.
"""
mutable struct CellNode{N}
    dim::Int
    label::Label
    subcells::Vector{Int}                 # node ids of immediate boundary cells (empty for dim=0)
    point::Union{Nothing,Pt{N,Float64}}   # set only for dim=0
    curve::Union{Nothing,Quadric{N,Float64}}   # set only for a dim=1 cell that lies on a non-linear quadric
    bbox_lo::Pt{N,Float64}
    bbox_hi::Pt{N,Float64}
end

"""
The output complex under construction: every stored cell (all dimensions
at once, per the plan -- unlike a mesh, nothing here is derivable from a
"maximal only" subset), indexed both by a stable integer id and by label
(`label_index`, supporting more than one disconnected piece per label).

`referenced_by[s]` lists every currently-live node that has `s` as an
immediate subcell -- the reverse of `nodes[p].subcells` -- maintained so
that when `s` is superseded, every one of its parents can be patched
immediately (see `supersede!`). `superseded_by[old]` records what a dead
node was replaced by, kept (rather than discarded) so any code that still
holds an old id from before a supersession can resolve it via `resolve`.

This keeps the complex "locally monotone": once a cell is created its
label and subcells are always either exactly right or explicitly marked
as superseded with a resolvable replacement, never silently stale --
adjacent cells sharing boundary structure (the normal case once
multi-point insertion is in play, not just the single-hyperplane-into-a-
pristine-bbox case) can be patched in place rather than requiring a
separate reconciliation pass after the fact.
"""
mutable struct CellComplex{N}
    nodes::Vector{CellNode{N}}
    label_index::Dict{Label,Vector{Int}}
    referenced_by::Dict{Int,Vector{Int}}
    superseded_by::Dict{Int,Vector{Int}}
end

CellComplex{N}() where {N} = CellComplex{N}(CellNode{N}[], Dict{Label,Vector{Int}}(), Dict{Int,Vector{Int}}(), Dict{Int,Vector{Int}}())

"""
Expands the straight-chord box `(lo, hi)` to also cover the true extent of
`curve`'s own arc between two points `p1`, `p2` known to lie on it -- a
genuinely curved edge (a parabolic bisector, in this codebase's only
curved case so far) can bulge outside the chord between its endpoints, so
the chord's own box alone can *underestimate* the cell's true extent.

A coordinate is extremal along an implicit curve `Q(x,y)=0` exactly where
the *other* coordinate's partial derivative vanishes (check against a
circle `Q=x²+y²-1`: `∂Q/∂y=2y=0` gives `y=0`, i.e. `x=±1` -- its true
x-extrema). Each candidate is found by substituting that linear constraint
into `Q=0` (a quadratic in the remaining free coordinate) via
`quadratic_roots`, and kept only if it actually falls on the p1-p2 arc --
not merely somewhere else on the same conic -- checked via `curve`'s own
natural single-valued axis (`curve_natural_axis`, `plot2d.jl`), not a raw
x/y coordinate range: an earlier version of this check used whichever
coordinate varies more between `p1` and `p2` (matching `tessellate_curve`'s
own former convention), which rejects a genuine candidate whenever the
arc's own turning point falls between `p1` and `p2` -- confirmed as a real
bug in the sibling functions that used the same convention (a
self-intersecting rendered polygon, and a "0 crossings" construction
failure), so fixed here the same way even though no concrete repro forced
the issue here specifically. That check deliberately is *not* "is the
extremal value itself within the endpoints' range" -- the whole point of a
real bulge is that its extremal coordinate lies *outside* that range;
it's the arc's own natural-axis projection that must stay inside it.
"""
function expand_bbox_for_curve(lo::Pt{2,Float64}, hi::Pt{2,Float64}, curve::Quadric{2,Float64}, p1::Pt{2,Float64}, p2::Pt{2,Float64})
    M, b = curve.M, curve.b
    axis = curve_natural_axis(curve)
    dom_lo, dom_hi = minmax(dot(axis, p1), dot(axis, p2))
    dom_tol = 1e-9 * max(1.0, dom_hi - dom_lo)
    accept(x, y) = let dom = dot(axis, SVector(x, y))
        dom_lo - dom_tol <= dom <= dom_hi + dom_tol
    end
    for row in (1, 2)   # row=1: ∂Q/∂x=0 constraint -> y-extrema; row=2: ∂Q/∂y=0 constraint -> x-extrema
        p, q, r = M[row, 1], M[row, 2], b[row]
        max(abs(p), abs(q)) < 1e-14 && continue   # this partial derivative is a nonzero constant -- no extremum this direction
        if abs(q) >= abs(p)
            α, β = -p / q, -r / q   # y = α*x + β
            A = M[1, 1] + 2 * M[1, 2] * α + M[2, 2] * α^2
            B = 2 * (M[1, 2] * β + M[2, 2] * α * β + b[1] + b[2] * α)
            C = M[2, 2] * β^2 + 2 * b[2] * β + curve.c
            for x in quadratic_roots(A, B, C)
                y = α * x + β
                accept(x, y) || continue
                lo, hi = min.(lo, SVector(x, y)), max.(hi, SVector(x, y))
            end
        else
            α, β = -q / p, -r / p   # x = α*y + β
            A = M[2, 2] + 2 * M[1, 2] * α + M[1, 1] * α^2
            B = 2 * (M[1, 2] * β + M[1, 1] * α * β + b[2] + b[1] * α)
            C = M[1, 1] * β^2 + 2 * b[1] * β + curve.c
            for y in quadratic_roots(A, B, C)
                x = α * y + β
                accept(x, y) || continue
                lo, hi = min.(lo, SVector(x, y)), max.(hi, SVector(x, y))
            end
        end
    end
    return lo, hi
end

"""
The bounding box of a brand-new node with the given `dim`/`subcells`/
`point`/`curve`, computed bottom-up from its already-cached-at-creation
subcells (`dim==0` is the base case: a single point). Curved-edge extrema
(see `expand_bbox_for_curve`) are only handled at `dim==1, N==2`, this
codebase's only case with genuinely curved bisectors so far, where the
reasoning for why the plain subcell-union box would otherwise
*underestimate* is concrete (a curved edge can bulge outside the chord
between its two endpoints) and the fix is exact. A hypothetical future
curved cell of dimension >= 2 (a curved *surface*, e.g. at N=3) is a
genuinely different question -- its boundary's own box does not obviously
bound its interior sag the way two endpoints bound a 1D arc -- and falls
back to the untightened subcell-union box here unexamined; don't assume
that's still conservative without rechecking when that case is actually
built (G3, task #21, hasn't reached it yet).
"""
function new_cell_bbox(cx::CellComplex{N}, dim::Int, subcells::Vector{Int}, point, curve) where {N}
    dim == 0 && return point, point
    lo = reduce((a, b) -> min.(a, b), (cx.nodes[s].bbox_lo for s in subcells))
    hi = reduce((a, b) -> max.(a, b), (cx.nodes[s].bbox_hi for s in subcells))
    if dim == 1 && curve !== nothing && N == 2
        p1, p2 = cx.nodes[subcells[1]].point, cx.nodes[subcells[2]].point
        lo, hi = expand_bbox_for_curve(lo, hi, curve, p1, p2)
    end
    return lo, hi
end

function add_cell!(cx::CellComplex{N}, dim::Int, label::Label, subcells::Vector{Int}, point=nothing; curve=nothing) where {N}
    bbox_lo, bbox_hi = new_cell_bbox(cx, dim, subcells, point, curve)
    push!(cx.nodes, CellNode{N}(dim, label, subcells, point, curve, bbox_lo, bbox_hi))
    id = length(cx.nodes)
    push!(get!(() -> Int[], cx.label_index, label), id)
    for s in subcells
        push!(get!(() -> Int[], cx.referenced_by, s), id)
    end
    return id
end

"""
Update `id`'s label (e.g. once it's known which input point(s) win there),
keeping `label_index` consistent -- removing `id` from its old label's
bucket (initially empty, for a freshly-`init_bbox_complex`'d node) and
adding it to the new one.
"""
function set_label!(cx::CellComplex, id::Int, label::Label)
    node = cx.nodes[id]
    old = node.label
    if old != label
        bucket = get(cx.label_index, old, nothing)
        bucket !== nothing && filter!(!=(id), bucket)
        node.label = label
        push!(get!(() -> Int[], cx.label_index, label), id)
    end
    return nothing
end

"""
Mark `old_id` as superseded by `new_ids` (e.g. because it was just split by
a new bisector), and immediately patch every currently-live cell that
referenced it as a subcell: `old_id` is replaced in-place by `new_ids` in
each parent's `subcells` list (and `referenced_by` updated to match).
`old_id` is also removed from its label's `label_index` bucket -- it's no
longer a live cell, just a resolvable historical reference.

This is the one place stale-reference propagation happens; every other
piece of code can assume `nodes[p].subcells` is always current.
"""
function supersede!(cx::CellComplex, old_id::Int, new_ids::Vector{Int})
    cx.superseded_by[old_id] = new_ids

    old_label = cx.nodes[old_id].label
    bucket = get(cx.label_index, old_label, nothing)
    bucket !== nothing && filter!(!=(old_id), bucket)

    parents = get(cx.referenced_by, old_id, Int[])
    for p in parents
        pnode = cx.nodes[p]
        idx = findfirst(==(old_id), pnode.subcells)
        idx === nothing && continue   # already patched (shouldn't happen, but be defensive)
        splice!(pnode.subcells, idx, new_ids)
        for nid in new_ids
            push!(get!(() -> Int[], cx.referenced_by, nid), p)
        end
    end
    delete!(cx.referenced_by, old_id)
    return nothing
end

"""
Resolve `id` to the currently-live node id(s) it corresponds to: `[id]` if
it was never superseded, or the (recursively resolved) replacement ids
otherwise. Needed by any code that might be holding an id captured before
a later supersession -- everything reached purely through `nodes[p].subcells`
is already current thanks to `supersede!`'s eager patching, so this is
mainly for callers holding onto an id across some other boundary (e.g.
across a whole `insert_point!` call).
"""
function resolve(cx::CellComplex, id::Int)
    haskey(cx.superseded_by, id) || return [id]
    out = Int[]
    for r in cx.superseded_by[id]
        append!(out, resolve(cx, r))
    end
    return out
end

"""
The cells `label` bounds -- free, per the plan: enumerable directly as the
stored cells whose label is a (not necessarily proper) subset of `label`,
no index required.
"""
function supercells(cx::CellComplex, label::Label)
    out = Label[]
    n = length(label)
    elems = collect(label)
    for mask in 0:(2^n-1)
        sub = Set(elems[i] for i in 1:n if (mask >> (i - 1)) & 1 == 1)
        haskey(cx.label_index, sub) && push!(out, sub)
    end
    return out
end

"""
All descendant 0-cells (actual points) of node `id`, found by walking the
`subcells` tree down to its leaves. For a flat (G1, all-hyperplane) cell
this is exactly its vertex set -- enough to determine how a new hyperplane
crosses it.
"""
function descendant_points(cx::CellComplex{N}, id::Int) where {N}
    node = cx.nodes[id]
    node.dim == 0 && return [node.point]
    pts = Pt{N,Float64}[]
    seen = Set{Int}()
    stack = [id]
    while !isempty(stack)
        cur = pop!(stack)
        cur in seen && continue
        push!(seen, cur)
        n = cx.nodes[cur]
        if n.dim == 0
            push!(pts, n.point)
        else
            append!(stack, n.subcells)
        end
    end
    return pts
end
