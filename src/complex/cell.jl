const VertexIdx = Int

"""A cell's label: the set of input sub-simplices tied there."""
const Label = Set{Set{VertexIdx}}

"""
One stored cell of a `CellComplex{N}`: its dimension, label, and the ids
of its immediate *subcells* (boundary pieces). A cell's shape *is* the
union of its subcells' shapes; the only cells with independent geometric
data beyond that are 0-cells (a point) and dim=1 cells on a genuinely
curved bisector (`curve`, `nothing` for a straight edge).

A cell's own bounding box is part of its identity, not derived: two live
cells sharing a label are only genuinely "the same place" if their boxes
are disjoint from every other same-label cell (`assert_label_bbox_invariant`).
Computed once, bottom-up, at creation (`add_cell!`) -- an O(1) lookup
everywhere it's needed instead of an O(subtree) walk.

The box is conservative, not always tight: if a later insertion subdivides
one of this cell's subcells via `supersede!`, the cached box is never
recomputed -- safe because subdividing a piece can only shrink its own
contribution to the union, never grow it.
"""
mutable struct CellNode{N}
    dim::Int
    label::Label
    subcells::Vector{Int}                 # node ids of immediate boundary cells (empty for dim=0)
    point::Union{Nothing,Pt{N,Float64}}   # set only for dim=0
    curve::Union{Nothing,Quadric{N,Float64}}   # set only for a dim=1 cell that lies on a non-linear quadric
    # `N=3` dim=1 edge confined to *two* independent curved surfaces at
    # once (a genuine space curve, unlike every other curved edge here,
    # which is confined to one quadric plus one flat plane). `ruled_frame`/
    # `ruled_trace_crossing` (clip.jl) use whichever of Q1/Q2 is rank<=2
    # and indefinite to parametrize the edge and locate a third quadric's
    # crossing along it.
    curve2::Union{Nothing,Quadric{N,Float64}}
    bbox_lo::Pt{N,Float64}
    bbox_hi::Pt{N,Float64}
    # Set only for a dim=0 vertex `clip_by_hyperplane!`'s vertex loop has
    # found *exactly* tied (exact_sign returned a genuine 0) before
    # symbolic_tiebreak picked a side -- the first-class record of the tie
    # the resolution would otherwise discard. `nothing` (not an empty
    # Label) in the common untied case, to stay allocation-free.
    exact_ties::Union{Nothing,Label}
end

"""Record `atom` as exactly tied at vertex `node` -- see `CellNode.exact_ties`."""
function record_exact_tie!(node::CellNode, atom::Set{VertexIdx})
    ties = node.exact_ties === nothing ? Label() : node.exact_ties
    push!(ties, atom)
    node.exact_ties = ties
    return nothing
end

"""
The output complex under construction: every stored cell (all dimensions
at once), indexed both by stable integer id and by label (`label_index`,
supporting more than one disconnected piece per label).

`referenced_by[s]` lists every live node with `s` as an immediate subcell
(the reverse of `nodes[p].subcells`), so when `s` is superseded, every
parent can be patched immediately (`supersede!`). `superseded_by[old]`
records what a dead node was replaced by, so a caller holding an old id
can resolve it via `resolve`.

Keeps the complex "locally monotone": a cell's label/subcells are always
either exactly right or explicitly superseded with a resolvable
replacement, never silently stale.
"""
mutable struct CellComplex{N}
    nodes::Vector{CellNode{N}}
    label_index::Dict{Label,Vector{Int}}
    referenced_by::Dict{Int,Vector{Int}}
    superseded_by::Dict{Int,Vector{Int}}
    # (origin, e1, e2) local frame of a flat N=3 face, keyed by node id --
    # populated lazily by `flat_face_frame_cached!`, propagated to
    # children when split (same plane both pieces). Needed because a
    # two-vertex "bigon" piece (one straight edge, one curved trace edge)
    # can't pin down a 2D frame from its own boundary points alone.
    face_frames::Dict{Int,NTuple{3,Pt{N,Float64}}}
end

CellComplex{N}() where {N} = CellComplex{N}(CellNode{N}[], Dict{Label,Vector{Int}}(), Dict{Int,Vector{Int}}(), Dict{Int,Vector{Int}}(), Dict{Int,NTuple{3,Pt{N,Float64}}}())

"""All feature atoms ever recorded exactly tied at `id` (`nothing` if none)."""
exact_ties(cx::CellComplex, id::Int) = cx.nodes[id].exact_ties

"""
Expands chord box `(lo,hi)` to cover the true extent of `curve`'s own arc
between `p1`,`p2` -- a curved edge can bulge outside its chord, so the
chord's box alone can underestimate. A coordinate is extremal along
`Q(x,y)=0` where the *other* coordinate's partial derivative vanishes;
each candidate is kept only if it falls on the p1-p2 arc, checked via
`curve`'s own natural axis (`curve_natural_axis`) rather than raw x/y
range -- using whichever coordinate varies more between `p1`,`p2` instead
rejects a genuine candidate whenever the arc's turning point falls between
them (a real bug once found in sibling functions using that convention).
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
Bounding box of a brand-new node, computed bottom-up from its subcells
(`dim==0`: a single point). Curved-edge extrema (`expand_bbox_for_curve`)
are only handled at `dim==1, N==2` -- a curved *surface* (dim>=2, e.g.
N=3) falls back to the untightened subcell-union box unexamined; its own
boundary box doesn't obviously bound its interior sag the way two
endpoints bound a 1D arc, so don't assume this is still conservative
without rechecking.
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

function add_cell!(cx::CellComplex{N}, dim::Int, label::Label, subcells::Vector{Int}, point=nothing; curve=nothing, curve2=nothing) where {N}
    bbox_lo, bbox_hi = new_cell_bbox(cx, dim, subcells, point, curve)
    push!(cx.nodes, CellNode{N}(dim, label, subcells, point, curve, curve2, bbox_lo, bbox_hi, nothing))
    id = length(cx.nodes)
    push!(get!(() -> Int[], cx.label_index, label), id)
    for s in subcells
        push!(get!(() -> Int[], cx.referenced_by, s), id)
    end
    return id
end

"""Update `id`'s label, keeping `label_index` consistent."""
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
Mark `old_id` as superseded by `new_ids`, and immediately patch every
live cell that referenced it as a subcell in place. `old_id` is removed
from `label_index` -- no longer live, just a resolvable historical
reference. The one place stale-reference propagation happens; everywhere
else can assume `nodes[p].subcells` is always current.
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
        # A `new_ids` entry can coincide with something `pnode` already
        # references independently (e.g. `weld_duplicate_edges!` merging
        # two separately-created edges that turn out to be the same
        # geometric edge into one canonical id) -- left alone, the splice
        # above leaves the same subcell referenced twice. Only collapsed
        # at dim>=2, where `subcells` is an unordered set of boundary
        # pieces (never legitimately repeated) -- a dim=1 edge's own
        # `subcells` is its two *positional* endpoints, where a repeat
        # means something different (a zero-length weld edge) and must
        # not be silently collapsed.
        if pnode.dim >= 2 && length(pnode.subcells) != length(Set(pnode.subcells))
            unique!(pnode.subcells)
        end
        for nid in new_ids
            refs = get!(() -> Int[], cx.referenced_by, nid)
            p in refs || push!(refs, p)
        end
    end
    delete!(cx.referenced_by, old_id)
    return nothing
end

"""
Resolve `id` to its currently-live node id(s): `[id]` if never superseded,
or the recursively-resolved replacements otherwise -- for callers holding
an id captured before a later supersession.
"""
function resolve(cx::CellComplex, id::Int)
    haskey(cx.superseded_by, id) || return [id]
    out = Int[]
    for r in cx.superseded_by[id]
        append!(out, resolve(cx, r))
    end
    return out
end

"""The cells `label` bounds: stored cells whose label is a subset of `label`."""
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

"""All descendant 0-cells (points) of node `id`, walking `subcells` to its leaves."""
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
