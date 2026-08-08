"""
Read-only "does the complex actually partition the box, no gaps or
overlaps" check -- different in kind from vertex-level cross-validation
(which only checks labels at discrete points). Checks *topology*,
everywhere, via a discrete sweep.

Sweeps a hyperplane `x[axis]=t` across the box. The cross-section's
combinatorial structure only changes at finitely many critical `t` (a
live vertex's own axis coordinate), so one representative `t` per gap
between consecutive critical values is exhaustive, not a sample. At each
slice, checks Euler's formula for a connected planar graph (`V-E+F==2`):
`V` = edges crossing `t` once, `E` = faces crossed exactly twice, `F` =
3-cells whose crossed faces form one closed cycle, plus 1 (outside face).
A gap or overlap unbalances this count.

Reuses the same primitives real construction trusts, against a synthetic
flat sweep-plane "bisector", read-only. Inherits construction's own
"no already-curved face-splitting" ceiling, so only certifies the `N=3`
configurations actually buildable right now.

An `N=2` method below sweeps a line instead of a plane, one dimension
down and simpler (no intermediate face dimension).

Returns a `Vector{String}` of issues (empty means clean) -- doesn't throw
on a violation, but does propagate a genuine scope error.
"""
function sweep_topology_check(cx::CellComplex{3}, lo::Pt{3,Float64}, hi::Pt{3,Float64}; axis::Int=1)
    issues = String[]
    live_ids(dim) = (id for id in eachindex(cx.nodes) if !haskey(cx.superseded_by, id) && cx.nodes[id].dim == dim)

    crit = sort(unique(vcat([lo[axis], hi[axis]], [cx.nodes[id].point[axis] for id in live_ids(0)])))
    for i in 1:length(crit)-1
        # Skip a gap so tiny its own midpoint would round to an endpoint
        # (e.g. a vertex almost exactly on the box wall) -- not a genuine
        # combinatorially-distinct slice; the wall is checked from the
        # next genuine gap inward instead.
        crit[i+1] - crit[i] < 1e-9 * max(1.0, hi[axis] - lo[axis]) && continue
        t = (crit[i] + crit[i+1]) / 2

        face_crossings = Dict{Int,Tuple{Int,Int}}()
        for f in live_ids(2)
            fnode = cx.nodes[f]
            (fnode.bbox_lo[axis] < t < fnode.bbox_hi[axis]) || continue
            crossed = try
                face_axis_crossings(cx, f, axis, t)
            catch e
                push!(issues, "slice axis=$axis t=$t: face $f: $(sprint(showerror, e))")
                continue
            end
            isempty(crossed) && continue
            if length(crossed) != 2
                push!(issues, "slice axis=$axis t=$t: face $f crossed $(length(crossed)) times, expected 0 or 2")
                continue
            end
            face_crossings[f] = (crossed[1], crossed[2])
        end

        n_active_cells = 0
        for c in live_ids(3)
            cnode = cx.nodes[c]
            (cnode.bbox_lo[axis] < t < cnode.bbox_hi[axis]) || continue
            items = Tuple{Int,Int,Int}[]
            for f in cnode.subcells
                haskey(face_crossings, f) || continue
                e1, e2 = face_crossings[f]
                push!(items, (f, e1, e2))
            end
            isempty(items) && continue
            loops = try
                group_into_cycles(items)
            catch e
                push!(issues, "slice axis=$axis t=$t: cell $c: $(sprint(showerror, e))")
                continue
            end
            if length(loops) != 1 || sum(length, loops) != length(items)
                push!(issues, "slice axis=$axis t=$t: cell $c's own active faces form $(length(loops)) cycles from $(length(items)) faces, not one")
                continue
            end
            n_active_cells += 1
        end

        V = length(Set(e for (e1, e2) in values(face_crossings) for e in (e1, e2)))
        E = length(face_crossings)
        F = n_active_cells + 1
        if V - E + F != 2
            push!(issues, "slice axis=$axis t=$t: Euler check failed: V=$V E=$E F=$F (V-E+F=$(V - E + F), expected 2) -- a gap, an overlap, or a disconnection")
        end
    end
    return issues
end

"""Synthetic flat "bisector" for `x[axis]=t`: `M=0`, `b=0.5 e_axis`, `c=-t`."""
function sweep_hyperplane_quadric(::Val{N}, axis::Int, t::Float64) where {N}
    b = SVector(ntuple(i -> i == axis ? 0.5 : 0.0, N))
    return Quadric{N,Float64}(zero(SMatrix{N,N,Float64}), b, -t)
end
sweep_plane_quadric(axis::Int, t::Float64) = sweep_hyperplane_quadric(Val(3), axis, t)

"""
How many times sweep plane `x[axis]=t` crosses live edge `edge_id` (0 or
1; more is a genuine anomaly reported by the caller). A straight edge
needs only its two endpoints' axis coordinates; a curved one is located
via its flat neighboring face's own frame.
"""
function edge_axis_crossing_count(cx::CellComplex{3}, edge_id::Int, axis::Int, t::Float64)
    enode = cx.nodes[edge_id]
    p1id, p2id = enode.subcells
    p1, p2 = cx.nodes[p1id].point, cx.nodes[p2id].point
    if enode.curve === nothing
        return (p1[axis] - t) * (p2[axis] - t) < 0 ? 1 : 0
    end
    flat_face = find_flat_neighbor_face(cx, edge_id)
    origin, e1, e2 = flat_face_frame_cached!(cx, flat_face)
    quad2 = restrict_to_plane(sweep_plane_quadric(axis, t), origin, e1, e2)
    edge_curve2 = restrict_to_plane(enode.curve, origin, e1, e2)
    to_local(p) = SVector(dot(p - origin, e1), dot(p - origin, e2))
    return length(edge_curve_crossings(quad2, to_local(p1), to_local(p2), edge_curve2))
end

"""
Which of face `face_id`'s own boundary edges sweep plane `x[axis]=t`
crosses -- 0 or 2 in this check's scope (single connected cut, same as
`clip_flat_face_3d!`'s v1 limit).
"""
function face_axis_crossings(cx::CellComplex{3}, face_id::Int, axis::Int, t::Float64)
    fnode = cx.nodes[face_id]
    floops = cyclic_boundary_walks(cx, fnode.subcells)
    length(floops) == 1 || error("face_axis_crossings: face $face_id has $(length(floops)) boundary loops -- holes aren't supported yet (v1 scope)")
    boundary_edge_ids = [e for (e, _, _) in floops[1]]
    length(boundary_edge_ids) == length(Set(boundary_edge_ids)) ||
        error("face_axis_crossings: face $face_id's own boundary walk repeats an edge -- a degenerate zero-area face (this check can't take a cross-section of something with no area)")
    crossed = Int[]
    for (e, _, _) in floops[1]
        n = edge_axis_crossing_count(cx, e, axis, t)
        n <= 1 || error("face_axis_crossings: edge $e crosses the sweep plane more than once -- not supported by this check (v1 scope: single connected cut)")
        n == 1 && push!(crossed, e)
    end
    length(crossed) in (0, 2) ||
        error("face_axis_crossings: face $face_id crossed $(length(crossed)) times, expected 0 or 2 (v1 scope: single connected cut)")
    return crossed
end

"""
Groups abstract "edges" (`(id, v1, v2)`, endpoints any hashable identity,
not necessarily real `cx` vertex ids) into maximal simple cycles by
shared endpoints -- used on the *abstract* cross-sectional graph (active
faces as "edges", crossed edges as "vertices"). Unlike
`cyclic_boundary_walks`, doesn't prune dangling pieces first: every
endpoint must have degree exactly 2 or it errors immediately, since a
well-formed closed 3-cell's crossed faces should always pair up.
"""
function group_into_cycles(items::Vector{Tuple{Int,Int,Int}})
    degree = Dict{Int,Int}()
    for (_, v1, v2) in items
        degree[v1] = get(degree, v1, 0) + 1
        degree[v2] = get(degree, v2, 0) + 1
    end
    all(d -> d == 2, values(degree)) ||
        error("group_into_cycles: not every cross-sectional vertex has degree exactly 2 -- not a union of simple closed curves")

    n = length(items)
    used = falses(n)
    loops = Vector{Int}[]
    for start in 1:n
        used[start] && continue
        used[start] = true
        order = [items[start][1]]
        entry_vertex = items[start][2]
        cur_vertex = items[start][3]
        while cur_vertex != entry_vertex
            found = 0
            for i in 1:n
                used[i] && continue
                _, a, b = items[i]
                if a == cur_vertex || b == cur_vertex
                    found = i
                    break
                end
            end
            found == 0 && error("group_into_cycles: items don't form a union of simple cycles")
            used[found] = true
            push!(order, items[found][1])
            _, a, b = items[found]
            cur_vertex = (a == cur_vertex) ? b : a
        end
        push!(loops, order)
    end
    return loops
end

"""
`N=2` counterpart, one dimension down: sweeps a line rather than a plane.
An edge's cross-section is a point; a top cell's is a set of disjoint
intervals bounded by such points.

`N=2` cells built from segment features can be genuinely non-convex
(multiply-connected/annular cells are supported), so a cell's boundary
can legitimately be crossed any even number of times, not just 0 or 2
(confirmed: a plain two-segment case crosses 4 times). A cell with `2k`
crossings contributes `k` intervals.

No intermediate face dimension at `N=2`, so the check is simpler than the
planar Euler formula: a connected subdivision into `F` intervals (summed
across all cells) has exactly `F+1` boundary points -- checked as
`V == F+1`. Also no face-frame indirection needed: an edge's own `curve`
field fully determines its shape.
"""
function sweep_topology_check(cx::CellComplex{2}, lo::Pt{2,Float64}, hi::Pt{2,Float64}; axis::Int=1)
    issues = String[]
    live_ids(dim) = (id for id in eachindex(cx.nodes) if !haskey(cx.superseded_by, id) && cx.nodes[id].dim == dim)

    crit = sort(unique(vcat([lo[axis], hi[axis]], [cx.nodes[id].point[axis] for id in live_ids(0)])))
    for i in 1:length(crit)-1
        crit[i+1] - crit[i] < 1e-9 * max(1.0, hi[axis] - lo[axis]) && continue
        t = (crit[i] + crit[i+1]) / 2
        quad = sweep_hyperplane_quadric(Val(2), axis, t)

        crossed_edges_global = Set{Int}()
        total_intervals = 0
        for c in live_ids(2)
            cnode = cx.nodes[c]
            (cnode.bbox_lo[axis] < t < cnode.bbox_hi[axis]) || continue
            crossed = try
                cell_axis_crossings_2d(cx, c, quad)
            catch e
                push!(issues, "slice axis=$axis t=$t: cell $c: $(sprint(showerror, e))")
                continue
            end
            isempty(crossed) && continue
            union!(crossed_edges_global, crossed)
            total_intervals += length(crossed) ÷ 2
        end

        V = length(crossed_edges_global)
        F = total_intervals
        if V != F + 1
            push!(issues, "slice axis=$axis t=$t: interval check failed: V=$V F=$F (expected V=F+1) -- a gap, an overlap, or a disconnection")
        end
    end
    return issues
end

"""
Which of top cell `cell_id`'s own boundary edges sweep line `quad`
crosses -- always even for a valid closed boundary (0, 2, 4, ...; more
than 2 is legitimate for a non-convex segment-feature cell, not an error).
"""
function cell_axis_crossings_2d(cx::CellComplex{2}, cell_id::Int, quad::Quadric{2,Float64})
    cnode = cx.nodes[cell_id]
    crossed = Int[]
    for e in cnode.subcells
        enode = cx.nodes[e]
        p1, p2 = cx.nodes[enode.subcells[1]].point, cx.nodes[enode.subcells[2]].point
        n = length(edge_curve_crossings(quad, p1, p2, enode.curve))
        n <= 1 || error("cell_axis_crossings_2d: edge $e crosses the sweep line more than once -- a single edge's own boundary being crossed twice by one sweep line isn't supported by this check")
        n == 1 && push!(crossed, e)
    end
    iseven(length(crossed)) ||
        error("cell_axis_crossings_2d: cell $cell_id crossed an odd number of times ($(length(crossed))) -- not a valid closed boundary")
    return crossed
end
