"""
Every stored live cell's descendant points are closer (or tied) to every
label in that cell's own label than to any *other* input point -- the
core correctness property of the whole construction, checked directly
against the raw distance formula rather than trusting the construction's
own bookkeeping.
"""
function check_all_cells_correctly_sided(cx::CellComplex{N}, points::Vector{Pt{N,Float64}}) where {N}
    for id in eachindex(cx.nodes)
        haskey(cx.superseded_by, id) && continue   # a dead node's stale .label is never cleared, only label_index is
        node = cx.nodes[id]
        isempty(node.label) && continue
        winners = Set(only(f) for f in node.label)
        for p in descendant_points(cx, id)
            ds = [sum(abs2, p - pt) for pt in points]
            m = minimum(ds)
            for w in winners
                @test ds[w] <= m + 1e-6 * max(1.0, m)
            end
        end
    end
end

@testset "points_complex: N=2, three points, all cells correctly sided" begin
    points = [SVector(0.0, 0.0), SVector(2.0, 0.0), SVector(1.0, 1.8)]
    cx = points_complex(points)
    check_all_cells_correctly_sided(cx, points)
    # every input point's label should actually exist as a live cell
    for i in 1:3
        @test haskey(cx.label_index, Label([Set([i])])) && !isempty(cx.label_index[Label([Set([i])])])
    end
end

@testset "points_complex: N=2, three points, grid cross-validated against oracle" begin
    points = [SVector(0.0, 0.0), SVector(2.0, 0.0), SVector(1.0, 1.8)]
    cx = points_complex(points)
    lo, hi = padded_bbox(points)

    # collect (label, polygon) for every live top-dimensional cell
    cells = Tuple{Set{Int},Vector{SVector{2,Float64}}}[]
    for id in eachindex(cx.nodes)
        haskey(cx.superseded_by, id) && continue
        node = cx.nodes[id]
        node.dim == 2 || continue
        isempty(node.label) && continue
        winners = Set(only(f) for f in node.label)
        push!(cells, (winners, polygon_vertices_2d(cx, id)))
    end
    # NOT necessarily one 2-cell per point: processing each existing
    # winner's territory as a separate `clip_by_hyperplane!` call can
    # legitimately produce two adjacent, touching pieces that both end up
    # labeled the same winner (e.g. a new point winning parts of two
    # different existing points' territories that happen to be adjacent) --
    # the sub/supercell duality design explicitly allows a label to have
    # more than one stored piece (see `CellComplex`'s docstring). This is a
    # real representational gap worth closing later (an adjacent-same-label
    # merge pass, or recognizing the adjacency during construction), but
    # it's not a correctness bug: every sampled point below is still
    # checked against the oracle and must match exactly.
    @test length(cells) >= 3

    mismatches = 0
    checked = 0
    for x in range(lo[1], hi[1], length=41), y in range(lo[2], hi[2], length=41)
        pt = SVector(x, y)
        expected = brute_force_label_points(points, pt)
        length(expected) > 1 && continue
        in_cells = [w for (w, poly) in cells if point_in_convex_polygon(poly, pt)]
        length(in_cells) != 1 && continue   # skip ambiguous/boundary points
        checked += 1
        got = in_cells[1]
        got != expected && (mismatches += 1)
    end
    @test checked > 500
    @test mismatches == 0
end

# Known gap: `insert_point!` clips each *existing* winner's territory
# independently, one `clip_by_hyperplane!` call per pre-existing top-cell.
# When the new point wins adjacent slices carved out of two different
# existing winners' territories, those slices used to become two distinct
# cells sharing the new point's label -- geometrically adjacent (sharing a
# boundary edge) and, mathematically, one connected region, but stored as
# two. Fixed by `merge_adjacent_same_label_cells!` (called automatically
# after every insertion): this test now confirms exactly one live cell
# remains for the previously-split label, and that the invariant
# `merge_adjacent_same_label_cells!` is meant to establish -- no two live
# cells share a label with touching or overlapping bounding boxes --
# actually holds (`assert_label_bbox_invariant`, which the merge itself
# already runs internally; checked again here directly as the point of
# this specific test).
@testset "points_complex: N=2, three points -- adjacent same-label cells are merged" begin
    points = [SVector(0.0, 0.0), SVector(2.0, 0.0), SVector(1.0, 1.8)]
    cx = points_complex(points)

    label3 = Label([Set([3])])
    live_top_cells = [id for id in cx.label_index[label3] if !haskey(cx.superseded_by, id) && cx.nodes[id].dim == 2]
    @test length(live_top_cells) == 1

    @test assert_label_bbox_invariant(cx) === nothing
end

@testset "points_complex: N=2, five points, grid cross-validated" begin
    points = [SVector(0.0, 0.0), SVector(3.0, 0.0), SVector(1.3, 2.5), SVector(-1.7, 1.4), SVector(3.6, 2.1)]
    cx = points_complex(points)
    check_all_cells_correctly_sided(cx, points)
    lo, hi = padded_bbox(points)

    cells = Tuple{Set{Int},Vector{SVector{2,Float64}}}[]
    for id in eachindex(cx.nodes)
        haskey(cx.superseded_by, id) && continue
        node = cx.nodes[id]
        node.dim == 2 || continue
        isempty(node.label) && continue
        winners = Set(only(f) for f in node.label)
        push!(cells, (winners, polygon_vertices_2d(cx, id)))
    end

    mismatches = 0
    checked = 0
    for x in range(lo[1], hi[1], length=51), y in range(lo[2], hi[2], length=51)
        pt = SVector(x, y)
        expected = brute_force_label_points(points, pt)
        length(expected) > 1 && continue
        in_cells = [w for (w, poly) in cells if point_in_convex_polygon(poly, pt)]
        length(in_cells) != 1 && continue
        checked += 1
        got = in_cells[1]
        got != expected && (mismatches += 1)
    end
    @test checked > 500
    @test mismatches == 0
end

@testset "points_complex: N=3, four points, all cells correctly sided" begin
    points = [SVector(0.0, 0.0, 0.0), SVector(2.0, 0.0, 0.0), SVector(0.9, 2.1, 0.0), SVector(1.1, 1.2, 2.0)]
    cx = points_complex(points)
    check_all_cells_correctly_sided(cx, points)
    for i in 1:4
        @test haskey(cx.label_index, Label([Set([i])])) && !isempty(cx.label_index[Label([Set([i])])])
    end
end
