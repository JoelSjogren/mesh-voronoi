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

# A point-vs-point bisector is always a flat hyperplane regardless of `N`
# (the two features' quadratic parts are both `M=I`, cancelling exactly in
# their difference) -- so the whole "genuinely new step 3D needs" gap this
# project's own README flags (a *curved* bisector surface meeting the
# 2-skeleton) simply doesn't arise for points-only input. Every piece of
# the construction (`init_bbox_complex`, `insert_point!`,
# `merge_adjacent_same_label_cells!`'s own union-find, `weld_*!`) is
# already written generically over `N`, so this is the 3D analogue of G1 --
# confirmed working with no code changes at all, via a much more exhaustive
# check than the `N=2` grid-sampling tests above can do: every live
# *vertex* (an exact point of the construction's own making, not a random
# sample) is cross-validated directly against the brute-force oracle,
# across a range of point counts. There is currently no 3D analogue of
# `find_containing_cell`/`polygon_vertices_2d` to grid-sample interior
# points the way the `N=2` tests above do -- vertex-level checking is both
# what's available and, if anything, a harder test (every vertex sits
# exactly on some boundary, where getting the label wrong is easiest).
@testset "points_complex: N=3, vertex-level cross-validation across several point counts" begin
    # Fixed point sets (not live-generated random ones -- this test file has
    # no `using Random`, and hardcoded coordinates keep the test
    # deterministic and reproducible like every other test here anyway).
    point_sets = Vector{SVector{3,Float64}}[
        [SVector(-4.266, -1.508, 1.988), SVector(1.283, 4.149, -3.072), SVector(2.702, 2.805, 1.703),
            SVector(-3.323, 0.711, -0.472), SVector(-1.977, -4.986, 0.67)],
        [SVector(-4.977, 1.807, 2.353), SVector(3.067, -2.586, 3.669), SVector(1.559, -0.571, 2.612),
            SVector(2.323, 4.993, 2.445), SVector(4.888, 1.047, 0.941), SVector(-0.713, -0.183, -1.128),
            SVector(-4.855, 3.433, 3.302), SVector(2.048, 1.444, 0.774)],
        [SVector(-1.603, 4.592, 5.0), SVector(-4.079, 1.984, -0.037), SVector(4.514, 2.96, 2.965),
            SVector(0.033, -1.836, 4.158), SVector(-1.907, -2.435, 4.33), SVector(-1.476, -0.324, -2.258),
            SVector(0.483, -1.72, -4.725), SVector(-2.943, -1.681, 0.961), SVector(2.79, -1.228, 3.997),
            SVector(-1.628, 1.014, 3.462), SVector(1.527, 3.072, 4.881), SVector(-3.085, -2.124, -3.1)],
        [SVector(3.545, 4.331, 1.772), SVector(0.035, -0.901, -1.869), SVector(-2.883, 2.016, 0.433),
            SVector(2.315, -1.433, 3.557), SVector(-3.752, 2.858, -4.746), SVector(2.943, 2.795, 4.602),
            SVector(-2.818, -4.232, -3.791), SVector(1.911, 2.458, 4.019), SVector(-4.781, 4.816, 4.696),
            SVector(-2.368, 1.097, 4.014), SVector(-4.631, -0.993, 1.26), SVector(2.913, 3.176, 3.51),
            SVector(-1.553, -3.885, -0.045), SVector(0.696, 1.557, -3.135), SVector(2.944, 0.864, -2.138),
            SVector(2.2, -2.898, 1.147), SVector(0.691, 2.78, 4.957), SVector(4.51, -1.691, -3.109),
            SVector(3.299, 0.796, -2.922), SVector(-3.292, -0.727, -1.637)],
    ]
    for points in point_sets
        n = length(points)
        cx = points_complex(points)

        top_ids = top_cell_ids(cx)
        # One live top cell per input point -- if two disjoint pieces of the
        # same point's own territory were ever left unmerged (the same
        # `merge_adjacent_same_label_cells!` gap the `N=2` tests above are
        # about), this count would come out higher than `n`.
        @test length(top_ids) == n
        @test length(Set(only(only(cx.nodes[id].label)) for id in top_ids)) == n

        checked = 0
        mismatches = 0
        for (id, node) in enumerate(cx.nodes)
            haskey(cx.superseded_by, id) && continue
            node.dim == 0 || continue
            expected = brute_force_label_points(points, node.point)
            got = Set(only(f) for f in node.label)
            checked += 1
            got != expected && (mismatches += 1)
        end
        @test checked > 0
        @test mismatches == 0
    end
end

# The 3D analogue of the `N=2` "4-point square" regression test elsewhere:
# a genuine multi-way tie (here, all 8 cube corners at once, at the exact
# center) must resolve to a single, correctly-labeled vertex -- not a stale
# duplicate left behind by an incomplete weld, and not a partial label
# reflecting only one pairwise comparison.
@testset "points_complex: N=3, cube corners tie at the exact center" begin
    points = SVector{3,Float64}[]
    for x in (0.0, 4.0), y in (0.0, 4.0), z in (0.0, 4.0)
        push!(points, SVector(x, y, z))
    end
    cx = points_complex(points)
    center = SVector(2.0, 2.0, 2.0)
    centers = [id for (id, n) in enumerate(cx.nodes)
               if !haskey(cx.superseded_by, id) && n.dim == 0 && n.point ≈ center]
    @test length(centers) == 1
    @test cx.nodes[only(centers)].label == Label([Set([i]) for i in 1:8])
end
