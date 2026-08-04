@testset "compactified_points_complex: cross-validation against the brute-force oracle" begin
    points = [
        SVector(0.0, 0.0), SVector(4.0, -1.0), SVector(6.0, 2.0), SVector(3.5, 4.5),
        SVector(0.5, 3.5), SVector(-2.0, 1.0), SVector(2.0, 1.5), SVector(3.0, 2.0),
    ]
    cx, hull, offset = compactified_points_complex(points)
    @test length(hull) == 6   # points 7,8 sit strictly interior, not on the hull

    lo = SVector(minimum(p[1] for p in offset), minimum(p[2] for p in offset))
    hi = SVector(maximum(p[1] for p in offset), maximum(p[2] for p in offset))
    checked = 0
    mismatches = 0
    for x in range(0.95 * lo[1], 0.95 * hi[1], length=61), y in range(0.95 * lo[2], 0.95 * hi[2], length=61)
        pt = SVector(x, y)
        expected = brute_force_label_points(points, pt)
        length(expected) > 1 && continue
        got = find_containing_cell(cx, pt)
        got === nothing && continue
        label = cx.nodes[got].label
        length(label) == 1 || continue
        checked += 1
        only(only(label)) != only(expected) && (mismatches += 1)
    end
    @test checked > 2000
    @test mismatches == 0
end

@testset "compactified_points_complex: hull vertices get unbounded cells, interior points don't" begin
    # Exactly the theoretical prediction the planning report works out: only
    # convex-hull vertices can have a cell reaching the offset boundary; a
    # point strictly interior to the hull is always fully enclosed.
    points = [
        SVector(0.0, 0.0), SVector(4.0, -1.0), SVector(6.0, 2.0), SVector(3.5, 4.5),
        SVector(0.5, 3.5), SVector(-2.0, 1.0), SVector(2.0, 1.5), SVector(3.0, 2.0),
    ]
    cx, hull, offset = compactified_points_complex(points)

    live_top_cells = [id for id in eachindex(cx.nodes) if !haskey(cx.superseded_by, id) &&
                       cx.nodes[id].dim == 2 && length(cx.nodes[id].label) == 1]
    @test length(live_top_cells) == length(points)   # each point merges into exactly one cell

    hull_winners = Set(findfirst(p -> p ≈ v, points) for v in hull)   # points 1..6, by construction of this example
    @test hull_winners == Set(1:6)   # points 7,8 (indices 7 and 8) are the ones strictly interior to the hull
    for id in live_top_cells
        winner = only(only(cx.nodes[id].label))
        pts = descendant_points(cx, id)
        on_boundary = any(p -> any(bv -> p ≈ bv, offset), pts)
        @test on_boundary == (winner in hull_winners)
    end
end

@testset "compactified_points_complex: boundary vertices match the true asymptotic winner" begin
    # The report's own central claim, checked directly: each offset-polygon
    # vertex's own label (after full construction) must be the single input
    # point that maximizes u.p for u = direction from the origin to that
    # boundary vertex -- the support-function characterization of "winner at
    # infinity" (Section 1 of the report), not just "looks plausible".
    points = [
        SVector(0.0, 0.0), SVector(4.0, -1.0), SVector(6.0, 2.0), SVector(3.5, 4.5),
        SVector(0.5, 3.5), SVector(-2.0, 1.0), SVector(2.0, 1.5), SVector(3.0, 2.0),
    ]
    cx, hull, offset = compactified_points_complex(points)
    for v in offset
        u = v / norm(v)
        expected = argmax([dot(u, p) for p in points])
        id = findfirst(n -> n.dim == 0 && n.point ≈ v, cx.nodes)
        @test id !== nothing
        @test only(only(cx.nodes[id].label)) == expected
    end
end
