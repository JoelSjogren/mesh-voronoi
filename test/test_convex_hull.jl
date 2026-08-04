@testset "convex_hull_2d: square with interior/collinear-edge points" begin
    pts = [
        SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0),
        SVector(0.5, 0.5),   # interior, must be dropped
        SVector(0.5, 0.0),   # on an edge, must be dropped
    ]
    hull = convex_hull_2d(pts)
    @test length(hull) == 4
    @test Set(hull) == Set([SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0)])
    # counter-clockwise: signed area positive
    area = sum(hull[i][1] * hull[mod1(i + 1, 4)][2] - hull[mod1(i + 1, 4)][1] * hull[i][2] for i in 1:4) / 2
    @test area > 0
end

@testset "convex_hull_2d: errors on collinear input" begin
    pts = [SVector(0.0, 0.0), SVector(1.0, 1.0), SVector(2.0, 2.0)]
    @test_throws ErrorException convex_hull_2d(pts)
end

@testset "offset_polygon: square offsets to a concentric larger square" begin
    hull = convex_hull_2d([SVector(0.0, 0.0), SVector(2.0, 0.0), SVector(2.0, 2.0), SVector(0.0, 2.0)])
    off = offset_polygon(hull, 1.0)
    @test length(off) == 4
    # each vertex should move directly outward along the diagonal from the center (1,1)
    center = SVector(1.0, 1.0)
    for (h, o) in zip(hull, off)
        dir_h = (h - center) / norm(h - center)
        dir_o = (o - center) / norm(o - center)
        @test dir_h ≈ dir_o atol = 1e-9
        @test norm(o - center) ≈ norm(h - center) + 1.0 * sqrt(2) atol = 1e-9
    end
end

@testset "offset_polygon: the report's own counterexample -- naive scaling misplaces a vertex, offsetting doesn't" begin
    # V1=(0,0), V2=(1,0.01), V3=(-100,100): a valid, non-degenerate (affinely
    # independent) triangle. Scaling rigidly about an interior origin near V1
    # sends the scaled V1 in direction (-1,0)-ish, but V3 actually maximizes
    # u.V for that direction (u.V3=100 vs u.V1=0), i.e. naive scaling puts
    # "V1's own vertex" outside V1's own normal cone. The offset construction
    # must not have this failure: its own new vertex near V1 must lie in a
    # region where V1 (not V3) is the true asymptotic winner.
    V1, V2, V3 = SVector(0.0, 0.0), SVector(1.0, 0.01), SVector(-100.0, 100.0)
    hull = convex_hull_2d([V1, V2, V3])
    @test length(hull) == 3

    # Confirm the naive-scaling failure mode itself, as a sanity check that
    # this really is the counterexample described (not a stale claim): scale
    # the hull about an interior point O near V1, and check whether the
    # scaled-V1 direction actually still favors V1.
    O = SVector(0.1, 0.001)
    @test any(v -> v ≈ V1, hull)
    u = (V1 - O) / norm(V1 - O)
    winner = hull[argmax([dot(u, v) for v in hull])]
    @test winner != V1   # confirms the naive approach really does fail here
    @test winner == V3

    # Now the offset construction: the new vertex "belonging to" V1 must
    # itself lie in a direction (from the hull's own centroid, any fixed
    # interior reference point) where V1 -- not V3 -- is the true asymptotic
    # winner (argmax of u.v over the hull's own vertices).
    off = offset_polygon(hull, 1e6)
    i1 = findfirst(v -> v ≈ V1, hull)
    new_v1 = off[i1]
    centroid = sum(hull) / 3
    u_off = (new_v1 - centroid) / norm(new_v1 - centroid)
    offset_winner = hull[argmax([dot(u_off, v) for v in hull])]
    @test offset_winner == V1
end

@testset "offset_polygon: errors on a degenerate (too-few-vertex) hull" begin
    @test_throws ErrorException offset_polygon([SVector(0.0, 0.0), SVector(1.0, 0.0)], 1.0)
end
