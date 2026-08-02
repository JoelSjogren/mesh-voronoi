@testset "clip_by_hyperplane!: N=2, hand-verifiable split" begin
    lo, hi = SVector(-1.0, -1.0), SVector(3.0, 1.0)
    cx, top = init_bbox_complex(Val(2), lo, hi)
    A = AffineQuadratic(SVector(0.0, 0.0))
    B = AffineQuadratic(SVector(2.0, 0.0))
    # bisector is exactly x=1: A's side is x<1, B's side is x>1
    a_id, b_id, cut_id = clip_by_hyperplane!(cx, top, A, B, Set([1]), Set([2]))

    a_pts = descendant_points(cx, a_id)
    b_pts = descendant_points(cx, b_id)
    @test all(p -> p[1] < 1.0 + 1e-9, a_pts)
    @test all(p -> p[1] > 1.0 - 1e-9, b_pts)
    @test !isempty(a_pts)
    @test !isempty(b_pts)

    # the two original vertices left of x=1 (there are exactly 2: (-1,-1),(-1,1))
    @test count(p -> p[1] ≈ -1.0, a_pts) == 2
    @test count(p -> p[1] ≈ 3.0, b_pts) == 2

    # the cut cell (a 1-dim edge, the trace x=1 within the box) has exactly
    # 2 vertices: (1,-1) and (1,1)
    cut_pts = descendant_points(cx, cut_id)
    @test length(cut_pts) == 2
    @test all(p -> p[1] ≈ 1.0, cut_pts)
    @test Set(round.(p[2]; digits=9) for p in cut_pts) == Set([-1.0, 1.0])
end

@testset "clip_by_hyperplane!: N=3, every descendant point correctly sided" begin
    lo, hi = SVector(-2.0, -2.0, -2.0), SVector(2.0, 2.0, 2.0)
    cx, top = init_bbox_complex(Val(3), lo, hi)
    A = AffineQuadratic(SVector(-0.7, 0.3, 0.1))
    B = AffineQuadratic(SVector(0.9, -0.2, 0.4))
    a_id, b_id, cut_id = clip_by_hyperplane!(cx, top, A, B, Set([1]), Set([2]))

    for p in descendant_points(cx, a_id)
        @test sqdist(A, p) <= sqdist(B, p) + 1e-9
    end
    for p in descendant_points(cx, b_id)
        @test sqdist(B, p) <= sqdist(A, p) + 1e-9
    end
    for p in descendant_points(cx, cut_id)
        @test abs(sqdist(A, p) - sqdist(B, p)) < 1e-6
    end
    @test !isempty(descendant_points(cx, a_id))
    @test !isempty(descendant_points(cx, b_id))
    @test !isempty(descendant_points(cx, cut_id))
end

@testset "clip_by_hyperplane!: random points, N=2 and N=3, cross-validated" begin
    for N in (2, 3)
        for trial in 1:20
            lo = SVector{N,Float64}(-3.0 .* ones(N))
            hi = SVector{N,Float64}(3.0 .* ones(N))
            cx, top = init_bbox_complex(Val(N), lo, hi)
            pa = SVector{N,Float64}(randn(N))
            pb = SVector{N,Float64}(randn(N))
            norm(pa - pb) < 1e-3 && continue   # avoid the degenerate near-identical case
            A, B = AffineQuadratic(pa), AffineQuadratic(pb)
            a_id, b_id, cut_id = clip_by_hyperplane!(cx, top, A, B, Set([1]), Set([2]))
            for p in descendant_points(cx, a_id)
                @test sqdist(A, p) <= sqdist(B, p) + 1e-6
            end
            for p in descendant_points(cx, b_id)
                @test sqdist(B, p) <= sqdist(A, p) + 1e-6
            end
        end
    end
end
